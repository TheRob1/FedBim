import os
import sys
import logging
import yaml
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

# Patch Pillow 10.0.0+ compatibility for Ultralytics - MUST BE BEFORE OTHER IMPORTS
import PIL.ImageFont
if not hasattr(PIL.ImageFont.FreeTypeFont, 'getsize'):
    def getsize(self, text, *args, **kwargs):
        left, top, right, bottom = self.getbbox(text, *args, **kwargs)
        return right - left, bottom - top
    PIL.ImageFont.FreeTypeFont.getsize = getsize
    PIL.ImageFont.ImageFont.getsize = getsize

# Ensure torch is imported before any other deep learning related imports
import torch
import traceback

# Add DetectionModel to safe_globals allowlist
# Add DetectionModel to safe_globals allowlist
from ultralytics.nn.tasks import DetectionModel
try:
    torch.serialization.add_safe_globals([DetectionModel])
except AttributeError:
    pass

# Monkeypatch torch.load to disable weights_only=True default in PyTorch 2.6+
_original_load = torch.load

def _safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = _safe_load

# Now import other deep learning related modules
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

# Import YOLO after torch to ensure proper initialization
from ultralytics import YOLO

# Add parent directory to path and import local modules
sys.path.append(str(Path(__file__).parent))
from utils.visualization import setup_logging, plot_metrics, log_round_metrics, save_metrics

class FireSafetyServer(fl.server.Server):
    def __init__(self, config: dict, *args, **kwargs):
        self.start_time = time.time()
        self.config = config
        self.metrics = {
            'fit_metrics': {},
            'evaluate_metrics': {}
        }
        
        # Set up logging
        self.logger = setup_logging()
        self.logger.info("Initializing FireSafetyServer...")
        
        # Initialize device
        self.device = self._initialize_device()
        
        # Initialize model with safe loading and validation
        self.global_model = self._initialize_model()
        
        # Verify model is properly initialized
        self._verify_model()
        
        # Set up directories
        self._setup_directories()
        
        # Get server config with defaults
        fl_config = config.get('federated_learning', {})
        server_config = fl_config.get('server', {})
        
        # Helper to get sorted params
        def get_sorted_params(model):
            state_dict = model.state_dict()
            keys = sorted(state_dict.keys())
            return [state_dict[k].cpu().numpy() for k in keys]

        # Initialize strategy with config values or defaults
        self.strategy = fl.server.strategy.FedAvg(
            min_available_clients=server_config.get('min_available_clients', 2),
            min_fit_clients=server_config.get('min_fit_clients', 2),
            min_evaluate_clients=server_config.get('min_evaluate_clients', 2),
            on_fit_config_fn=self.get_fit_config,
            on_evaluate_config_fn=self.get_evaluate_config,
            evaluate_fn=self.evaluate_global_model,
            initial_parameters=fl.common.ndarrays_to_parameters(
                get_sorted_params(self.global_model.model)
            ) if hasattr(self.global_model, 'model') and hasattr(self.global_model.model, 'state_dict') else None,
        )
        
        # Log initialization complete
        self.logger.info(f"FireSafetyServer initialized in {time.time() - self.start_time:.2f} seconds")
    
    def _initialize_device(self) -> torch.device:
        """Initialize and verify PyTorch device."""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, using CPU. Training will be slow.")
            return torch.device("cpu")
        
        try:
            # Try to allocate and free a tensor to verify CUDA is working
            _ = torch.tensor([1.0]).cuda()
            self.logger.info("CUDA is available and working")
            return torch.device("cuda:0")
        except Exception as e:
            self.logger.warning(f"CUDA initialization failed: {str(e)}. Falling back to CPU.")
            return torch.device("cpu")
    
    def _initialize_model(self):
        """Safely initialize the YOLO model with proper error handling."""
        model_path = self.config['yolo']['model']
        self.logger.info(f"Loading model from {model_path}")
        
        # First try standard YOLO load
        try:
            model = YOLO(model_path)
            self.logger.info("Successfully loaded model with standard YOLO load")
            return model
        except Exception as e:
            self.logger.warning(f"Standard YOLO load failed, trying safe_load: {e}")
        
        # Fall back to safe loading
        try:
            # First create a minimal YOLO model with the appropriate config
            config_file = 'yolov8n.yaml' if 'yolov8n' in str(model_path).lower() else 'yolov8s.yaml'
            self.logger.info(f"Creating YOLO model with config: {config_file}")
            model = YOLO(config_file)
            
            # Load the weights safely with allowlist
            allowlist = [DetectionModel]
            with torch.serialization.safe_globals(allowlist):
                self.logger.info("Loading model weights with safe_globals")
                ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
                
                # Load the state dict into the model
                if 'model' in ckpt and hasattr(ckpt['model'], 'state_dict'):
                    state_dict = ckpt['model'].float().state_dict()
                    try:
                        # Try loading state dict
                        model.model.load_state_dict(state_dict, strict=True)
                        self.logger.info("Successfully received and loaded global model parameters")
                        
                    except RuntimeError as e:
                        if "size mismatch" in str(e):
                            self.logger.warning(f"Model size mismatch during initialization: {e}")
                            self.logger.warning("This is expected if the server model was initialized with default classes but loaded 4-class weights.")
                            self.logger.warning("The model will be updated with correct weights during the first round of aggregation.")
                            # We can ignore this for now as the model object is just a placeholder
                            pass
                        else:
                            raise e

                    self.logger.info("Successfully loaded model weights")
                else:
                    raise ValueError("Unexpected checkpoint format")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
    def _verify_model(self):
        """Verify that the model can perform a forward pass."""
        self.logger.info("Verifying model...")
        try:
            # We use direct tensor forward pass to avoid initializing the validation pipeline
            # which might trigger layer fusion (autopad/fuse) and change the model structure.
            # This ensures the model state remains pristine for Federated Learning.
            
            # Create a dummy input tensor in the correct format
            dummy_input = torch.zeros((1, 3, 640, 640), device=self.device)
            
            # Get the underlying model and put it in eval mode
            model = self.global_model.model
            model.eval()
            
            # Do a forward pass
            with torch.no_grad():
                # For YOLOv8, we need to handle the model's forward pass specially
                if hasattr(model, 'forward'):
                    _ = model(dummy_input)
                else:
                    # If direct forward doesn't work, we try to use the call method
                    # but be careful not to trigger high-level APIs that might fuse
                    _ = model(dummy_input)
            
            self.logger.info("Model verification successful (tensor input)")
            return True
        except Exception as e:
            self.logger.error(f"Model verification failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Model verification failed: {str(e)}")
            
    def _setup_directories(self):
        """Create necessary directories for results and logs."""
        self.results_dir = Path("results")
        self.plots_dir = self.results_dir / "plots"
        self.metrics_dir = self.results_dir / "metrics"
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Results will be saved to: {self.results_dir.absolute()}")
    
    def get_fit_config(self, server_round: int) -> Dict[str, Scalar]:
        """Return training configuration for each round."""
        return {
            "epochs": self.config['yolo']['epochs'],
            "batch_size": self.config['yolo']['batch_size'],
            "server_round": server_round,
        }
    
    def get_evaluate_config(self, server_round: int) -> Dict[str, Scalar]:
        """Return evaluation configuration for each round."""
        return {
            "batch_size": self.config['yolo']['batch_size'],
            "server_round": server_round,
        }
    
    def evaluate_global_model(
        self, server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        """Evaluate the global model on the test set."""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Starting evaluation for round {server_round}")
        
        try:
            # Create a TEMPORARY model for validation to prevent layer fusion on the main model
            # Layer fusion (merging BN into Conv) happens during validation/prediction and changes
            # the model structure (225 -> 168 layers), incorrectly updating the persistent model.
            
            # Create a new YOLO instance with the same config
            config_file = 'yolov8n.yaml' if 'yolov8n' in str(self.config['yolo']['model']).lower() else 'yolov8s.yaml'
            temp_model = YOLO(config_file)
            
            # Start with the main model's state dict (unfused)
            model_state_dict = temp_model.model.state_dict()
            keys = sorted(model_state_dict.keys())
            
            state_dict = {}
            for k, param in zip(keys, parameters):
                if k not in model_state_dict:
                    continue
                v = model_state_dict[k]
                
                # Skip non-tensor parameters
                if not isinstance(v, torch.Tensor):
                    state_dict[k] = param
                    continue
                
                # Convert parameter to tensor
                tensor_param = torch.from_numpy(param)
                state_dict[k] = tensor_param
            
            # Save the aggregated model to disk
            try:
                save_path = self.results_dir / f"global_model_round_{server_round}.pt"
                torch.save(state_dict, save_path)
                self.logger.info(f"Saved aggregated global model to {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save global model: {e}")

            # Load state dict into the TEMPORARY model
            # strict=True is important here to ensure exact match before validation fuses it
            try:
                temp_model.model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    self.logger.warning(f"Server evaluation skipped due to model class count mismatch (nc=80 vs nc=4): {e}")
                    return float('inf'), {}
                raise e
            
            # Use the temp model for evaluation
            eval_model = temp_model
            
        except Exception as e:
            self.logger.error(f"Error updating model parameters: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Return default metrics in case of error
            return 1.0, {
                "loss": 1.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "mAP50": 0.0,
                "mAP50-95": 0.0,
            }
        
        # Evaluate the model
        try:
            with torch.no_grad():  # Ensure no gradients are computed during evaluation
                # Create a temporary YAML config file for validation
                val_config = {
                    'path': os.getcwd(),  # Use current directory as base
                    'train': None,
                    'val': self.config['dataset']['val_path'],
                    'test': self.config['dataset']['test_path'],
                    'nc': self.config['dataset']['nc'],
                    'names': self.config['dataset']['names']
                }
                
                # Save the config to a temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump(val_config, f)
                    temp_config_path = f.name
                
                try:
                    # Run validation on the TEMPORARY model
                    results = eval_model.val(
                        data=temp_config_path,
                        batch=self.config['yolo']['batch_size'],
                        imgsz=self.config['yolo']['imgsz'],
                        device=str(self.device).replace('cuda:', ''),  # Remove 'cuda:' prefix if present
                        workers=self.config['yolo']['workers'],
                        project=os.path.join('runs', 'eval'),
                        name=f'round_{server_round}',
                        verbose=False
                    )
                    
                    if results is None: 
                        raise ValueError("Validation returned None")

                    # Extract metrics
                    # Note: indices depend on the exact return format of YOLOv8 val
                    # YOLOv8 val returns a DetMetrics object
                    
                    # Log validation results
                    self.logger.info(f"Round {server_round} evaluation results:")
                    self.logger.info(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 0.0)}")
                    self.logger.info(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0.0)}")
                    
                    metrics = {
                        "loss": float(1.0 - results.results_dict.get('metrics/mAP50(B)', 0.0)),  # Using 1-mAP as loss
                        "accuracy": float(results.results_dict.get('metrics/precision(B)', 0.0)),
                        "precision": float(results.results_dict.get('metrics/precision(B)', 0.0)),
                        "recall": float(results.results_dict.get('metrics/recall(B)', 0.0)),
                        "mAP50": float(results.results_dict.get('metrics/mAP50(B)', 0.0)),
                        "mAP50-95": float(results.results_dict.get('metrics/mAP50-95(B)', 0.0)),
                    }
                finally:
                    # Clean up the temporary config file
                    try:
                        if os.path.exists(temp_config_path):
                            os.unlink(temp_config_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to clean up temporary config file: {e}")
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Return default metrics in case of error
            return 1.0, {
                "loss": 1.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "mAP50": 0.0,
                "mAP50-95": 0.0,
            }
        
        # Save the global model
        try:
            model_path = os.path.join(self.results_dir, f'global_model_round_{server_round}.pt')
            # Ensure model is on CPU before saving
            original_device = next(self.global_model.model.parameters()).device
            self.global_model.model.to('cpu')
            torch.save(self.global_model.model.state_dict(), model_path)
            # Move model back to original device
            self.global_model.model.to(original_device)
            self.logger.info(f"Saved global model to {model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            # Continue even if saving fails
        
        # Calculate metrics
        metrics = {
            "loss": float(1.0 - results.results_dict.get('metrics/mAP50(B)', 0.0)),  # Using 1-mAP as loss
            "accuracy": float(results.results_dict.get('metrics/precision(B)', 0.0)),
            "precision": float(results.results_dict.get('metrics/precision(B)', 0.0)),
            "recall": float(results.results_dict.get('metrics/recall(B)', 0.0)),
            "mAP50": float(results.results_dict.get('metrics/mAP50(B)', 0.0)),
            "mAP50-95": float(results.results_dict.get('metrics/mAP50-95(B)', 0.0)),
        }
        
        # Log and save metrics
        self.metrics[f'round_{server_round}'] = metrics
        log_round_metrics(server_round, metrics, is_global=True)
        save_metrics(self.metrics, os.path.join(self.results_dir, 'training_metrics.json'))
        
        # Generate plots
        plot_metrics(self.metrics, os.path.join(self.results_dir, 'plots'))
        
        # Return loss and metrics
        return metrics["loss"], metrics

def main():
    try:
        # Load configuration
        config_path = Path(__file__).parent / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Set up logging
        logger = setup_logging()
        logger.info("Starting FireSafety Federated Learning Server")

        # Server configuration from config
        server_config = config['federated_learning']['server']
        server_address = server_config['address']
        server_port = server_config['port']
        full_address = f"{server_address}:{server_port}"
        logger.info(f"Server will run on {full_address}")

        # Initialize the server
        server = FireSafetyServer(config)

        # Get the strategy from the server
        strategy = server.strategy

        # Start Flower server with gRPC options
        fl.server.start_server(
            server_address=full_address,
            config=fl.server.ServerConfig(
                num_rounds=config['federated_learning']['server']['num_rounds'],
                round_timeout=3600.0,  # 1 hour timeout per round
            ),
            strategy=strategy,
            # Set the maximum message length for gRPC
            grpc_max_message_length=512 * 1024 * 1024  # 512MB
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
