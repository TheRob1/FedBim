import warnings
import os
import sys

# Suppress annoying deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
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

# Monkey-patch torch.load to handle newer PyTorch versions defaults
import torch
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# Add essentials to safe_globals allowlist as a backup
from ultralytics.nn.tasks import DetectionModel
import torch.nn as nn
try:
    import ultralytics.nn.modules as modules
    torch.serialization.add_safe_globals([
        DetectionModel, nn.modules.container.Sequential, nn.modules.conv.Conv2d, 
        nn.modules.batchnorm.BatchNorm2d, nn.modules.activation.SiLU
    ])
except Exception:
    pass

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
        device_config = str(self.config['yolo'].get('device', '0'))
        try:
            # If CUDA_VISIBLE_DEVICES is set by the runner, 'cuda:0' refers to that specific GPU.
            if torch.cuda.is_available() and device_config != 'cpu':
                device = torch.device("cuda:0")
                # Test CUDA to verify it's working
                _ = torch.tensor([1.0]).to(device)
                self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
            else:
                device = torch.device("cpu")
                self.logger.warning("Using CPU for training. This will be slow.")
            return device
        except Exception as e:
            self.logger.warning(f"Device initialization failed: {str(e)}. Falling back to CPU.")
            return torch.device("cpu")
    
    def _initialize_model(self):
        """Initialize the YOLO model."""
        model_path = self.config['yolo']['model']
        self.logger.info(f"Loading model from {model_path}")
        
        try:
            # For 8.0.0, loading the .pt directly is the most reliable way 
            # to ensure the architecture is consistent for FL.
            # Even if it has 80 classes, it will work with labels 0-3.
            model = YOLO(model_path)
            model.to(self.device)
            self.logger.info(f"Successfully loaded model and moved to {self.device}")
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
        self.logger.info(f"Sending fit configuration for round {server_round} to clients...")
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
            
            # Create a new YOLO instance with the same model to ensure architecture matches
            temp_model = YOLO(self.config['yolo']['model'])
            
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
            
            # Load state dict into the TEMPORARY model
            # strict=True is important here to ensure exact match before validation fuses it
            temp_model.model.load_state_dict(state_dict, strict=True)
            
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
                    'path': str(Path().absolute()),
                    'train': str(Path(self.config['dataset']['train_path']).absolute() / 'images'),
                    'val': str(Path(self.config['dataset']['val_path']).absolute() / 'images'),
                    'test': str(Path(self.config['dataset']['test_path']).absolute() / 'images'),
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
                    self.logger.info(f"Running validation on {temp_config_path}...")
                    results = eval_model.val(
                        data=temp_config_path,
                        batch=self.config['yolo']['batch_size'],
                        imgsz=self.config['yolo']['imgsz'],
                        device=str(self.device).replace('cuda:', ''),  # Remove 'cuda:' prefix if present
                        workers=0,
                        project=os.path.join('runs', 'eval'),
                        name=f'round_{server_round}',
                        verbose=True
                    )
                    
                    if results is None: 
                        # Try to retrieve metrics from the validator if val() returned None
                        if hasattr(eval_model, 'validator') and eval_model.validator is not None:
                            results = eval_model.validator.metrics
                        else:
                            self.logger.warning("Validation failed to return results. Using dummy metrics to allow training to proceed.")
                            results = {
                                'metrics/precision(B)': 0.0,
                                'metrics/recall(B)': 0.0,
                                'metrics/mAP50(B)': 0.0,
                                'metrics/mAP50-95(B)': 0.0
                            }

                    # Extract metrics safely
                    val_metrics = getattr(results, 'results_dict', results) if results is not None else {}
                    if not isinstance(val_metrics, dict):
                        val_metrics = {}

                    # Log validation results
                    self.logger.info(f"Round {server_round} evaluation results:")
                    self.logger.info(f"mAP50: {val_metrics.get('metrics/mAP50(B)', 0.0)}")
                    self.logger.info(f"mAP50-95: {val_metrics.get('metrics/mAP50-95(B)', 0.0)}")
                    
                    metrics = {
                        "loss": float(1.0 - val_metrics.get('metrics/mAP50(B)', 0.0)),  # Using 1-mAP as loss
                        "accuracy": float(val_metrics.get('metrics/precision(B)', 0.0)),
                        "precision": float(val_metrics.get('metrics/precision(B)', 0.0)),
                        "recall": float(val_metrics.get('metrics/recall(B)', 0.0)),
                        "mAP50": float(val_metrics.get('metrics/mAP50(B)', 0.0)),
                        "mAP50-95": float(val_metrics.get('metrics/mAP50-95(B)', 0.0)),
                    }
                finally:
                    # Clean up the temporary config file
                    try:
                        if os.path.exists(temp_config_path):
                            os.unlink(temp_config_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to clean up temporary config file: {e}")
            
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
            
            # Log and save metrics
            self.metrics[f'round_{server_round}'] = metrics
            log_round_metrics(server_round, metrics, is_global=True)
            save_metrics(self.metrics, os.path.join(self.results_dir, 'training_metrics.json'))
            
            # Generate plots
            plot_metrics(self.metrics, os.path.join(self.results_dir, 'plots'))
            
            # Return loss and metrics
            return metrics["loss"], metrics

        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Return default metrics in case of error
            error_metrics = {
                "loss": 1.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "mAP50": 0.0,
                "mAP50-95": 0.0,
            }
            return 1.0, error_metrics

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
