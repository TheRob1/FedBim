import warnings
import os
import sys

# Suppress annoying deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
import logging
import yaml
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

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
import torch.nn as nn

# Monkey-patch torch.load to handle newer PyTorch versions defaults
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from ultralytics.nn.tasks import DetectionModel
try:
    import ultralytics.nn.modules as modules
    torch.serialization.add_safe_globals([
        DetectionModel, nn.modules.container.Sequential, nn.modules.conv.Conv2d, 
        nn.modules.batchnorm.BatchNorm2d, nn.modules.activation.SiLU
    ])
except Exception:
    torch.serialization.add_safe_globals([DetectionModel, nn.modules.container.Sequential])

# Now import other deep learning related modules
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

# Import YOLO after torch to ensure proper initialization
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Import local modules
sys.path.append(str(Path(__file__).parent))
from utils.visualization import setup_logging, log_round_metrics, save_metrics

# Constants
TRAINING_TIMEOUT = 2 * 60 * 60  # 2 hours
gRPC_OPTIONS = [
    ('grpc.max_send_message_length', 512 * 1024 * 1024),  # 512MB
    ('grpc.max_receive_message_length', 512 * 1024 * 1024),  # 512MB
    ('grpc.keepalive_time_ms', 30000),  # 30 seconds
    ('grpc.keepalive_timeout_ms', 10000),  # 10 seconds
    ('grpc.keepalive_permit_without_calls', True),
    ('grpc.http2.max_pings_without_data', 0),
    ('grpc.http2.min_time_between_pings_ms', 10000),  # 10 seconds
    ('grpc.http2.min_ping_interval_without_data_ms', 5000),  # 5 seconds
]

class TrainingError(Exception):
    """Custom exception for training-related errors."""
    pass

def train_with_timeout(model: YOLO, data_config: Union[str, dict], epochs: int, batch_size: int, 
                      imgsz: int, device: str, workers: int, project: str, name: str, 
                      verbose: bool = True) -> Any:
    """Helper function to run training with proper error handling and logging.
    
    Args:
        model: The YOLO model to train
        data_config: Path to data config file or dict with data config
        epochs: Number of training epochs
        batch_size: Batch size for training
        imgsz: Image size for training
        device: Device to use for training ('cpu', 'cuda:0', etc.)
        workers: Number of worker threads for data loading
        project: Project name for saving results
        name: Name for this training run
        verbose: Whether to print training progress
        
    Returns:
        Training results
        
    Raises:
        TrainingError: If training fails
    """
    try:
        # Ensure data_config is a file path
        if isinstance(data_config, dict):
            # Save config to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(data_config, f)
                data_config_path = f.name
        else:
            data_config_path = data_config
            
        # Run training
        results = model.train(
            data=data_config_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            workers=workers,
            project=project,
            name=name,
            exist_ok=True,
            optimizer='SGD',
            verbose=verbose
        )
        
        # Clean up temporary file if we created one
        if 'data_config_path' in locals() and os.path.exists(data_config_path):
            try:
                os.unlink(data_config_path)
            except Exception as e:
                logging.warning(f"Failed to delete temporary config file {data_config_path}: {e}")
        
        return results
        
    except Exception as e:
        raise TrainingError(f"Training failed: {str(e)}")

class FireSafetyClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, config: dict):
        """Initialize the federated learning client.
        
        Args:
            client_id: Unique identifier for this client
            config: Configuration dictionary containing model and training parameters
            
        Raises:
            RuntimeError: If model initialization fails
        """
        self.start_time = time.time()
        self.client_id = client_id
        self.config = config
        
        # Set up logging
        self.logger = setup_logging()
        self.logger.info(f"Initializing client {client_id}...")
        
        try:
            # Initialize device
            self.device = self._initialize_device()
            
            # Initialize model with safe loading
            self.model = self._initialize_model()
            
            # Set up model overrides
            self._setup_model_overrides()
            
            # Set up data paths for this client
            self._setup_client_data()
            
            # Initialize metrics
            self.metrics = {
                'fit_metrics': {},
                'evaluate_metrics': {}
            }
            
            # Verify model is working
            self._verify_model()
            
            self.logger.info(f"Client {client_id} initialized in {time.time() - self.start_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize client {client_id}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Client initialization failed: {str(e)}")
    
    def _initialize_device(self) -> torch.device:
        """Initialize and verify PyTorch device."""
        device_config = str(self.config['yolo'].get('device', '0'))
        try:
            # If CUDA_VISIBLE_DEVICES is set, 'cuda:0' always refers to the first visible GPU
            # even if that was originally GPU 6.
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
    
    def _initialize_model(self) -> YOLO:
        """Initialize the YOLO model."""
        model_path = Path(self.config['yolo']['model'])
        self.logger.info(f"Loading model from {model_path}")
        
        try:
            # We use the standard YOLO load to match the server.
            model = YOLO(str(model_path))
            # Explicitly move model to device to avoid weight/input mismatch
            model.to(self.device)
            self.logger.info(f"Successfully loaded model and moved to {self.device}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def _setup_model_overrides(self):
        """Configure model training parameters."""
        self.model.overrides = {
            'device': str(self.device),
            'data': self._create_data_yaml(),
            'imgsz': self.config['yolo']['imgsz'],
            'batch': self.config['yolo']['batch_size'],
            'epochs': self.config['yolo']['epochs'],
            'workers': min(4, os.cpu_count() // 2),  # Use at most half the CPU cores
            'optimizer': self.config['yolo'].get('optimizer', 'auto'),
            'project': str(Path('runs').absolute() / 'train'),
            'name': f'client_{self.client_id}',
            'exist_ok': True,
            'verbose': True,
            # Training hyperparameters with fallbacks
            'momentum': self.config['yolo'].get('momentum', 0.937),
            'weight_decay': self.config['yolo'].get('weight_decay', 0.0005),
            'warmup_epochs': self.config['yolo'].get('warmup_epochs', 3.0),
            'warmup_momentum': self.config['yolo'].get('warmup_momentum', 0.8),
            'warmup_bias_lr': self.config['yolo'].get('warmup_bias_lr', 0.1),
            'box': self.config['yolo'].get('box', 7.5),
            'cls': self.config['yolo'].get('cls', 0.5),
            'dfl': self.config['yolo'].get('dfl', 1.5),
            'fl_gamma': self.config['yolo'].get('fl_gamma', 0.0),
            # Data augmentation
            'hsv_h': self.config['yolo'].get('hsv_h', 0.015),
            'hsv_s': self.config['yolo'].get('hsv_s', 0.7),
            'hsv_v': self.config['yolo'].get('hsv_v', 0.4),
            'degrees': self.config['yolo'].get('degrees', 0.0),
            'translate': self.config['yolo'].get('translate', 0.1),
            'scale': self.config['yolo'].get('scale', 0.5),
            'shear': self.config['yolo'].get('shear', 0.0),
            'perspective': self.config['yolo'].get('perspective', 0.0),
            'flipud': self.config['yolo'].get('flipud', 0.0),
            'fliplr': self.config['yolo'].get('fliplr', 0.5),
            'mosaic': self.config['yolo'].get('mosaic', 1.0),
            'mixup': self.config['yolo'].get('mixup', 0.0),
            'copy_paste': self.config['yolo'].get('copy_paste', 0.0),
        }
    
    def _verify_model(self):
        """Verify that the model is properly initialized and can perform a forward pass."""
        try:
            # Create a dummy input tensor
            imgsz = self.config['yolo']['imgsz']
            dummy_input = torch.randn(1, 3, imgsz, imgsz).to(self.device)
            
            # Test forward pass using the underlying model
            # We need to access .model to get the PyTorch module
            model = self.model.model
            model.eval()
            
            with torch.no_grad():
                _ = model(dummy_input)
                
            self.logger.info("Model verification successful")
            
        except Exception as e:
            self.logger.error(f"Model verification failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model verification failed: {str(e)}")
    
    def _create_data_yaml(self) -> str:
        """Create a temporary data.yaml file for YOLO training."""
        data = {
            'train': str(Path(f'data/client_{self.client_id}/train/images').absolute()),
            'val': str(Path(f'data/client_{self.client_id}/valid/images').absolute()),
            'test': str(Path(self.config['dataset']['test_path']).absolute() / 'images'),
            'nc': self.config['dataset']['nc'],
            'names': self.config['dataset']['names']
        }
        
        os.makedirs('temp', exist_ok=True)
        yaml_path = os.path.join('temp', f'client_{self.client_id}_data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)
            
        return yaml_path
    
    def _setup_client_data(self):
        """Set up client-specific data directories."""
        # In a real scenario, you would split your data here
        # For this example, we'll just use the same data for all clients
        # but in practice, you'd want to split the data differently for each client
        client_data_dir = os.path.join('data', f'client_{self.client_id}')
        os.makedirs(os.path.join(client_data_dir, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(client_data_dir, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(client_data_dir, 'valid', 'images'), exist_ok=True)
        os.makedirs(os.path.join(client_data_dir, 'valid', 'labels'), exist_ok=True)
        
        # In a real scenario, you would copy/symlink the appropriate data here
        # For now, we'll just use the full dataset for all clients
        # This is just a placeholder - in practice, you'd want to split the data
        
    def get_parameters(self, config) -> List[np.ndarray]:
        """Return the current model parameters as a list of NumPy arrays."""
        self.logger.debug(f"Client {self.client_id}: Getting parameters")
        state_dict = self.model.model.state_dict()
        keys = sorted(state_dict.keys())
        return [state_dict[k].cpu().numpy() for k in keys]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set the model parameters from a list of NumPy arrays."""
        self.logger.debug(f"Client {self.client_id}: Setting parameters")
        state_dict = self.model.model.state_dict()
        keys = sorted(state_dict.keys())
        
        if len(keys) != len(parameters):
            raise RuntimeError(f"Parameter mismatch: expected {len(keys)} BUT got {len(parameters)}")
            
        new_state_dict = {}
        for k, param in zip(keys, parameters):
             new_state_dict[k] = torch.tensor(param)
             
        self.model.model.load_state_dict(new_state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model on the local dataset with timeout handling."""
        start_time = time.time()
        round_num = config.get('server_round', 0)
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Client {self.client_id}: Starting training round {round_num}")
        
        # Set a timeout for the entire training process (2 hours)
        training_timeout = 2 * 60 * 60
        
        try:
            # Set the model parameters
            self.set_parameters(parameters)
            
            # Get training parameters
            epochs = config.get("epochs", self.config['yolo']['epochs'])
            batch_size = config.get("batch_size", self.config['yolo']['batch_size'])
            imgsz = self.config['yolo']['imgsz']
            workers = self.config['yolo']['workers']
            project = os.path.join('runs', 'train')
            name = f'client_{self.client_id}_round_{round_num}'
            data_config = self._create_data_yaml()
            
            # Run training in a separate thread with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    train_with_timeout,
                    model=self.model,
                    data_config=data_config,
                    epochs=epochs,
                    batch_size=batch_size,
                    imgsz=imgsz,
                    device=str(self.device).replace('cuda:', ''),
                    workers=0, # Set to 0 for increased stability in multi-tenant environments
                    project=project,
                    name=name,
                    verbose=True
                )
                
                try:
                    train_results = future.result(timeout=training_timeout)
                    self.logger.info(f"Training completed successfully in {time.time() - start_time:.2f} seconds")
                except FutureTimeout:
                    self.logger.error(f"Training timed out after {training_timeout} seconds")
                    raise TimeoutError(f"Training exceeded {training_timeout} seconds")
            
            # Get the updated parameters
            parameters_prime = self.get_parameters(config={})
            
            # Count local training examples
            local_train_dir = os.path.join('data', f'client_{self.client_id}', 'train', 'images')
            num_examples_train = len(os.listdir(local_train_dir)) if os.path.exists(local_train_dir) else 0
            
            # Calculate metrics safely (YOLOv8.0.0 might return None)
            train_metrics = getattr(train_results, 'results_dict', {}) if train_results else {}
            
            # Try to recover metrics from the model object if results_dict is missing
            if not train_metrics and hasattr(self.model, 'trainer') and self.model.trainer is not None:
                trainer = self.model.trainer
                loss_items = None
                
                # Check for label_loss_items (can be attribute or method)
                labels = None
                if hasattr(trainer, 'label_loss_items'):
                    labels = trainer.label_loss_items
                    if callable(labels):
                        labels = labels()
                
                # Check for loss_items (values)
                values = None
                if hasattr(trainer, 'loss_items'):
                    values = trainer.loss_items
                    if callable(values):
                        values = values()
                
                # If we have both labels and values, and they match in length, create a dict
                if labels is not None and values is not None and len(labels) == len(values):
                    train_metrics = {f"train/{l}" if not isinstance(l, str) or not l.startswith("train/") else l: float(v) 
                                   for l, v in zip(labels, values)}
                
                # Fallback 1: If labels actually contains numeric values (old behavior)
                elif labels is not None and len(labels) > 0 and not isinstance(labels[0], str):
                    train_metrics = {
                        'train/box_loss': float(labels[0]) if len(labels) > 0 else 0.0,
                        'train/cls_loss': float(labels[1]) if len(labels) > 1 else 0.0,
                        'train/dfl_loss': float(labels[2]) if len(labels) > 2 else 0.0,
                    }
                
                # Fallback 2: Check for trainer.metrics
                elif hasattr(trainer, 'metrics') and isinstance(trainer.metrics, dict):
                    train_metrics = trainer.metrics
                
                # Fallback 3: If we have values but no labels, assume standard YOLO order
                elif values and len(values) >= 3:
                    train_metrics = {
                        'train/box_loss': float(values[0]),
                        'train/cls_loss': float(values[1]),
                        'train/dfl_loss': float(values[2]),
                    }

            metrics = {
                'loss': float(train_metrics.get('train/box_loss', 0.0) + 
                             train_metrics.get('train/cls_loss', 0.0) + 
                             train_metrics.get('train/dfl_loss', 0.0)),
                'box_loss': float(train_metrics.get('train/box_loss', 0.0)),
                'cls_loss': float(train_metrics.get('train/cls_loss', 0.0)),
                'dfl_loss': float(train_metrics.get('train/dfl_loss', 0.0)),
                'lr': float(train_metrics.get('lr/pg0', 0.0)),
                'training_time': time.time() - start_time
            }
            
            # Log metrics
            self.metrics['fit_metrics'][f'round_{round_num}'] = metrics
            log_round_metrics(round_num, metrics, is_global=False, client_id=self.client_id)
            
            # Save metrics to file
            client_metrics_dir = os.path.join('results', 'client_metrics')
            os.makedirs(client_metrics_dir, exist_ok=True)
            save_metrics(
                self.metrics, 
                os.path.join(client_metrics_dir, f'client_{self.client_id}_metrics.json')
            )
            
            self.logger.info(f"Client {self.client_id}: Completed training round {round_num} in {metrics['training_time']:.2f} seconds")
            
            return parameters_prime, num_examples_train, metrics
            
        except Exception as e:
            self.logger.error(f"Error during fit on client {self.client_id}: {str(e)}", exc_info=True)
            raise
    
    def evaluate(self, parameters, config):
        """Evaluate the model on the local test set."""
        start_time = time.time()
        round_num = config.get('server_round', 0)
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Client {self.client_id}: Starting evaluation round {round_num}")
        
        try:
            # We use a TEMPORARY model for evaluation to prevent layer fusion
            # which happens during val() and modifies the model structure permanently
            
            # Create a new YOLO instance with the same model to ensure architecture matches
            temp_model = YOLO(str(self.config['yolo']['model']))
            
            # Load parameters into the temporary model
            state_dict = temp_model.model.state_dict()
            keys = sorted(state_dict.keys())
            
            if len(keys) != len(parameters):
                 raise RuntimeError(f"Parameter mismatch during eval: expected {len(keys)} BUT got {len(parameters)}")
                 
            new_state_dict = {}
            for k, param in zip(keys, parameters):
                 new_state_dict[k] = torch.tensor(param)
            
            temp_model.model.load_state_dict(new_state_dict, strict=True)
            
            # Evaluate the temp model
            results = temp_model.val(
                data=self._create_data_yaml(),
                batch=config.get("batch_size", self.config['yolo']['batch_size']),
                imgsz=self.config['yolo']['imgsz'],
                device=str(self.device).replace('cuda:', ''),
                workers=0,
                project=os.path.join('runs', 'val'),
                name=f'client_{self.client_id}_round_{round_num}',
                exist_ok=True
            )
            
            # Calculate metrics safely (YOLOv8.0.0 might return None)
            val_metrics = getattr(results, 'results_dict', {}) if results else {}
            
            # Try to recover metrics from the model object if results_dict is missing
            if not val_metrics and hasattr(temp_model, 'validator') and temp_model.validator is not None:
                val_metrics = temp_model.validator.metrics

            metrics = {
                'loss': float(1.0 - val_metrics.get('metrics/mAP50(B)', 0.0)),  # Using 1-mAP as loss
                'accuracy': float(val_metrics.get('metrics/precision(B)', 0.0)),
                'precision': float(val_metrics.get('metrics/precision(B)', 0.0)),
                'recall': float(val_metrics.get('metrics/recall(B)', 0.0)),
                'mAP50': float(val_metrics.get('metrics/mAP50(B)', 0.0)),
                'mAP50-95': float(val_metrics.get('metrics/mAP50-95(B)', 0.0)),
                'evaluation_time': time.time() - start_time
            }
            
            # Use local validation images for counting
            local_val_dir = os.path.join('data', f'client_{self.client_id}', 'valid', 'images')
            num_examples_test = len(os.listdir(local_val_dir)) if os.path.exists(local_val_dir) else 0
            self.metrics['evaluate_metrics'][f'round_{round_num}'] = metrics
            
            # Log metrics
            log_round_metrics(round_num, metrics, is_global=False, client_id=self.client_id)
            
            # Save metrics to file
            client_metrics_dir = os.path.join('results', 'client_metrics')
            os.makedirs(client_metrics_dir, exist_ok=True)
            save_metrics(
                self.metrics, 
                os.path.join(client_metrics_dir, f'client_{self.client_id}_metrics.json')
            )
            
            self.logger.info(f"Client {self.client_id}: Completed evaluation round {round_num} in {metrics['evaluation_time']:.2f} seconds")
            
            return float(metrics['loss']), num_examples_test, metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation on client {self.client_id}: {str(e)}", exc_info=True)
            raise

def client_fn(cid: str) -> FireSafetyClient:
    """Create and return a client instance with gRPC options.
    
    Args:
        cid: Client ID as a string
        
    Returns:
        Initialized FireSafetyClient instance
        
    Raises:
        RuntimeError: If client initialization fails
    """
    logger = setup_logging()
    
    try:
        # Load configuration
        config_path = Path(__file__).parent / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Initializing client {cid}")
        
        # Create and start client
        client = FireSafetyClient(int(cid), config)
        
        # Get server address from config
        server_address = config['federated_learning']['server']['address']
        
        # Start Flower client with gRPC options
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client,
            grpc_max_message_length=512 * 1024 * 1024,  # 512MB
        )
        
        return client
        
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize client {cid}: {str(e)}", exc_info=True)
        raise RuntimeError(f"Client {cid} initialization failed: {str(e)}")

if __name__ == "__main__":
    import argparse
    import signal
    
    def signal_handler(sig, frame):
        logger = logging.getLogger(__name__)
        logger.info("Received shutdown signal. Exiting gracefully...")
        sys.exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description="Start a Flower client.")
        parser.add_argument("--cid", type=int, required=True, help="Client ID")
        parser.add_argument("--server-address", type=str, default=None, 
                          help="Server address (default: read from config)")
        parser.add_argument("--log-level", type=str, default="INFO",
                          choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                          help="Logging level (default: INFO)")
        # In client.py, update the argument parser to include config:
        parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
        
        args = parser.parse_args()
        
        # Configure logging
        logger = setup_logging(level=args.log_level)
        logger.info(f"Starting client {args.cid}")
        
        # Load configuration
        config_path = Path(__file__).parent / 'config.yaml'
        if args.config:
             config_path = Path(args.config)
             
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Determine server address
        if args.server_address:
            config['federated_learning']['server']['address'] = args.server_address
        else:
            # Construct address from config
            server_config = config['federated_learning']['server']
            address = server_config.get('address', '127.0.0.1')
            port = server_config.get('port', 8080)
            args.server_address = f"{address}:{port}"
        
        # Create and start client
        client = FireSafetyClient(args.cid, config)
        
        # Start Flower client with gRPC options
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client,
            grpc_max_message_length=512 * 1024 * 1024,  # 512MB
        )
        
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Client error: {str(e)}", exc_info=True)
        sys.exit(1)

