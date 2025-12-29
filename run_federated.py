#!/usr/bin/env python3
"""
Run script for federated learning with YOLOv8 and Flower.
Handles data distribution, server, and client processes.
"""
import warnings
import os
import sys

# Suppress annoying deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
import yaml
import random
import torch
import numpy as np
import argparse
import subprocess
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import shutil
import signal
import atexit
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global variables to store process references
processes = []
logger = None

def setup_logging():
    """Set up logging configuration."""
    import logging
    from datetime import datetime
    
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / f'federated_learning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def cleanup():
    """Clean up function to terminate all subprocesses."""
    logger.info("Cleaning up processes...")
    for p in processes:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception as e:
            logger.warning(f"Error terminating process: {e}")
    logger.info("All processes terminated.")

def run_command(
    command: str, 
    cwd: str = None, 
    shell: bool = True, 
    wait: bool = True,
    log_prefix: str = ""
):
    """Run a shell command with improved output handling and logging."""
    logger.info(f"{log_prefix}Executing: {command}")
    
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        if wait:
            # Stream output in real-time
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    output_lines.append(output)
                    logger.info(f"{log_prefix}{output}")
            
            # Check for any errors
            _, stderr = process.communicate()
            if process.returncode != 0:
                error_msg = f"Command failed with code {process.returncode}: {command}"
                if stderr:
                    error_msg += f"\nError: {stderr.strip()}"
                logger.error(error_msg)
                raise subprocess.CalledProcessError(process.returncode, command, stderr)
            
            return process, "\n".join(output_lines)
            
        else:
            # For non-waiting processes, start threads to handle output
            def stream_output(pipe, log_func):
                for line in iter(pipe.readline, ''):
                    log_func(f"{log_prefix}{line.strip()}")
                pipe.close()
            
            # Start output and error threads
            threading.Thread(
                target=stream_output, 
                args=(process.stdout, logger.info),
                daemon=True
            ).start()
            
            threading.Thread(
                target=stream_output, 
                args=(process.stderr, logger.info),
                daemon=True
            ).start()
            
            processes.append(process)
            return process, None
            
    except Exception as e:
        logger.error(f"Failed to run command: {e}", exc_info=True)
        raise

def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def distribute_data_dirichlet(
    base_dir: str, 
    output_dir: str,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    val_ratio: float = 0.2
) -> Dict[str, Dict[str, List[str]]]:
    """
    Distribute data using Dirichlet distribution for non-IID scenarios.
    
    Args:
        base_dir: Base directory containing 'train', 'valid', 'test' folders
        output_dir: Directory to save client data
        num_clients: Number of clients
        alpha: Concentration parameter for Dirichlet distribution (smaller = more non-IID)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Get images and their classes by reading label files
    train_dir = os.path.join(base_dir, 'train')
    img_dir = os.path.join(train_dir, 'images')
    lbl_dir = os.path.join(train_dir, 'labels')
    
    all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    class_images = {}
    
    logger.info("Scanning labels to determine class distribution...")
    for img_name in all_images:
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(lbl_dir, label_name)
        
        assigned_class = 'unknown'
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    assigned_class = first_line.split()[0]
        
        if assigned_class not in class_images:
            class_images[assigned_class] = []
        class_images[assigned_class].append(img_name)
    
    class_names = sorted(class_images.keys())
    logger.info(f"Found {len(class_names)} classes: {class_names}")
    
    # Generate Dirichlet distribution for each class
    client_distributions = {}
    for class_name, images in class_images.items():
        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        client_distributions[class_name] = (proportions * len(images)).astype(int)
        
        # Ensure we distribute all images
        total = sum(client_distributions[class_name])
        if total < len(images):
            client_distributions[class_name][0] += len(images) - total
    
    # Distribute images to clients
    for client_id in range(num_clients):
        print(f"\nPreparing data for client {client_id + 1}...")
        
        # Create client directories
        client_dir = os.path.join(output_dir, f'client_{client_id + 1}')
        os.makedirs(os.path.join(client_dir, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(client_dir, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(client_dir, 'valid', 'images'), exist_ok=True)
        os.makedirs(os.path.join(client_dir, 'valid', 'labels'), exist_ok=True)
        
        # For each class, copy the assigned number of images to this client
        for class_name, images in class_images.items():
            # Get images for this client and class
            num_images = client_distributions[class_name][client_id]
            if num_images == 0:
                continue
                
            # Sample without replacement
            selected_images = random.sample(images, num_images)
            
            # Copy images and labels
            for img_name in selected_images:
                # Source paths
                src_img = os.path.join(train_dir, 'images', img_name)
                label_name = os.path.splitext(img_name)[0] + '.txt'
                src_label = os.path.join(train_dir, 'labels', label_name)
                
                # Destination paths
                dst_img = os.path.join(client_dir, 'train', 'images', img_name)
                dst_label = os.path.join(client_dir, 'train', 'labels', label_name)
                
                # Copy files
                if os.path.exists(src_img):
                    shutil.copy2(src_img, dst_img)
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
        
        # For validation, we'll use a fixed split for all clients
        # In a real scenario, you might want to customize this per client
        val_ratio = 0.2  # 20% of training data for validation
        val_dir = os.path.join(base_dir, 'valid')
        if os.path.exists(val_dir):
            val_images = [f for f in os.listdir(os.path.join(val_dir, 'images')) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in val_images:
                # Source paths
                src_img = os.path.join(val_dir, 'images', img_name)
                label_name = os.path.splitext(img_name)[0] + '.txt'
                src_label = os.path.join(val_dir, 'labels', label_name)
                
                # Destination paths
                dst_img = os.path.join(client_dir, 'valid', 'images', img_name)
                dst_label = os.path.join(client_dir, 'valid', 'labels', label_name)
                
                # Copy files
                if os.path.exists(src_img):
                    shutil.copy2(src_img, dst_img)
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
    
    print("\nData distribution complete!")
    print(f"Distributed data to {num_clients} clients in {output_dir}")

def start_server():
    """Start the federated learning server."""
    try:
        config = load_config()
        device = str(config['yolo'].get('device', '0'))
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if device != 'cpu':
            env["CUDA_VISIBLE_DEVICES"] = device
        
        server_cmd = f"{sys.executable} server.py --config config.yaml"
        logger.info(f"Starting federated learning server on GPU {device}...")
        
        process = subprocess.Popen(
            server_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )
        
        # Monitor output in a thread
        def stream_output(pipe, log_prefix):
            for line in iter(pipe.readline, ''):
                logger.info(f"{log_prefix}{line.strip()}")
            pipe.close()
            
        threading.Thread(target=stream_output, args=(process.stdout, "[SERVER] "), daemon=True).start()
        threading.Thread(target=stream_output, args=(process.stderr, "[SERVER] "), daemon=True).start()
        
        processes.append(process)
        return process, None
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise

def start_client(client_id: int, config_path: str = 'config.yaml'):
    """Start a federated learning client."""
    try:
        config = load_config(config_path)
        device = str(config['yolo'].get('device', '0'))
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if device != 'cpu':
            env["CUDA_VISIBLE_DEVICES"] = device
            
        client_cmd = f"{sys.executable} client.py --cid {client_id} --config {config_path}"
        logger.info(f"Starting client {client_id} on GPU {device}...")
        
        process = subprocess.Popen(
            client_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )
        
        # Monitor output in a thread
        def stream_output(pipe, log_prefix):
            for line in iter(pipe.readline, ''):
                logger.info(f"{log_prefix}{line.strip()}")
            pipe.close()
            
        threading.Thread(target=stream_output, args=(process.stdout, f"[CLIENT-{client_id}] "), daemon=True).start()
        threading.Thread(target=stream_output, args=(process.stderr, f"[CLIENT-{client_id}] "), daemon=True).start()
        
        processes.append(process)
        return process, None
    except Exception as e:
        logger.error(f"Failed to start client {client_id}: {e}", exc_info=True)
        raise

def monitor_processes(processes: List[subprocess.Popen], timeout: int = 5) -> bool:
    """Monitor processes and return True if all completed successfully."""
    try:
        for process in processes:
            try:
                process.wait(timeout=timeout)
                if process.returncode != 0:
                    logger.error(f"Process failed with return code {process.returncode}")
                    return False
            except subprocess.TimeoutExpired:
                pass # Normal behavior for long-running processes
        return True
    except Exception as e:
        logger.error(f"Error monitoring processes: {e}")
        return False

def save_training_metrics(metrics: Dict[str, Any], output_dir: str = 'results'):
    """Save training metrics to a JSON file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        metrics_file = os.path.join(output_dir, 'training_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved training metrics to {metrics_file}")
    except Exception as e:
        logger.error(f"Failed to save training metrics: {e}")

def main():
    global logger
    logger = setup_logging()
    
    # Register cleanup handler
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda s, f: (cleanup(), sys.exit(0)))
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Run federated learning with YOLOv8 and Flower')
        parser.add_argument('--num-clients', type=int, default=2, help='Number of clients to start')
        parser.add_argument('--distribute-data', action='store_true', help='Distribute data among clients')
        parser.add_argument('--data-dir', type=str, default='606_train_152_val_12_test', help='Base directory for data')
        parser.add_argument('--output-dir', type=str, default='data', help='Output directory for client data')
        parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet distribution parameter')
        parser.add_argument('--seed', type=int, default=42, help='Random seed')
        parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
        
        args = parser.parse_args()
        
        # Load configuration
        config = load_config(args.config)
        
        # Set random seeds for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # Distribute data if requested
        if args.distribute_data:
            logger.info(f"Distributing data among {args.num_clients} clients...")
            distribute_data_dirichlet(
                base_dir=args.data_dir,
                output_dir=args.output_dir,
                num_clients=args.num_clients,
                alpha=args.alpha,
                seed=args.seed
            )
        
        # Start server
        logger.info("Starting federated learning server...")
        server_process, _ = start_server()
        
        # Give server time to start
        time.sleep(5)
        
        # Start clients with a stagger
        logger.info(f"Starting {args.num_clients} clients...")
        client_processes = []
        for i in range(args.num_clients):
            process, _ = start_client(i + 1, args.config)
            client_processes.append(process)
            if i < args.num_clients - 1:
                logger.info("Waiting 5 seconds before starting next client...")
                time.sleep(5)
        
        # Monitor processes
        logger.info("All processes started. Monitoring...")
        while True:
            if not monitor_processes(client_processes + [server_process]):
                logger.error("One or more processes failed. Shutting down...")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\nReceived keyboard interrupt. Shutting down...")
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
