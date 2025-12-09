import os
import sys
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
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
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils.visualization import setup_logging, plot_metrics, log_round_metrics, save_metrics
from client import FireSafetyClient, client_fn
from server import FireSafetyServer

class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_history = {
            'train': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'mAP50': [], 'mAP50_95': []},
            'val': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'mAP50': [], 'mAP50_95': []}
        }
        self.best_mAP = 0.0
        self.best_round = 0

    def aggregate_fit(self, server_round, results, failures):
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        
        # Log training metrics
        for client_result in results:
            metrics = client_result[1].metrics
            for metric_name in self.metrics_history['train']:
                if metric_name in metrics:
                    self.metrics_history['train'][metric_name].append(metrics[metric_name])
        
        return aggregated_weights

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Log validation metrics
        for client_result in results:
            metrics = client_result[1].metrics
            for metric_name in self.metrics_history['val']:
                if metric_name in metrics:
                    self.metrics_history['val'][metric_name].append(metrics[metric_name])
        
        # Update best model
        current_mAP = np.mean([r[1].metrics.get('mAP50', 0) for r in results])
        if current_mAP > self.best_mAP:
            self.best_mAP = current_mAP
            self.best_round = server_round
            
        return aggregated_metrics

def plot_training_history(history: Dict[str, Dict[str, List[float]]], output_dir: str):
    """Plot training and validation metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    for metric in history['train'].keys():
        plt.figure(figsize=(10, 6))
        
        if metric in history['train'] and history['train'][metric]:
            plt.plot(history['train'][metric], label=f'Train {metric}')
        if metric in history['val'] and history['val'][metric]:
            plt.plot(history['val'][metric], label=f'Validation {metric}')
            
        plt.title(f'{metric.capitalize()} over Rounds')
        plt.xlabel('Rounds')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'{metric}.png'))
        plt.close()

def compare_with_pretrained(fl_history: dict, pretrained_metrics: dict, output_dir: str):
    """Compare federated learning results with pretrained models."""
    os.makedirs(output_dir, exist_ok=True)
    
    for metric in ['mAP50', 'precision', 'recall']:
        if metric in fl_history['val'] and fl_history['val'][metric]:
            plt.figure(figsize=(12, 6))
            
            # Plot federated learning results
            rounds = range(1, len(fl_history['val'][metric]) + 1)
            plt.plot(rounds, fl_history['val'][metric], 'b-', label='Federated Learning')
            
            # Plot pretrained models
            for model_name, metrics in pretrained_metrics.items():
                if metric in metrics:
                    plt.axhline(y=metrics[metric], color='r', linestyle='--', 
                              label=f'{model_name} ({metrics[metric]:.3f})')
            
            plt.title(f'Comparison of {metric.upper()} with Pretrained Models')
            plt.xlabel('Rounds')
            plt.ylabel(metric.upper())
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'))
            plt.close()

def load_pretrained_models(models_dir: str) -> Dict[str, dict]:
    """Load metrics from pretrained models."""
    pretrained_metrics = {}
    
    # List all model directories
    model_dirs = [d for d in os.listdir(models_dir) 
                 if os.path.isdir(os.path.join(models_dir, d))]
    
    for model_dir in model_dirs:
        metrics_file = os.path.join(models_dir, model_dir, 'metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                try:
                    metrics = json.load(f)
                    pretrained_metrics[model_dir] = metrics
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse metrics for {model_dir}")
    
    return pretrained_metrics

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    logger = setup_logging()
    
    # Initialize strategy
    strategy = CustomFedAvg(
        min_fit_clients=config['federated_learning']['min_fit_clients'],
        min_evaluate_clients=config['federated_learning']['min_evaluate_clients'],
        min_available_clients=config['federated_learning']['min_available_clients'],
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Start Flower server
    server = FireSafetyServer(config=config, strategy=strategy)
    
    # Start server
    fl.server.start_server(
        server_address=f"{config['federated_learning']['server']['address']}:"
                     f"{config['federated_learning']['server']['port']}",
        config=fl.server.ServerConfig(
            num_rounds=config['federated_learning']['server']['num_rounds']
        ),
        strategy=strategy
    )
    
    # After training completes, plot metrics
    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training history
    plot_training_history(strategy.metrics_history, output_dir)
    
    # Compare with pretrained models if available
    pretrained_models_dir = 'Visual_Annotation_Tool'
    if os.path.exists(pretrained_models_dir):
        pretrained_metrics = load_pretrained_models(pretrained_models_dir)
        compare_with_pretrained(strategy.metrics_history, pretrained_metrics, output_dir)
    
    logger.info(f"Training completed. Results saved to {output_dir}")

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, float]:
    """Compute weighted average of metrics."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == "__main__":
    main()
