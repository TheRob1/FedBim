import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from datetime import datetime
import logging

def setup_logging(log_dir: str = "logs", level: str = "INFO"):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"federated_learning_{timestamp}.log")
    
    # Get numeric level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
        
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("federated_learning")

def plot_metrics(metrics: Dict[str, Dict], output_dir: str = "results/plots"):
    """Plot and save training metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    rounds = sorted(int(k.split('_')[-1]) for k in metrics.keys() if k.startswith('round_'))
    if not rounds:
        logging.warning("No round metrics found to plot")
        return
    
    # Prepare data
    data = {
        'Round': rounds,
        'Loss': [metrics[f'round_{r}'].get('loss', 0) for r in rounds],
        'Accuracy': [metrics[f'round_{r}'].get('accuracy', 0) for r in rounds],
        'Precision': [metrics[f'round_{r}'].get('precision', 0) for r in rounds],
        'Recall': [metrics[f'round_{r}'].get('recall', 0) for r in rounds],
    }
    
    # Create plots
    for metric in ['Loss', 'Accuracy', 'Precision', 'Recall']:
        plt.figure(figsize=(10, 6))
        plt.plot(data['Round'], data[metric], 'b-', marker='o')
        plt.title(f'Federated Learning - {metric} over Rounds')
        plt.xlabel('Round')
        plt.ylabel(metric)
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{metric.lower()}_over_rounds.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved {metric} plot to {plot_path}")

def compare_models(baseline_metrics: Dict, fed_metrics: Dict, output_dir: str = "results/comparison"):
    """Compare baseline model with federated model."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    baseline_scores = [baseline_metrics.get(m, 0) for m in metrics]
    fed_scores = [fed_metrics.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline')
    rects2 = ax.bar(x + width/2, fed_scores, width, label='Federated')
    
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison: Baseline vs Federated')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    
    fig.tight_layout()
    comparison_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comparison_path)
    plt.close()
    logging.info(f"Saved model comparison plot to {comparison_path}")

def log_round_metrics(round_num: int, metrics: Dict, is_global: bool = False, client_id: Optional[int] = None):
    """Log metrics for a training round."""
    prefix = "Global" if is_global else f"Local (Client {client_id})" if client_id is not None else "Local"
    logging.info(f"\n{'='*50}")
    logging.info(f"{prefix} Round {round_num} Metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            logging.info(f"  {k}: {v:.4f}")
        else:
            logging.info(f"  {k}: {v}")
    logging.info("="*50 + "\n")

def save_metrics(metrics: Dict, filename: str = "training_metrics.json"):
    """Save metrics to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved metrics to {filename}")

def load_metrics(filename: str) -> Dict:
    """Load metrics from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}
