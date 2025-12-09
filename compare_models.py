import os
import torch
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

class ModelComparator:
    def __init__(self, config_path='config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results_dir = 'comparison_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize models to compare
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all models for comparison."""
        # Load the federated learning model (most recent one)
        results_dir = 'results'
        if os.path.exists(results_dir) and os.listdir(results_dir):
            model_files = [f for f in os.listdir(results_dir) if f.endswith('.pt')]
            if model_files:
                # Sort by round number and get the latest
                model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                latest_model = os.path.join(results_dir, model_files[-1])
                self.models['federated'] = {
                    'path': latest_model,
                    'model': None,  # Will be loaded on demand
                    'metrics': {}
                }
        
        # Load models from Visual_Annotation_Tool
        model_dirs = [
            'Detection_Condition_amodal_Yolov8',
            'Detection_Condition_modal_Yolov8',
            'Detection_FSE_Yolov8',
            'Detection_fire_class_symbols_Yolov8',
            'Detection_inspection_tags_Yolov8',
            'Detection_marking_Yolov8'
        ]
        
        for model_dir in model_dirs:
            model_path = os.path.join('Visual_Annotation_Tool', model_dir, 'best.pt')
            if os.path.exists(model_path):
                model_name = model_dir.split('_', 1)[1]  # Remove 'Detection_' prefix
                self.models[model_name] = {
                    'path': model_path,
                    'model': None,  # Will be loaded on demand
                    'metrics': {}
                }
    
    def load_model(self, model_info):
        """Load a model if not already loaded."""
        if model_info['model'] is None:
            try:
                model = YOLO(model_info['path'])
                model.overrides = {
                    'device': str(self.device),
                    'imgsz': self.config['yolo']['imgsz'],
                    'batch': self.config['yolo']['batch_size'],
                    'workers': self.config['yolo']['workers']
                }
                model_info['model'] = model
            except Exception as e:
                print(f"Error loading model {model_info['path']}: {e}")
                return False
        return True
    
    def evaluate_models(self):
        """Evaluate all models on the test set."""
        test_data = {
            'val': self.config['dataset']['test_path'],  # Using test set for evaluation
            'nc': self.config['dataset']['nc'],
            'names': self.config['dataset']['names']
        }
        
        for model_name, model_info in tqdm(self.models.items(), desc="Evaluating models"):
            if not self.load_model(model_info):
                continue
                
            try:
                # Evaluate the model
                results = model_info['model'].val(
                    data=test_data,
                    batch_size=self.config['yolo']['batch_size'],
                    imgsz=self.config['yolo']['imgsz'],
                    device=self.device,
                    workers=self.config['yolo']['workers'],
                    project=os.path.join('runs', 'eval'),
                    name=f'compare_{model_name}',
                    exist_ok=True
                )
                
                # Store metrics
                model_info['metrics'] = {
                    'precision': float(results.results_dict.get('metrics/precision(B)', 0.0)),
                    'recall': float(results.results_dict.get('metrics/recall(B)', 0.0)),
                    'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0.0)),
                    'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0.0)),
                    'speed': float(results.speed['inference']),
                    'params': sum(p.numel() for p in model_info['model'].model.parameters())
                }
                
                print(f"\nModel: {model_name}")
                for k, v in model_info['metrics'].items():
                    print(f"  {k}: {v:.4f}")
                
            except Exception as e:
                print(f"Error evaluating model {model_name}: {e}")
    
    def plot_comparison(self):
        """Create comparison plots of model metrics."""
        if not any('metrics' in model_info for model_info in self.models.values()):
            print("No evaluation results found. Run evaluate_models() first.")
            return
        
        # Prepare data for plotting
        metrics = ['precision', 'recall', 'mAP50', 'mAP50-95', 'speed', 'params']
        data = {m: [] for m in metrics}
        data['model'] = []
        
        for model_name, model_info in self.models.items():
            if not model_info.get('metrics'):
                continue
                
            data['model'].append(model_name)
            for m in metrics:
                data[m].append(model_info['metrics'].get(m, 0))
        
        # Create bar plots for each metric
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='model', y=metric, data=data)
            plt.title(f'Model Comparison - {metric.upper()}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'{metric}_comparison.png'))
            plt.close()
        
        # Save metrics to JSON
        metrics_dict = {
            model_name: model_info.get('metrics', {}) 
            for model_name, model_info in self.models.items()
        }
        
        with open(os.path.join(self.results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"\nComparison results saved to {self.results_dir}")

def main():
    # Initialize the comparator
    comparator = ModelComparator()
    
    # Evaluate all models
    print("Evaluating models...")
    comparator.evaluate_models()
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    comparator.plot_comparison()
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main()
