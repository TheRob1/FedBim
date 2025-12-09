import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_training_progress(log_dir="logs", results_dir="results"):
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize metrics dictionary
    metrics = {
        'training_start_time': None,
        'training_duration': None,
        'rounds_completed': 0,
        'clients_connected': 2,  # From the run command
        'model_metrics': {
            'best_accuracy': 0,
            'best_mAP50': 0,
            'best_round': 0
        },
        'system_metrics': {
            'cuda_available': False,
            'device_used': 'CPU'  # From the logs
        }
    }
    
    # Get current timestamp
    current_time = datetime.now()
    
    # Try to find the latest log file
    try:
        log_files = [f for f in os.listdir(log_dir) if f.startswith('federated_learning_') and f.endswith('.log')]
        if log_files:
            latest_log = max(log_files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
            log_path = os.path.join(log_dir, latest_log)
            
            # Parse log file for metrics
            with open(log_path, 'r') as f:
                lines = f.readlines()
                
                # Find training start time
                for line in lines:
                    if "Starting FireSafety Federated Learning Server" in line:
                        metrics['training_start_time'] = line.split(' - ')[0]
                        break
                
                # Count completed rounds
                metrics['rounds_completed'] = sum("Starting evaluation for round" in line for line in lines)
                
                # Find best model metrics
                for line in lines:
                    if "mAP50" in line and "metrics" in line:
                        try:
                            map50 = float(line.split("mAP50': ")[1].split(",")[0])
                            if map50 > metrics['model_metrics']['best_mAP50']:
                                metrics['model_metrics']['best_mAP50'] = round(map50, 4)
                                metrics['model_metrics']['best_round'] = int(line.split("round_")[1].split("'")[0])
                        except:
                            continue
    except Exception as e:
        print(f"Error parsing log files: {e}")
    
    # Calculate training duration if we have a start time
    if metrics['training_start_time']:
        try:
            start_time = datetime.strptime(metrics['training_start_time'], "%Y-%m-%d %H:%M:%S,%f")
            metrics['training_duration'] = str(current_time - start_time)
        except:
            metrics['training_duration'] = "Unknown"
    
    # Save metrics to JSON
    metrics_path = os.path.join(results_dir, 'training_summary.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create a simple text report
    report = f"""
    ===== FEDERATED LEARNING PROGRESS REPORT =====
    Generated at: {current_time}
    
    Training Session:
    - Start Time: {metrics['training_start_time'] or 'N/A'}
    - Duration: {metrics['training_duration'] or 'N/A'}
    - Rounds Completed: {metrics['rounds_completed']}
    - Clients Connected: {metrics['clients_connected']}
    
    Model Performance:
    - Best mAP50: {metrics['model_metrics']['best_mAP50']:.4f}
    - Best Round: {metrics['model_metrics']['best_round']}
    
    System Information:
    - Device: {metrics['system_metrics']['device_used']}
    - CUDA Available: {metrics['system_metrics']['cuda_available']}
    
    ============================================
    """
    
    # Save report to file
    report_path = os.path.join(results_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nReport saved to: {report_path}")
    print(f"Detailed metrics saved to: {metrics_path}")

if __name__ == "__main__":
    analyze_training_progress()