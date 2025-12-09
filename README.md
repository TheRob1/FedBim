# Federated Learning for Fire Safety Equipment Detection

This project implements a federated learning system for detecting fire safety equipment using YOLOv8 and the Flower framework. The system allows multiple clients to collaboratively train a model without sharing their raw data.

## Features

- Federated learning with YOLOv8 for object detection
- Support for multiple clients with different data distributions
- Model comparison with existing models in the Visual_Annotation_Tool
- Comprehensive evaluation metrics and visualization
- Easy-to-use configuration system

## Prerequisites

- Python 3.8+
- PyTorch 1.8.0+
- Ultralytics YOLOv8
- Flower Framework
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd FedBim
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
FedBim/
├── 606_train_152_val_12_test/     # Dataset (train/val/test splits)
│   ├── train/                     # Training images and labels
│   ├── valid/                     # Validation images and labels
│   └── test/                      # Test images and labels
├── Visual_Annotation_Tool/        # Existing models for comparison
├── client.py                      # Federated learning client
├── server.py                      # Federated learning server
├── compare_models.py              # Script to compare model performance
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Configuration

Edit the `config.yaml` file to configure:
- Federated learning parameters (number of rounds, clients, etc.)
- YOLOv8 model hyperparameters
- Dataset paths
- Training settings

## Usage

### 1. Prepare the Data

Ensure your dataset is organized in the following structure:
```
606_train_152_val_12_test/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 2. Start the Federated Learning Server

Run the following command to start the federated learning server:
```bash
python server.py
```

### 3. Start the Clients

Open new terminal windows and start the clients. Each client should have a unique ID:

```bash
# Client 1
python client.py --cid 1

# Client 2 (in a new terminal)
python client.py --cid 2
```

### 4. Monitor Training

Training progress will be displayed in the server and client terminals. Model checkpoints and evaluation results will be saved in the `runs/` directory.

### 5. Compare Models

After training, you can compare the federated model with existing models using:

```bash
python compare_models.py
```

This will generate comparison plots and metrics in the `comparison_results/` directory.

## Model Comparison

The `compare_models.py` script will compare the federated learning model with existing models in the `Visual_Annotation_Tool` directory. It will generate:
- Precision, recall, and mAP comparison plots
- Model size and speed metrics
- Detailed metrics in JSON format

## Customization

- **Model Architecture**: Change the model architecture in `config.yaml` by modifying the `yolo.model` parameter.
- **Training Parameters**: Adjust learning rate, batch size, and other hyperparameters in `config.yaml`.
- **Data Splitting**: Modify the client data distribution in the `_setup_client_data` method of `client.py`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Flower Framework](https://flower.dev/)
- [PyTorch](https://pytorch.org/)
