import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import yaml
import os

# Monkey-patch torch.load
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

torch.serialization.add_safe_globals([DetectionModel, nn.modules.container.Sequential])

def main():
    # 1. Create a dummy data.yaml with 4 classes
    data_yaml = {
        'train': '606_train_152_val_12_test/train/images',
        'val': '606_train_152_val_12_test/valid/images',
        'nc': 4,
        'names': ['blanket', 'call_point', 'detector', 'extinguisher']
    }
    with open('temp_init_data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    
    print("Initializing model with 4 classes...")
    # 2. Load the pretrained model
    model = YOLO('yolov8n.pt')
    
    # 3. Use the 'train' method setup logic to rebuild the head without actually training much
    # We can use 1 epoch but stop it or just use the internal 'trainer' to build.
    # Actually, if we just call model.train(data='temp_init_data.yaml', epochs=1, imgsz=640, device='cpu', batch=1)
    # it will rebuild the head.
    
    # To avoid actual training, we can just do:
    model.train(data='temp_init_data.yaml', epochs=1, imgsz=640, device='cpu', batch=2, plots=False, save=False, exist_ok=True)
    
    # 4. Save the new base model
    model.save('yolov8n_nc4.pt')
    print("Saved yolov8n_nc4.pt")

    # Cleanup
    if os.path.exists('temp_init_data.yaml'):
        os.remove('temp_init_data.yaml')

if __name__ == "__main__":
    main()
