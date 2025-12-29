import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Monkey-patch torch.load
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

torch.serialization.add_safe_globals([DetectionModel, nn.modules.container.Sequential])

try:
    # Initialize a new model with 4 classes
    model = YOLO('yolov8n.yaml')
    model.model.nc = 4
    # Note: Just changing nc after init doesn't rebuild the head.
    # To get a 4-class head, we usually pass nc to the constructor if possible, 
    # but YOLO() doesn't expose it easily for yaml.
    
    # Let's try to load a 'fused' model or see if fusion changes it to 64.
    model = YOLO('yolov8n.pt')
    model.fuse()
    print("After fusion:")
    for k, v in model.model.state_dict().items():
        if 'model.22.cv3.2.1.conv.weight' in k:
            print(f"{k}: {v.shape}")
            
except Exception as e:
    print(f"Error: {e}")
