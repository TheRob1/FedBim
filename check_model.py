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
    model = YOLO('yolov8n.pt')
    print(f"NC: {model.model.nc}")
    for k, v in model.model.state_dict().items():
        if 'model.22.cv3.2.1.conv.weight' in k:
            print(f"{k}: {v.shape}")
except Exception as e:
    print(f"Error: {e}")
