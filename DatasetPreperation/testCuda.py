import torch
from ultralytics import YOLO

model = YOLO("../pipeline/runs/detect/train5/weights/best.pt")
print("YOLO device:", model.device)


print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using device:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")
