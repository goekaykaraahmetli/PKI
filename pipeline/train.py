from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained model

model.train(
    data='./BoxingPlayer/data.yaml',
    epochs=50,
    imgsz=640,
    device=0  # use AMD GPU
)
