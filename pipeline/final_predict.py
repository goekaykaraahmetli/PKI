import cv2
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image


CLASSES = [
    "Blok lewą ręką",
    "Blok prawą ręką",
    "Chybienie lewą ręką",
    "Chybienie prawą ręką",
    "Głowa lewą ręką",
    "Głowa prawą ręką",
    "Korpus lewą ręką",
    "Korpus prawą ręką"
]

TRANSLATIONS = {
    "Blok lewą ręką": "Block (Left Hand)",
    "Blok prawą ręką": "Block (Right Hand)",
    "Chybienie lewą ręką": "Miss (Left Hand)",
    "Chybienie prawą ręką": "Miss (Right Hand)",
    "Głowa lewą ręką": "Head Shot (Left Hand)",
    "Głowa prawą ręką": "Head Shot (Right Hand)",
    "Korpus lewą ręką": "Body Shot (Left Hand)",
    "Korpus prawą ręką": "Body Shot (Right Hand)"
}

NUM_CLASSES = len(CLASSES)
CLASSIFIER_WEIGHTS = "../DatasetPreperation/best_dual_fighters.pth"  # path to CNN model weights

# CNN model Definition
class DualFighterModel(nn.Module):
    def __init__(self, num_classes=8):
        super(DualFighterModel, self).__init__()
        # Pretrained ResNet50
        self.base_model = models.resnet50(pretrained=True)
        # Replace final FC with Identity so we get a 2048-dim feature vector
        self.base_model.fc = nn.Identity()
        
        # Then classify 2 * 2048 = 4096 -> 512 -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(2048*2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, inputs):
        f1, f2 = inputs
        feat1 = self.base_model(f1)  # shape: [batch, 2048]
        feat2 = self.base_model(f2)  # shape: [batch, 2048]
        combined = torch.cat([feat1, feat2], dim=1)  # [batch, 4096]
        out = self.classifier(combined)              # [batch, num_classes]
        return out

# The same transforms used at validation / inference for your classifier
classification_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def load_classifier(device='cuda'):
    model = DualFighterModel(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=device))
    model.eval()
    model.to(device)
    return model

# Yolo + CNN on Video
def get_box_color(class_name: str):
    #Returns a distinct BGR color based on the YOLO class name.
    class_name = class_name.lower()
    if class_name == "blue-player":
        return (255, 0, 0)   # Blue in BGR
    elif class_name == "red-player":
        return (0, 0, 255)   # Red in BGR
    elif class_name == "referee":
        return (255, 255, 255) # White in BGR
    else:
        return (0, 255, 0)   # Green fallback

def process_video(input_path, output_path, conf_thresh=0.25, device='cuda'):
    # Load YOLO
    yolo_model = YOLO("runs/detect/train5/weights/best.pt")

    # Load dual-fighter classification model
    classifier_model = load_classifier(device=device)

    FIGHTER_LABELS = {"blue-player", "red-player"}

    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = yolo_model(frame, conf=conf_thresh)
        boxes = results[0].boxes

        # Collect exactly two bounding boxes for fighters
        fighter_boxes = []
        if boxes is not None and len(boxes) > 0:
            detections = boxes.data.cpu().numpy()  # [N, 6]: [x1,y1,x2,y2,conf,cls]
            for det in detections:
                x1, y1, x2, y2, det_conf, cls_id = det
                cls_id = int(cls_id)
                class_name = yolo_model.names[cls_id]
                # Keep only recognized fighters
                if class_name.lower() in FIGHTER_LABELS:
                    fighter_boxes.append((x1, y1, x2, y2, det_conf, class_name))

        # If we have exactly 2 fighter boxes, run classification
        action_label_en = None
        if len(fighter_boxes) == 2:
            # sort by left->right so we consistently do f1, f2
            fighter_boxes.sort(key=lambda b: b[0])  # sort by x1
            (x1a, y1a, x2a, y2a, confA, nameA) = fighter_boxes[0]
            (x1b, y1b, x2b, y2b, confB, nameB) = fighter_boxes[1]

            # Convert coords to int
            x1a, y1a, x2a, y2a = map(int, [x1a, y1a, x2a, y2a])
            x1b, y1b, x2b, y2b = map(int, [x1b, y1b, x2b, y2b])

            # Crop
            f1_crop = frame[max(0, y1a):max(0, y2a), max(0, x1a):max(0, x2a)]
            f2_crop = frame[max(0, y1b):max(0, y2b), max(0, x1b):max(0, x2b)]

            # If valid
            if f1_crop.size > 0 and f2_crop.size > 0:
                f1_pil = Image.fromarray(cv2.cvtColor(f1_crop, cv2.COLOR_BGR2RGB))
                f2_pil = Image.fromarray(cv2.cvtColor(f2_crop, cv2.COLOR_BGR2RGB))

                # transform
                f1_tensor = classification_transform(f1_pil).unsqueeze(0).to(device)
                f2_tensor = classification_transform(f2_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = classifier_model((f1_tensor, f2_tensor))  # [1,num_classes]
                    _, pred_idx = torch.max(outputs, dim=1)
                    # Polish label
                    action_label_pl = CLASSES[pred_idx.item()]
                    # Translate to English
                    action_label_en = TRANSLATIONS.get(action_label_pl, action_label_pl)

        # Draw YOLO bounding boxes (all classes)
        if boxes is not None and len(boxes) > 0:
            for det in boxes.data.cpu().numpy():
                x1, y1, x2, y2, det_conf, cls_id = det
                cls_id = int(cls_id)
                class_name = yolo_model.names[cls_id]
                
                color = get_box_color(class_name)  # New function for color mapping
                label_str = f"{class_name} {det_conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label_str, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Overlay English action label if found
        if action_label_en:
            cv2.putText(frame, f"Action: {action_label_en}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        writer.write(frame)
        cv2.imshow("Annotated Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + Two-Fighter Classification with English Translation")
    parser.add_argument("input_video", help="Path to input video (mp4)")
    parser.add_argument("output_video", help="Path to output mp4 file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    process_video(args.input_video, args.output_video, conf_thresh=args.conf, device=args.device)
    