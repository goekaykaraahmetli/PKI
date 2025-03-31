import sys
import torch
import torch.nn as nn
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
NUM_CLASSES = len(CLASSES)
MODEL_PATH = "best_dual_fighters.pth"  # path to model weights


class DualFighterModel(nn.Module):
    def __init__(self, num_classes):
        super(DualFighterModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(2048*2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, inputs):
        f1, f2 = inputs
        feat1 = self.base_model(f1)
        feat2 = self.base_model(f2)
        combined = torch.cat([feat1, feat2], dim=1)
        out = self.classifier(combined)
        return out

# Define transforms consistent with validation / training transforms
val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(f1_path, f2_path, device='cuda'):
    """
    Load two images (fighter1, fighter2), run inference,
    and return predicted action label.
    """
    # Load images
    f1_img = Image.open(f1_path).convert("RGB")
    f2_img = Image.open(f2_path).convert("RGB")

    # Apply transforms
    f1_tensor = val_transforms(f1_img).unsqueeze(0)  # shape [1,3,224,224]
    f2_tensor = val_transforms(f2_img).unsqueeze(0)

    # Prepare model
    model = DualFighterModel(num_classes=NUM_CLASSES)
    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)

    # Predict
    with torch.no_grad():
        f1_tensor, f2_tensor = f1_tensor.to(device), f2_tensor.to(device)
        outputs = model((f1_tensor, f2_tensor))  # shape [1, num_classes]
        _, pred_idx = torch.max(outputs, dim=1)
        predicted_label = CLASSES[pred_idx.item()]
    return predicted_label

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict_2fighters.py fighter1.jpg fighter2.jpg")
        sys.exit(1)

    fighter1_path = sys.argv[1]
    fighter2_path = sys.argv[2]

    # If no GPU is available, or want to force CPU, set device='cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    action = predict(fighter1_path, fighter2_path, device=device)
    print(f"Predicted action: {action}")
