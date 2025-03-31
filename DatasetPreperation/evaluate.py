import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset & model definitions
from train_dual_fighters import FighterPairDataset, DualFighterModel

def main():
    TEST_DIR   = 'dataset_images/test'
    BATCH_SIZE = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test transforms (same as val transforms)
    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    #Load Dataset
    test_dataset = FighterPairDataset(TEST_DIR, transform=test_transforms)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes (original): {test_dataset.classes}")
    num_classes = len(test_dataset.classes)

    # English labels for confusion matrix
    english_labels = [
        "Block (Left Hand)",
        "Block (Right Hand)",
        "Missed Punch (Left Hand)",
        "Missed Punch (Right Hand)",
        "Punch to the Head (Left Hand)",
        "Punch to the Head (Right Hand)",
        "Punch to the Torso (Left Hand)",
        "Punch to the Torso (Right Hand)"
    ]

    # Load model weights
    model = DualFighterModel(num_classes=num_classes)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load('best_dual_fighters.pth', map_location=DEVICE))
    model.eval()

    #Evaluation Loop
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for (f1_batch, f2_batch), labels in test_loader:
            f1_batch = f1_batch.to(DEVICE)
            f2_batch = f2_batch.to(DEVICE)
            labels   = labels.to(DEVICE)

            outputs = model((f1_batch, f2_batch))
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=english_labels))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Save Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=english_labels, yticklabels=english_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix_english.png")
    plt.show()

if __name__ == '__main__':
    main()
