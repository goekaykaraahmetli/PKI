import os
import random
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

"""
Pairs two images (fighter1, fighter2) into a single sample, each labeled with the same action class folder.
"""
class FighterPairDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # For each action label (folder)
        for action_label in os.listdir(root_dir):
            action_path = os.path.join(root_dir, action_label)
            if not os.path.isdir(action_path):
                continue

            frame_dict = {}
            for img_file in os.listdir(action_path):
                if img_file.endswith(".jpg"):
                    if "_f1.jpg" in img_file:
                        prefix = img_file.replace("_f1.jpg", "")
                        frame_dict.setdefault(prefix, {})["f1"] = os.path.join(action_path, img_file)
                    elif "_f2.jpg" in img_file:
                        prefix = img_file.replace("_f2.jpg", "")
                        frame_dict.setdefault(prefix, {})["f2"] = os.path.join(action_path, img_file)

            # Keep only pairs that have both f1 and f2
            for prefix, files_dict in frame_dict.items():
                if "f1" in files_dict and "f2" in files_dict:
                    self.data.append((files_dict["f1"], files_dict["f2"], action_label))

        # Gather unique classes
        self.classes = sorted(list({label for _, _, label in self.data}))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        f1_path, f2_path, action_label = self.data[idx]
        f1_img = Image.open(f1_path).convert("RGB")
        f2_img = Image.open(f2_path).convert("RGB")

        if self.transform:
            f1_img = self.transform(f1_img)
            f2_img = self.transform(f2_img)

        label_idx = self.class_to_idx[action_label]

        # Return ((fighter1_img, fighter2_img), label_idx)
        return (f1_img, f2_img), label_idx

"""
Defining Model:
Shared ResNet50 backbone for both images, concatenating the 2 feature vectors, and outputs a single action class.
"""
class DualFighterModel(nn.Module):

    def __init__(self, num_classes):
        super(DualFighterModel, self).__init__()

        # Pretrained ResNet50, unfreeze it
        self.base_model = models.resnet50(pretrained=True)
        # Replace final FC with Identity to get a 2048-dim feature vector
        self.base_model.fc = nn.Identity()

        # After we get (2048) for each fighter, we concat => 4096
        # Then pass through a small MLP for classification
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

"""
Training Loop
"""
def main():
    # Settings
    DATA_DIR = 'dataset_images'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR   = os.path.join(DATA_DIR, 'val')
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LR = 1e-4
    PATIENCE = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Datasets
    train_dataset = FighterPairDataset(TRAIN_DIR, transform=train_transforms)
    val_dataset   = FighterPairDataset(VAL_DIR,   transform=val_transforms)

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    loaders = {'train': train_loader, 'val': val_loader}

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # Build the model
    model = DualFighterModel(num_classes=len(train_dataset.classes))
    model = model.to(DEVICE)

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    trials = 0

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            dataset_size = len(loaders[phase].dataset)

            for (f1_batch, f2_batch), labels in tqdm(loaders[phase], desc=f"{phase} epoch {epoch}", leave=False):
                f1_batch, f2_batch = f1_batch.to(DEVICE), f2_batch.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model((f1_batch, f2_batch))
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Stats
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f"Epoch {epoch} [{phase}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            # Early stopping
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    trials = 0
                    # Save best model
                    torch.save(model.state_dict(), 'best_dual_fighters.pth')
                    print("  [*] Model improved, saving weights.")
                else:
                    trials += 1
                    if trials >= PATIENCE:
                        print("Early stopping!")
                        break
        else:
            # Only executed if the inner loop didn't break
            continue
        break  # Break outer loop if early stopping triggered

    print(f"Training complete. Best val accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()
