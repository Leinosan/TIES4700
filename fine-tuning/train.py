import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

import os

data_root = os.environ.get("LOCAL_SCRATCH", ".")


# -----------------------------
# 1. DATASET & DATALOADERS
# -----------------------------

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=(-0.03, 0.03)),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
            p=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
}

# -----------------------------
# TRAINING FUNCTIONS
# -----------------------------


def is_image_file(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy


# -----------------------------
# MAIN TRAINING SCRIPT
# -----------------------------
if __name__ == "__main__":

    train_tfms = data_transforms['train']
    val_tfms = data_transforms['val']

    train_ds = datasets.ImageFolder(
        "RODI-DATA_split/train",
        transform=train_tfms,
        is_valid_file=is_image_file
    )
    val_ds = datasets.ImageFolder(
        "RODI-DATA_split/val",
        transform=val_tfms,
        is_valid_file=is_image_file
    )

    train_loader = DataLoader(train_ds, batch_size=64,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64,
                            shuffle=False, num_workers=4)

    num_classes = len(train_ds.classes)

    # Load model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Best model tracking
    best_acc = 0.0
    best_model_path = "best_resnet50.pth"

    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # 3. STAGE 1 — HEAD-ONLY TRAINING
    # -----------------------------
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3)

    print("Stage 1: Training classifier head only")
    for epoch in range(3):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_acc:.4f}")

    # -----------------------------
    # 4. STAGE 2 — UNFREEZE LAST BLOCK
    # -----------------------------
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=3e-4, weight_decay=1e-4)

    print("\nStage 2: Unfreezing last ResNet block")
    for epoch in range(5):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_acc:.4f}")

    # -----------------------------
    # 5. STAGE 3 — FULL FINE-TUNING
    # -----------------------------
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print("\nStage 3: Full fine-tuning")
    for epoch in range(12):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_acc:.4f}")
