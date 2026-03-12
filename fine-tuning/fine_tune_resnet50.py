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


def validate(model, loader, criterion, device, print_report=False):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / len(loader.dataset)

    if print_report:
        from sklearn.metrics import classification_report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds,
                                    target_names=loader.dataset.classes))

    return total_loss / len(loader), accuracy


# -----------------------------
# MAIN TRAINING SCRIPT
# -----------------------------
if __name__ == "__main__":

    train_tfms = data_transforms['train']
    val_tfms = data_transforms['val']

    train_ds = datasets.ImageFolder(
        os.path.join(data_root, "RODI-DATA_split/train"),
        transform=train_tfms,
        is_valid_file=is_image_file
    )
    val_ds = datasets.ImageFolder(
        os.path.join(data_root, "RODI-DATA_split/val"),
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

    # -----------------------------
    # 6. FINAL EVALUATION ON BEST MODEL
    # -----------------------------
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
    import numpy as np

    def evaluate(model, loader, criterion, device, class_names):
        model.eval()
        total_loss = 0
        correct = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)

                total_loss += loss.item()
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        accuracy = correct / len(loader.dataset)

        # Per-class report
        print("\nClassification Report:")
        print(classification_report(all_labels,
              all_preds, target_names=class_names))

        # Binary fish vs non-fish
        fish_idx = class_names.index("fish")
        binary_labels = (all_labels == fish_idx).astype(int)
        binary_preds = (all_preds == fish_idx).astype(int)
        fish_probs = all_probs[:, fish_idx]

        fish_tp = ((binary_preds == 1) & (binary_labels == 1)).sum()
        fish_fp = ((binary_preds == 1) & (binary_labels == 0)).sum()
        fish_fn = ((binary_preds == 0) & (binary_labels == 1)).sum()
        non_fish_tp = ((binary_preds == 0) & (binary_labels == 0)).sum()
        non_fish_fp = ((binary_preds == 0) & (binary_labels == 1)).sum()
        non_fish_fn = ((binary_preds == 1) & (binary_labels == 0)).sum()

        p_fish = fish_tp / \
            (fish_tp + fish_fp) if (fish_tp + fish_fp) > 0 else 0
        r_fish = fish_tp / \
            (fish_tp + fish_fn) if (fish_tp + fish_fn) > 0 else 0
        f1_fish = 2 * p_fish * r_fish / \
            (p_fish + r_fish) if (p_fish + r_fish) > 0 else 0

        p_nonfish = non_fish_tp / \
            (non_fish_tp + non_fish_fp) if (non_fish_tp + non_fish_fp) > 0 else 0
        r_nonfish = non_fish_tp / \
            (non_fish_tp + non_fish_fn) if (non_fish_tp + non_fish_fn) > 0 else 0

        auroc = roc_auc_score(binary_labels, fish_probs)
        auprc = average_precision_score(binary_labels, fish_probs)

        print(f"Binary Fish vs Non-Fish Metrics:")
        print(
            f"  P_fish={p_fish:.3f}, R_fish={r_fish:.3f}, F1_fish={f1_fish:.3f}")
        print(f"  P_non_fish={p_nonfish:.3f}, R_non_fish={r_nonfish:.3f}")
        print(
            f"  Accuracy={accuracy:.3f}, AUROC={auroc:.3f}, AUPRC={auprc:.3f}")

        return accuracy

    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    class_names = train_loader.dataset.classes

    print("\n--- Training Set ---")
    evaluate(model, train_loader, criterion, device, class_names)

    print("\n--- Validation Set ---")
    evaluate(model, val_loader, criterion, device, class_names)
