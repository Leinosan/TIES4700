from __future__ import print_function, division
from fine_tuning_config_file import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import time
import copy
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set device
use_gpu = GPU_MODE
if use_gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        transforms.GaussianBlur(kernel_size=3, p=0.1),
        transforms.RandomErasing(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = DATA_DIR
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val']}

# Handle class imbalance with WeightedRandomSampler
class_counts = np.array([len(np.where(np.array(dsets['train'].targets) == i)[0])
                         for i in range(NUM_CLASSES)])
print(f"Class counts: {dict(zip(dsets['train'].classes, class_counts))}")

class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[label] for label in dsets['train'].targets]
sampler = torch.utils.data.WeightedRandomSampler(
    sample_weights, len(sample_weights))

dset_loaders = {
    'train': torch.utils.data.DataLoader(dsets['train'], batch_size=BATCH_SIZE,
                                         sampler=sampler, num_workers=4, pin_memory=True),
    'val':   torch.utils.data.DataLoader(dsets['val'],   batch_size=BATCH_SIZE,
                                         shuffle=False,  num_workers=4, pin_memory=True),
}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes


def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()
    best_epoch = 0
    best_model = model
    best_acc = 0.0
    patience = 7  # slightly more patience than ResNet18 since it's bigger
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            counter = 0

            for data in dset_loaders[phase]:
                inputs, labels = data
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if counter % 10 == 0:
                    print("Reached iteration ", counter)
                counter += 1

                try:
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels)
                except:
                    print('unexpected error, could not calculate loss or do a sum.')

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects.item() / float(dset_sizes[phase])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val':
                scheduler.step(epoch_loss)
                print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch
                    epochs_no_improve = 0
                    print('new best accuracy = ', best_acc)
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model, best_acc, best_epoch


def main():
    # ResNet50 - deeper than ResNet18, more capacity for complex features
    model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze early layers - ResNet50 has layer1-4 same as ResNet18
    for param in model_ft.parameters():
        param.requires_grad = False

    # Unfreeze last three layer blocks - more than ResNet18 since we have GPU
    for param in model_ft.layer2.parameters():
        param.requires_grad = True
    for param in model_ft.layer3.parameters():
        param.requires_grad = True
    for param in model_ft.layer4.parameters():
        param.requires_grad = True

    # ResNet50's fc layer has 2048 input features (vs 512 in ResNet18)
    num_ftrs = model_ft.fc.in_features  # will be 2048
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    print(f"ResNet50 fc input features: {num_ftrs}")  # confirms 2048
    print(f"Total trainable parameters: "
          f"{sum(p.numel() for p in model_ft.parameters() if p.requires_grad):,}")

    model_ft = model_ft.to(device)

    # Weighted loss for class imbalance
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Slightly lower LR than ResNet18 - ResNet50 is more sensitive to LR
    optimizer_ft = optim.Adam(
        filter(lambda p: p.requires_grad, model_ft.parameters()), lr=BASE_LR * 0.5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, mode='min', factor=0.1, patience=3)

    trained, best_acc, best_epoch = train_model(
        model_ft, criterion, optimizer_ft, scheduler, num_epochs=50)

    torch.save({
        'epoch': best_epoch,
        'model_state_dict': trained.state_dict(),
        'best_acc': best_acc,
        'num_classes': NUM_CLASSES,
        'class_names': dset_classes,
        'architecture': 'resnet50',
    }, 'model_resnet50.pt')


if __name__ == '__main__':
    main()
