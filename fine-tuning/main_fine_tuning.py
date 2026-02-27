from __future__ import print_function, division
from fine_tuning_config_file import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# If you want to keep a track of your network on tensorboard, set USE_TENSORBOARD TO 1 in config file.
if USE_TENSORBOARD:
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=TENSORBOARD_SERVER)
    try:
        cc.remove_experiment(EXP_NAME)
    except:
        pass
    foo = cc.create_experiment(EXP_NAME)


# Set device - uses MPS on Apple Silicon, falls back to CPU
use_gpu = GPU_MODE
if use_gpu:
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

count = 0

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.1),
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = DATA_DIR
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes


def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=30):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            counter = 0

            for data in dset_loaders[phase]:
                inputs, labels = data

                # Move data to device (works for MPS, CUDA, or CPU)
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
                if USE_TENSORBOARD:
                    foo.add_scalar_value('epoch_loss', epoch_loss, step=epoch)
                    foo.add_scalar_value('epoch_acc', epoch_acc, step=epoch)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)
                    print('new best accuracy = ', best_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model


def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (DECAY_WEIGHT ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def main():
    model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    # Move model and criterion to device
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=0.0001)

    trained = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                          num_epochs=30)

    torch.save(trained.state_dict(), 'fine_tuned_best_model.pt')


if __name__ == '__main__':
    main()
