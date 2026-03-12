# CNN Baseline + ResNet18 Transfer Learning

- **SmallCNN (scratch)** trained from random initialization
- **ResNet18 (ImageNet pretrained)** fine-tuned for fish vs non-fish
- Two learning-rate variants for ResNet18 (LR = 1e-4 and 1e-5)

The original multi-class dataset is converted into **binary classification**:
- `fish = 1`
- `non-fish = 0` (all other classes)

The notebooks report **training and validation results** using the required 8 metrics:
- Precision/Recall/F1 for fish
- Precision/Recall for non-fish
- Accuracy
- AUROC
- AUPRC

---

## Repository structure

- `LR-test/`  
  Contains notebooks:
  - `CNN.ipynb`
  - `ResNet18_LR1e-4.ipynb`
  - `ResNet18_LR1e-5.ipynb`

## Data

Expected folder structure:
`
RODI-DATA/
  Train/
    artifact/
    fish/
    insect/
    leave/
    partial-overlapping-objects/
    rounded/
    shadow/
    variable/
    wormish/
`

Set the path in the config:

- Windows example: `C:\Users\NAME\RODI-DATA\Train`

---

## Environment

Recommended packages (typical):

- Python 3.10+
- torch, torchvision
- numpy
- scikit-learn
- pillow

Install (pip):
```bash
pip install torch torchvision numpy scikit-learn pillow
