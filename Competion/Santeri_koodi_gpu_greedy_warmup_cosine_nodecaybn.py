# %% [markdown]
# ## 0) Setup

# !!!! Comments were generated using ChatGPT. !!!!
# %%
import os
import random
import time
import math
from pathlib import Path
from glob import glob
import csv
import json

from pprint import pformat
from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
from PIL import Image

from typing import Optional, Union, List

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

# (Optional) helps reduce fragmentation / OOMs on some workloads (ignored if unsupported)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


# %% [markdown]
# ## 1) Configuration

# %%
# ===== User config =====
DATA_ROOT = Path("C:/Users/leino/Documents/JYU Opinnot/TIES4700 - Deep Learning/RODI-DATA")
TRAIN_DIR = DATA_ROOT / "Train_aug"

# Model options:
#   - "resnet18"
#   - "resnet50"
#   - "convnext_small"
#   - "convnext_base"
#   - "convnext_large"
MODEL_NAME = "resnet50"
IMG_SIZE = 224

# "Greedy GPU" knobs
BATCH_SIZE = 256   # 128 for convnext_base, 256 resnet50

NUM_EPOCHS = 20
LR = 5e-4
WEIGHT_DECAY = 3e-4

# LR schedule:
WARMUP_EPOCHS = 0.5
MIN_LR = 1e-6

# DataLoader throughput
NUM_WORKERS = 4
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = True

# CUDA performance
USE_TF32 = True                      # for Ampere+ GPUs; safe for most vision models
USE_AMP = True                       # bf16 if supported else fp16
USE_CHANNELS_LAST = True             # helps ConvNeXt/ResNet throughput on CUDA
USE_TORCH_COMPILE = False            # PyTorch 2+; can increase compile-time & VRAM usage
TORCH_COMPILE_MODE = "max-autotune"  # or "default", "reduce-overhead"

SEED = 42

# Best-checkpoint metric (REQUIRED): F1 on fish with 0.5 threshold
BEST_METRIC_NAME = "f1_fish"

# Where to save outputs
OUT_DIR = Path("Competion/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Save best model weights (state_dict) here:
CKPT_PATH = OUT_DIR / f"model_{MODEL_NAME}.pt"

# Optional metadata about the best checkpoint
BEST_META_PATH = OUT_DIR / f"model_{MODEL_NAME}_meta.json"

print("TRAIN_DIR:", TRAIN_DIR.resolve())
print("MODEL_NAME:", MODEL_NAME)
print("CKPT_PATH:", CKPT_PATH.resolve())


# %% [markdown]
# ## 2) Reproducibility + device + performance flags

# %%
def seed_everything(seed: int = 42):
    # Make Python's built-in RNG deterministic
    random.seed(seed)
    # Make NumPy RNG deterministic
    np.random.seed(seed)
    # Seed PyTorch RNG for CPU ops
    torch.manual_seed(seed)
    # Seed all CUDA RNGs (all GPUs) if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN determinism OFF + benchmark ON favors speed (not strict reproducibility)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# Apply seeding using the experiment's global SEED
seed_everything(SEED)

# Pick GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" device selected ->", device)

if device.type == "cuda":
    try:
        # Enable/disable TF32 acceleration on Ampere+ GPUs (faster matmul/conv, slight precision tradeoff)
        torch.backends.cuda.matmul.allow_tf32 = bool(USE_TF32)
        torch.backends.cudnn.allow_tf32 = bool(USE_TF32)
    except Exception as e:
        # Older PyTorch / builds may not expose TF32 flags
        print("TF32 flags not available:", e)

    try:
        # Control float32 matmul precision policy (PyTorch 2.x+)
        torch.set_float32_matmul_precision("high" if USE_TF32 else "highest")
    except Exception:
        # Ignore if this API isn't present
        pass

# Choose AMP dtype
def get_amp_dtype() -> Optional[torch.dtype]:
    # AMP only makes sense when enabled and running on CUDA
    if not (USE_AMP and device.type == "cuda"):
        return None
    try:
        # Prefer bf16 when supported (typically more stable than fp16, no GradScaler needed)
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        # If the capability check isn't available, fall back to fp16
        pass
    # Default AMP dtype on CUDA when bf16 isn't supported
    return torch.float16

# Resolve AMP dtype once and log the choice
AMP_DTYPE = get_amp_dtype()
print("AMP enabled:", bool(AMP_DTYPE), "| dtype:", AMP_DTYPE)

# GradScaler is needed only for fp16 AMP to avoid underflow; bf16 typically doesn't require scaling
USE_SCALER = bool(AMP_DTYPE == torch.float16 and device.type == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER) if device.type == "cuda" else None


# %% [markdown]
# ## 3) Transforms: padding to square (black) -> resize -> (train-only: random horizontal flip) -> to tensor -> normalize

# %%
# Standard ImageNet normalization stats (works well for many pretrained-style pipelines)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class PadToSquare:
    def __init__(self, fill=0):
        # Fill color used for padding (0 = black)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        # Ensure consistent 3-channel input
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        # No-op if already square
        if w == h:
            return img
        # Pad the shorter side to match the longer one
        size = max(w, h)
        new_img = Image.new("RGB", (size, size), (self.fill, self.fill, self.fill))
        # Center the original image in the new square canvas
        left = (size - w) // 2
        top = (size - h) // 2
        new_img.paste(img, (left, top))
        return new_img

# Training transforms: pad->resize->augment (flip)->tensor->normalize
train_tfms = transforms.Compose([
    PadToSquare(fill=0),
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Validation transforms: deterministic (no random augmentation)
val_tfms = transforms.Compose([
    PadToSquare(fill=0),
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# %% [markdown]
# ## 4) Models

# %%
def _make_resnet_binary(variant: str) -> nn.Module:
    # Normalize model name for robust matching
    variant = variant.lower().strip()
    # Build chosen ResNet backbone (no pretrained weights)
    if variant == "resnet18":
        m = torchvision.models.resnet18(weights=None)
    elif variant == "resnet50":
        m = torchvision.models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported ResNet variant: {variant}")
    # Replace the classification head with a single-logit output for binary classification
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, 1)
    return m


def _make_convnext_binary(variant: str) -> nn.Module:
    # Normalize model name
    variant = variant.lower().strip()
    # Allow a few friendly aliases / known names
    name_map = {
        "convnext_small": "convnext_small",
        "convnext_base": "convnext_base",
        "convnext_large": "convnext_large",
    }
    fn_name = name_map.get(variant, variant)
    # Ensure the requested ConvNeXt constructor exists in this torchvision version
    if not hasattr(torchvision.models, fn_name):
        raise ValueError(
            f"ConvNeXt variant '{variant}' not available in this torchvision ({torchvision.__version__}). "
            f"Available variants might include: convnext_tiny/small/base/large."
        )
    fn = getattr(torchvision.models, fn_name)
    # Instantiate backbone (no pretrained weights)
    m = fn(weights=None)
    # ConvNeXt classifier is expected to be an nn.Sequential
    if not hasattr(m, "classifier") or not isinstance(m.classifier, nn.Sequential):
        raise RuntimeError("Unexpected ConvNeXt model structure: missing 'classifier' Sequential")
    last = m.classifier[-1]
    # Last classifier module should be Linear for easy head replacement
    if not isinstance(last, nn.Linear):
        raise RuntimeError("Unexpected ConvNeXt classifier: last layer is not nn.Linear")
    # Swap final layer to a single-logit output for BCEWithLogitsLoss
    in_features = last.in_features
    m.classifier[-1] = nn.Linear(in_features, 1)
    return m


def create_model(model_name: str) -> nn.Module:
    # Unified model factory handling common name variants
    name = model_name.lower().strip()
    if name in {"resnet18", "resnet-18", "resnet50", "resnet-50"}:
        return _make_resnet_binary(name.replace("-", ""))
    if name.startswith("convnext"):
        return _make_convnext_binary(name)
    raise ValueError(f"Unknown model_name={model_name!r}.")


# %% [markdown]
# ## 4.5) AdamW parameter groups

# %%
def build_adamw_param_groups(model: nn.Module, weight_decay: float):
    # Split parameters into: apply weight decay vs. skip weight decay
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        # Skip frozen params
        if not p.requires_grad:
            continue
        n = name.lower()
        # Common heuristic: no decay for biases, 1D params, and norm/bn layers
        if p.ndim <= 1 or n.endswith(".bias") or ("bn" in n) or ("norm" in n):
            no_decay.append(p)
        else:
            decay.append(p)

    # Build optimizer param_groups with different weight_decay values
    param_groups = []
    if decay:
        param_groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        param_groups.append({"params": no_decay, "weight_decay": 0.0})
    return param_groups


# %% [markdown]
# ## 5) Metrics

# %%
def confusion_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    # Convert inputs to expected dtypes
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float32)
    # Threshold probabilities to get binary predictions
    y_pred = (y_prob >= threshold).astype(np.int64)
    # Compute confusion matrix counts
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn

def safe_div(a, b):
    # Avoid ZeroDivisionError; return 0 when denominator is 0
    return float(a) / float(b) if b != 0 else 0.0

def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    # Derive confusion counts from probabilistic predictions
    tp, tn, fp, fn = confusion_from_probs(y_true, y_prob, threshold=threshold)

    # "fish" treated as positive class (1)
    prec_fish = safe_div(tp, tp + fp)
    rec_fish  = safe_div(tp, tp + fn)
    f1_fish = safe_div(2.0 * prec_fish * rec_fish, (prec_fish + rec_fish))

    # Metrics for the negative class (non-fish) using symmetric definitions
    prec_nonfish = safe_div(tn, tn + fn)
    rec_nonfish  = safe_div(tn, tn + fp)

    # Overall accuracy
    acc = safe_div(tp + tn, tp + tn + fp + fn)

    # AUROC/AUPRC (try sklearn first, then torchmetrics, else NaN)
    auroc = None
    auprc = None
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        # Only defined when both classes exist in y_true
        if len(np.unique(y_true)) == 2:
            auroc = float(roc_auc_score(y_true, y_prob))
            auprc = float(average_precision_score(y_true, y_prob))
        else:
            auroc = float("nan")
            auprc = float("nan")
    except Exception:
        try:
            from torchmetrics.functional.classification import (
                binary_auroc as tm_binary_auroc,
                binary_average_precision as tm_binary_auprc,
            )
            yt = torch.tensor(y_true, dtype=torch.int64)
            yp = torch.tensor(y_prob, dtype=torch.float32)
            auroc = float(tm_binary_auroc(yp, yt).item())
            auprc = float(tm_binary_auprc(yp, yt).item())
        except Exception:
            # If neither backend is available, mark as NaN
            auroc = float("nan")
            auprc = float("nan")

    # Return a single dict with headline metrics + confusion counts
    return {
        "precision_fish": prec_fish,
        "recall_fish": rec_fish,
        "f1_fish": f1_fish,
        "precision_nonfish": prec_nonfish,
        "recall_nonfish": rec_nonfish,
        "accuracy": acc,
        "auroc": auroc,
        "auprc": auprc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

# Keys to print/log in a compact report
METRIC_PRINT_KEYS = (
    "precision_fish",
    "recall_fish",
    "f1_fish",
    "precision_nonfish",
    "recall_nonfish",
    "accuracy",
    "auroc",
    "auprc",
)

def metrics_report(metrics: dict) -> dict:
    # Filter metrics dict to only the commonly reported fields
    return {k: metrics[k] for k in METRIC_PRINT_KEYS}


# %% [markdown]
# ## 6) Training + evaluation loops (AMP + channels_last)

# %%
def get_binary_label_from_imagefolder_label(lbl: torch.Tensor, fish_class_index: int) -> torch.Tensor:
    # Convert ImageFolder multi-class label into binary (fish=1, other=0)
    return (lbl == fish_class_index).long()

def _to_device_batch(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Move images to target device; non_blocking helps when pinned memory is enabled
    images = images.to(device, non_blocking=True)
    # Optionally use channels_last for better GPU memory access on conv nets
    if device.type == "cuda" and USE_CHANNELS_LAST:
        images = images.to(memory_format=torch.channels_last)
    return images

def _maybe_channels_last_model(model: nn.Module, device: torch.device) -> nn.Module:
    # Ensure model weights/layout match channels_last when enabled
    if device.type == "cuda" and USE_CHANNELS_LAST:
        return model.to(memory_format=torch.channels_last)
    return model

def _autocast_ctx():
    # Return an autocast context manager configured for this run
    if device.type != "cuda" or AMP_DTYPE is None:
        # Disabled autocast context (keeps call sites uniform)
        return torch.autocast(device_type="cpu", enabled=False)
    # Enabled autocast on CUDA with chosen dtype
    return torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True)

@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, fish_class_index: int, device: torch.device):
    # Evaluation mode disables dropout, uses running stats for norms, etc.
    model.eval()
    probs_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    loss_all: List[float] = []
    criterion = nn.BCEWithLogitsLoss()

    for images, labels in loader:
        # Move batch to device (and channels_last if enabled)
        images = _to_device_batch(images, device)
        # Build binary labels and move to device
        y = get_binary_label_from_imagefolder_label(labels, fish_class_index).float().to(device, non_blocking=True)

        with _autocast_ctx():
            # Model outputs one logit per sample; squeeze to shape [B]
            logits = model(images).squeeze(1)  # [B]
            loss = criterion(logits, y)
            # Convert logits to probabilities for metric computation
            prob = torch.sigmoid(logits)

        # Accumulate outputs on CPU for numpy-based metric computation
        probs_all.append(prob.detach().float().cpu().numpy())
        y_all.append(y.detach().float().cpu().numpy())
        loss_all.append(float(loss.item()))

    # Concatenate across batches; handle empty loader defensively
    y_true = np.concatenate(y_all) if y_all else np.array([])
    y_prob = np.concatenate(probs_all) if probs_all else np.array([])
    # Compute metrics at fixed threshold 0.5
    m = metrics_from_probs(y_true.astype(np.int64), y_prob.astype(np.float64), threshold=0.5)
    # Add mean BCE loss over batches
    m["loss"] = float(np.mean(loss_all)) if loss_all else float("nan")
    return m, y_true, y_prob

def run_train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                        fish_class_index: int, device: torch.device, epoch: int,
                        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None):
    # Training mode enables dropout, updates norm stats, etc.
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    n = 0

    for step, (images, labels) in enumerate(loader, start=1):
        images = _to_device_batch(images, device)
        y = get_binary_label_from_imagefolder_label(labels, fish_class_index).float().to(device, non_blocking=True)

        # Clear gradients efficiently (set_to_none saves memory/overhead)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_ctx():
            # Forward pass produces logits; loss uses logits directly
            logits = model(images).squeeze(1)
            loss = criterion(logits, y)

        # Backprop + optimizer step (optionally using GradScaler for fp16 AMP)
        if device.type == "cuda" and USE_SCALER and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        # Step LR scheduler per-optimizer-step (not per-epoch)
        if scheduler is not None:
            scheduler.step()

        # Track epoch-average loss (weighted by batch size)
        bs = images.size(0)
        running_loss += float(loss.item()) * bs
        n += bs

        # Periodic progress logging
        if step == 1 or step % 50 == 0:
            print(f"    [epoch {epoch:03d}] step {step:04d} | loss {loss.item():.4f}")

    # Return mean loss over all samples
    return running_loss / max(n, 1)

def select_best_metric(metrics: dict, metric_name: str) -> float:
    # Pick a single scalar metric for "best checkpoint" selection
    if metric_name not in metrics:
        raise KeyError(f"Metric {metric_name!r} not found in: {list(metrics.keys())}")
    val = metrics[metric_name]
    # Treat missing/None/NaN as invalid so it won't win
    if val is None:
        return float("-inf")
    try:
        if math.isnan(val):
            return float("-inf")
    except Exception:
        pass
    return float(val)


# %% [markdown]
# ## 6.5) LR schedule:

# %%
def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    num_epochs: int,
    peak_lr: float,
    min_lr: float = 1e-6,
    warmup_epochs: int = 1,
):
    # Total number of optimizer steps across training
    total_steps = int(steps_per_epoch) * int(num_epochs)
    # Warmup steps (at least 1), capped to avoid exceeding total_steps
    warmup_steps = max(1, int(steps_per_epoch) * int(warmup_epochs))
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))

    # Multiplicative factor corresponding to min_lr at the end of cosine decay
    min_mult = float(min_lr) / float(peak_lr)

    def lr_lambda(step: int):
        # step is 0-based and counts optimizer steps
        if step < warmup_steps:
            # Linear warmup from ~0 to 1.0
            return float(step + 1) / float(warmup_steps)
        # cosine decay for the remaining steps
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale cosine output into [min_mult, 1.0]
        return min_mult + (1.0 - min_mult) * cosine

    # LambdaLR applies lr = base_lr * lr_lambda(step)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)


# %% [markdown]
# ## 7) Data: stratified split + DataLoaders

# %%
def get_image_group_id(img_path: Union[str, Path]) -> str:
    # Group augmented/derived files back to their original image id (prevents leakage)
    name = Path(img_path).name
    if "__orig" in name:
        return name.split("__orig", 1)[0]
    if "__aug__" in name:
        return name.split("__aug__", 1)[0]
    # Default group id = filename stem
    return Path(name).stem

def stratified_group_split_indices(
    paths: List[Union[str, Path]],
    y: np.ndarray,
    train_frac: float = 0.8,
    seed: int = 42
):
    # Build mapping from group_id -> sample indices
    group_to_indices = {}
    for idx, p in enumerate(paths):
        gid = get_image_group_id(p)
        group_to_indices.setdefault(gid, []).append(idx)

    # Derive one label per group (must be consistent within group)
    group_ids = sorted(group_to_indices.keys())
    group_labels = []
    for gid in group_ids:
        labels_in_group = np.unique(y[group_to_indices[gid]])
        if len(labels_in_group) != 1:
            # Prevent mixing labels inside a group (would break stratification assumptions)
            raise ValueError(f"Mixed labels found inside group {gid!r}: {labels_in_group.tolist()}")
        group_labels.append(int(labels_in_group[0]))
    group_labels = np.asarray(group_labels, dtype=np.int64)

    try:
        # Preferred: sklearn stratified split at group level
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_frac, random_state=seed)
        train_group_idx, val_group_idx = next(sss.split(np.zeros(len(group_ids)), group_labels))
    except Exception:
        # Fallback: manual stratified split by shuffling indices of each class
        rng = np.random.default_rng(seed)
        idx0 = np.where(group_labels == 0)[0]
        idx1 = np.where(group_labels == 1)[0]
        rng.shuffle(idx0); rng.shuffle(idx1)
        n0_train = int(len(idx0) * train_frac)
        n1_train = int(len(idx1) * train_frac)
        train_group_idx = np.concatenate([idx0[:n0_train], idx1[:n1_train]])
        val_group_idx = np.concatenate([idx0[n0_train:], idx1[n1_train:]])
        rng.shuffle(train_group_idx); rng.shuffle(val_group_idx)

    # Convert group-level split back to sample-level indices
    train_group_ids = {group_ids[i] for i in train_group_idx}
    val_group_ids   = {group_ids[i] for i in val_group_idx}

    train_idx = [idx for gid in train_group_ids for idx in group_to_indices[gid]]
    val_idx   = [idx for gid in val_group_ids for idx in group_to_indices[gid]]

    return train_idx, val_idx

def make_loader(ds, batch_size: int, shuffle: bool) -> DataLoader:
    # pin_memory speeds up host->GPU transfers; only helpful with CUDA
    pin = (device.type == "cuda")
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(NUM_WORKERS),
        pin_memory=pin,
        drop_last=False,
    )
    # Only set these when using worker processes
    if int(NUM_WORKERS) > 0:
        kwargs["prefetch_factor"] = int(PREFETCH_FACTOR)
        kwargs["persistent_workers"] = bool(PERSISTENT_WORKERS)
    return DataLoader(ds, **kwargs)


# %% [markdown]
# ## 8) Main: train + save best

# %%
def main():
    # Ensure dataset path exists before constructing datasets/loaders
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"TRAIN_DIR not found: {TRAIN_DIR.resolve()}")

    # Full dataset (used to read class_to_idx and targets consistently)
    full_ds = torchvision.datasets.ImageFolder(root=str(TRAIN_DIR), transform=val_tfms)
    class_to_idx = full_ds.class_to_idx
    print("Classes found:", class_to_idx)

    # Require a 'fish' folder and treat it as the positive class
    if "fish" not in class_to_idx:
        raise ValueError(
            f"Could not find a 'fish' folder under {TRAIN_DIR}. Found: {list(class_to_idx.keys())}"
        )
    fish_idx = class_to_idx["fish"]

    # Convert ImageFolder targets into binary labels (fish=1, other=0)
    targets = np.array(full_ds.targets)
    binary_targets = (targets == fish_idx).astype(np.int64)
    image_paths = [p for p, _ in full_ds.samples]

    # Stratified split at group level to avoid leakage from augmented versions
    train_idx, val_idx = stratified_group_split_indices(
        image_paths,
        binary_targets,
        train_frac=0.8,
        seed=SEED,
    )

    # Separate datasets so train set can use augmentations while eval/val stay deterministic
    train_ds = torchvision.datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_tfms)
    train_eval_ds = torchvision.datasets.ImageFolder(root=str(TRAIN_DIR), transform=val_tfms)
    val_ds   = torchvision.datasets.ImageFolder(root=str(TRAIN_DIR), transform=val_tfms)

    # Apply index splits via Subset
    train_subset = Subset(train_ds, train_idx)
    train_eval_subset = Subset(train_eval_ds, train_idx)
    val_subset   = Subset(val_ds, val_idx)

    # Print split sizes and class balance
    train_y = binary_targets[train_idx]
    val_y   = binary_targets[val_idx]
    print(f"Train size: {len(train_subset)}  (fish: {int(train_y.sum())}, non-fish: {int((train_y==0).sum())})")
    print(f"Val size  : {len(val_subset)}    (fish: {int(val_y.sum())}, non-fish: {int((val_y==0).sum())})")

    def make_optimizer(m: nn.Module):
        # AdamW with decoupled weight decay + param grouping for norms/biases
        return torch.optim.AdamW(build_adamw_param_groups(m, WEIGHT_DECAY), lr=LR)

    batch_size = BATCH_SIZE

    # Build loaders (train shuffled; eval/val not shuffled)
    train_loader = make_loader(train_subset, batch_size=batch_size, shuffle=True)
    train_eval_loader = make_loader(train_eval_subset, batch_size=batch_size, shuffle=False)
    val_loader   = make_loader(val_subset, batch_size=batch_size, shuffle=False)

    print("Using batch_size:", batch_size)
    print("NUM_WORKERS:", NUM_WORKERS, "| pin_memory:", device.type == "cuda")

    # Create model and move to device (plus optional channels_last)
    model = create_model(MODEL_NAME).to(device)
    model = _maybe_channels_last_model(model, device)

    # Optional torch.compile for speed (PyTorch 2.x, CUDA only here)
    if USE_TORCH_COMPILE and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=TORCH_COMPILE_MODE)
            print("torch.compile enabled | mode:", TORCH_COMPILE_MODE)
        except Exception as e:
            # Fall back gracefully if compile fails
            print("torch.compile failed; continuing uncompiled:", repr(e))

    optimizer = make_optimizer(model)

    # Warmup + cosine learning rate scheduler (stepped every optimizer step)
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        steps_per_epoch=len(train_loader),
        num_epochs=NUM_EPOCHS,
        peak_lr=LR,
        min_lr=MIN_LR,
        warmup_epochs=WARMUP_EPOCHS,
    )

    # Basic model info
    print(model.__class__.__name__)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Quick sanity check to ensure the dataloader yields correctly shaped batches
    print("Trying to fetch one batch...")
    images, labels = next(iter(train_loader))
    print("Batch OK:", images.shape, labels.shape, labels[:10])

    # Track best checkpoint according to BEST_METRIC_NAME
    best_metric = float("-inf")
    best_epoch = -1

    all_time = 0.0
    print("===> Starting: training loop")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n====> Epoch {epoch}/{NUM_EPOCHS} (MODEL={MODEL_NAME})")

        # One epoch of optimization
        t0 = time.time()
        train_loss = run_train_one_epoch(model, train_loader, optimizer, fish_idx, device, epoch, scheduler=scheduler)
        if device.type == "cuda":
            # Ensure GPU work finishes before timing/printing
            torch.cuda.synchronize()
        print(f"===> Train done | loss={train_loss:.4f} | time={time.time()-t0:.1f}s")
        all_time += time.time() - t0

        # Log current LR (best-effort)
        try:
            print(f"LR now: {optimizer.param_groups[0]['lr']:.2e}")
        except Exception:
            pass

        # Evaluate on train split (deterministic transforms) to monitor overfitting
        print("===> Starting: training metrics")
        train_metrics, _, _ = run_eval(model, train_eval_loader, fish_idx, device)
        print(f"Train loss (optimization): {train_loss:.4f}")
        print(f"Train eval loss: {train_metrics['loss']:.4f}")
        print("Train metrics:")
        print(pformat(metrics_report(train_metrics), sort_dicts=False))

        # Evaluate on validation set for model selection
        print("===> Starting: validation")
        val_metrics, _, _ = run_eval(model, val_loader, fish_idx, device)
        print(f"Val loss: {val_metrics['loss']:.4f}")
        print("Validation metrics:")
        print(pformat(metrics_report(val_metrics), sort_dicts=False))

        # Choose the metric used to decide "best model"
        metric_value = select_best_metric(val_metrics, BEST_METRIC_NAME)

        if metric_value > best_metric:
            # Update best and persist checkpoint + run metadata
            best_metric = metric_value
            best_epoch = epoch

            # Save only weights (state_dict)
            torch.save(model.state_dict(), CKPT_PATH)

            # Save metadata JSON alongside the checkpoint for reproducibility
            BEST_META_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(BEST_META_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": int(epoch),
                        "best_metric": float(best_metric),
                        "metric_name": BEST_METRIC_NAME,
                        "model_name": MODEL_NAME,
                        "class_to_idx": class_to_idx,
                        "batch_size": int(batch_size),
                        "learning_rate": float(LR),
                        "weight_decay": float(WEIGHT_DECAY),
                        "warmup_epochs": float(WARMUP_EPOCHS),
                        "min_lr": float(MIN_LR),
                        "number_of_workers": int(NUM_WORKERS),
                        "training_time": float(all_time),
                        "amp_dtype": str(AMP_DTYPE),
                        "tf32": bool(USE_TF32),
                        "channels_last": bool(USE_CHANNELS_LAST),
                        "torch_compile": bool(USE_TORCH_COMPILE and hasattr(torch, "compile")),
                    },
                    f,
                    indent=2,
                )

            print(f"===> New best! Saved weights -> {CKPT_PATH} (best {BEST_METRIC_NAME}={best_metric:.6f})")
            print(f"===> Saved metadata -> {BEST_META_PATH}")

    # Final summary after training ends
    print("\n===> Training finished")
    print("Best epoch:", best_epoch, "| best", BEST_METRIC_NAME, "=", best_metric)
    print("Best weights file:", CKPT_PATH.resolve())

if __name__ == "__main__":
    main()
