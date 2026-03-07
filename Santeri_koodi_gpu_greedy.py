# -*- coding: utf-8 -*-
"""
GPU-greedy training/inference script.

Converted & optimized from: Santeri_koodi_modified_clean.py
Date: 2026-03-05

Main performance changes vs the original:
- Enables TF32 (Ampere+) and high matmul precision for faster fp32 math
- Uses Automatic Mixed Precision (bf16 if supported, else fp16) to increase throughput
- Uses channels_last memory format (often faster for CNN/ConvNeXt on CUDA)
- Improves DataLoader throughput (workers/pin_memory/prefetch/persistent_workers)
- Optional torch.compile (PyTorch 2+) for extra speed
- Optional batch-size autotuning to use as much VRAM as possible without OOM

Tip: If you get instability/NaNs, set USE_AMP=False and/or USE_TF32=False.
"""

# %% [markdown]
# ## 0) Setup

# %%
import os
import random
import time
import math
from pathlib import Path
from glob import glob
import csv
import json

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
DATA_ROOT = Path(r"C:\Users\leino\Documents\JYU Opinnot\TIES4700 - Deep Learning\RODI-DATA")
TRAIN_DIR = DATA_ROOT / "Train_iso_aug"
TEST_DIR  = DATA_ROOT / "Test"   # images directly inside this folder

# Model options:
#   - "cnn"
#   - "resnet18"
#   - "resnet50"
#   - "convnext" (alias for convnext_tiny)
#   - "convnext_tiny"
#   - "convnext_small"
#   - "convnext_base"
#   - "convnext_large"
MODEL_NAME = "convnext_large"
IMG_SIZE = 224

# "Greedy GPU" knobs (safe defaults; flip to taste)
AUTO_TUNE_BATCH_SIZE = False  # try to find the largest batch size that fits in VRAM (only on CUDA)
BATCH_SIZE = 64                  # used if AUTO_TUNE_BATCH_SIZE=False
BATCH_SIZE_CANDIDATES = [256, 192, 160, 128, 96, 64, 48, 32, 24, 16, 8]

NUM_EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4

# DataLoader throughput (requires __main__ guard on Windows; this script has it)
NUM_WORKERS = 4
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = True

# CUDA performance
USE_TF32 = True                 # for Ampere+ GPUs; safe for most vision models
USE_AMP = True                 # bf16 if supported else fp16
USE_CHANNELS_LAST = True        # helps ConvNeXt/ResNet throughput on CUDA
USE_TORCH_COMPILE = False       # PyTorch 2+; can increase compile-time & VRAM usage
TORCH_COMPILE_MODE = "max-autotune"  # or "default", "reduce-overhead"

SEED = 42

# Best-checkpoint metric (REQUIRED): F1 on fish with 0.5 threshold
BEST_METRIC_NAME = "f1_fish"

# Where to save outputs
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Save best model weights (state_dict) here:
CKPT_PATH = OUT_DIR / f"mode_{MODEL_NAME}.pt"

# Optional metadata about the best checkpoint (epoch + metric)
BEST_META_PATH = OUT_DIR / f"mode_{MODEL_NAME}_meta.json"

# Test predictions output (strict CSV: img_name, probability)
PRED_CSV_PATH = OUT_DIR / "predictions.csv"

print("TRAIN_DIR:", TRAIN_DIR.resolve())
print("TEST_DIR :", TEST_DIR.resolve())
print("MODEL_NAME:", MODEL_NAME)
print("CKPT_PATH:", CKPT_PATH.resolve())


# %% [markdown]
# ## 2) Reproducibility + device + performance flags

# %%
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic can be slower; enable if you need exact reproducibility
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" device selected ->", device)

if device.type == "cuda":
    # TF32 can speed up matmuls/convs on Ampere+ with minimal accuracy impact for many models
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(USE_TF32)
        torch.backends.cudnn.allow_tf32 = bool(USE_TF32)
    except Exception as e:
        print("TF32 flags not available:", e)

    # PyTorch 2+ matmul precision control
    try:
        torch.set_float32_matmul_precision("high" if USE_TF32 else "highest")
    except Exception:
        pass

# Choose AMP dtype
def get_amp_dtype() -> Optional[torch.dtype]:
    if not (USE_AMP and device.type == "cuda"):
        return None
    # Prefer bf16 if hardware supports it (more stable than fp16)
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16

AMP_DTYPE = get_amp_dtype()
print("AMP enabled:", bool(AMP_DTYPE), "| dtype:", AMP_DTYPE)

# Scaler is only needed for fp16; bf16 usually doesn't require scaling
USE_SCALER = bool(AMP_DTYPE == torch.float16 and device.type == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER) if device.type == "cuda" else None


# %% [markdown]
# ## 3) Transforms: pad-to-square (keep aspect ratio) → resize → tensor → normalize

# %%
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class PadToSquare:
    """Pads an image with zeros (black) to make it square, keeping aspect ratio."""
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        new_img = Image.new("RGB", (size, size), (self.fill, self.fill, self.fill))
        left = (size - w) // 2
        top = (size - h) // 2
        new_img.paste(img, (left, top))
        return new_img

train_tfms = transforms.Compose([
    PadToSquare(fill=0),
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tfms = transforms.Compose([
    PadToSquare(fill=0),
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# %% [markdown]
# ## 4) Models

# %%
class SimpleCNN(nn.Module):
    """A compact CNN for 224x224 RGB images. Outputs a single logit."""
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x  # [B, 1] logits


def _make_resnet_binary(variant: str) -> nn.Module:
    variant = variant.lower().strip()
    if variant == "resnet18":
        m = torchvision.models.resnet18(weights=None)
    elif variant == "resnet50":
        m = torchvision.models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported ResNet variant: {variant}")
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, 1)
    return m


def _make_convnext_binary(variant: str) -> nn.Module:
    variant = variant.lower().strip()
    name_map = {
        "convnext": "convnext_tiny",
        "convnext_tiny": "convnext_tiny",
        "convnext_small": "convnext_small",
        "convnext_base": "convnext_base",
        "convnext_large": "convnext_large",
    }
    fn_name = name_map.get(variant, variant)
    if not hasattr(torchvision.models, fn_name):
        raise ValueError(
            f"ConvNeXt variant '{variant}' not available in this torchvision ({torchvision.__version__}). "
            f"Available variants might include: convnext_tiny/small/base/large."
        )
    fn = getattr(torchvision.models, fn_name)
    m = fn(weights=None)
    if not hasattr(m, "classifier") or not isinstance(m.classifier, nn.Sequential):
        raise RuntimeError("Unexpected ConvNeXt model structure: missing 'classifier' Sequential")
    last = m.classifier[-1]
    if not isinstance(last, nn.Linear):
        raise RuntimeError("Unexpected ConvNeXt classifier: last layer is not nn.Linear")
    in_features = last.in_features
    m.classifier[-1] = nn.Linear(in_features, 1)
    return m


def create_model(model_name: str) -> nn.Module:
    name = model_name.lower().strip()
    if name in {"cnn", "simplecnn", "simple_cnn"}:
        return SimpleCNN()
    if name in {"resnet18", "resnet-18", "resnet50", "resnet-50"}:
        return _make_resnet_binary(name.replace("-", ""))
    if name.startswith("convnext"):
        return _make_convnext_binary(name)
    raise ValueError(f"Unknown model_name={model_name!r}.")


# %% [markdown]
# ## 5) Metrics

# %%
def confusion_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float32)
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn

def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0

def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    tp, tn, fp, fn = confusion_from_probs(y_true, y_prob, threshold=threshold)

    prec_fish = safe_div(tp, tp + fp)
    rec_fish  = safe_div(tp, tp + fn)
    f1_fish = safe_div(2.0 * prec_fish * rec_fish, (prec_fish + rec_fish))

    prec_nonfish = safe_div(tn, tn + fn)
    rec_nonfish  = safe_div(tn, tn + fp)

    acc = safe_div(tp + tn, tp + tn + fp + fn)

    auroc = None
    auprc = None
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
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
            auroc = float("nan")
            auprc = float("nan")

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


# %% [markdown]
# ## 6) Training + evaluation loops (AMP + channels_last)

# %%
def get_binary_label_from_imagefolder_label(lbl: torch.Tensor, fish_class_index: int) -> torch.Tensor:
    return (lbl == fish_class_index).long()

def _to_device_batch(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if device.type == "cuda" and USE_CHANNELS_LAST:
        images = images.to(memory_format=torch.channels_last)
    return images

def _maybe_channels_last_model(model: nn.Module, device: torch.device) -> nn.Module:
    if device.type == "cuda" and USE_CHANNELS_LAST:
        return model.to(memory_format=torch.channels_last)
    return model

def _autocast_ctx():
    if device.type != "cuda" or AMP_DTYPE is None:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True)

@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, fish_class_index: int, device: torch.device):
    model.eval()
    probs_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    loss_all: List[float] = []
    criterion = nn.BCEWithLogitsLoss()

    for images, labels in loader:
        images = _to_device_batch(images, device)
        y = get_binary_label_from_imagefolder_label(labels, fish_class_index).float().to(device, non_blocking=True)

        with _autocast_ctx():
            logits = model(images).squeeze(1)  # [B]
            loss = criterion(logits, y)
            prob = torch.sigmoid(logits)

        probs_all.append(prob.detach().float().cpu().numpy())
        y_all.append(y.detach().float().cpu().numpy())
        loss_all.append(float(loss.item()))

    y_true = np.concatenate(y_all) if y_all else np.array([])
    y_prob = np.concatenate(probs_all) if probs_all else np.array([])
    m = metrics_from_probs(y_true.astype(np.int64), y_prob.astype(np.float64), threshold=0.5)
    m["loss"] = float(np.mean(loss_all)) if loss_all else float("nan")
    return m, y_true, y_prob

def run_train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                        fish_class_index: int, device: torch.device, epoch: int):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    n = 0

    for step, (images, labels) in enumerate(loader, start=1):
        images = _to_device_batch(images, device)
        y = get_binary_label_from_imagefolder_label(labels, fish_class_index).float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with _autocast_ctx():
            logits = model(images).squeeze(1)
            loss = criterion(logits, y)

        if device.type == "cuda" and USE_SCALER and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        running_loss += float(loss.item()) * bs
        n += bs

        if step == 1 or step % 50 == 0:
            print(f"    [epoch {epoch:03d}] step {step:04d} | loss {loss.item():.4f}")

    return running_loss / max(n, 1)

def select_best_metric(metrics: dict, metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(f"Metric {metric_name!r} not found in: {list(metrics.keys())}")
    val = metrics[metric_name]
    if val is None:
        return float("-inf")
    try:
        if math.isnan(val):
            return float("-inf")
    except Exception:
        pass
    return float(val)


# %% [markdown]
# ## 7) Data: stratified split + DataLoaders (throughput tuned)

# %%
def stratified_split_indices(y: np.ndarray, train_frac: float = 0.8, seed: int = 42):
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_frac, random_state=seed)
        train_idx, val_idx = next(sss.split(np.zeros(len(y)), y))
        return train_idx.tolist(), val_idx.tolist()
    except Exception:
        rng = np.random.default_rng(seed)
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        rng.shuffle(idx0); rng.shuffle(idx1)
        n0_train = int(len(idx0) * train_frac)
        n1_train = int(len(idx1) * train_frac)
        train_idx = np.concatenate([idx0[:n0_train], idx1[:n1_train]])
        val_idx = np.concatenate([idx0[n0_train:], idx1[n1_train:]])
        rng.shuffle(train_idx); rng.shuffle(val_idx)
        return train_idx.tolist(), val_idx.tolist()

def make_loader(ds, batch_size: int, shuffle: bool) -> DataLoader:
    pin = (device.type == "cuda")
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(NUM_WORKERS),
        pin_memory=pin,
        drop_last=False,
    )
    if int(NUM_WORKERS) > 0:
        kwargs["prefetch_factor"] = int(PREFETCH_FACTOR)
        kwargs["persistent_workers"] = bool(PERSISTENT_WORKERS)
    return DataLoader(ds, **kwargs)

def autotune_batch_size(
    train_subset: Subset,
    fish_idx: int,
    model_name: str,
    optimizer_factory,
    candidates: List[int],
) -> int:
    """Pick the largest batch size that fits in VRAM by trying candidates (largest -> smallest)."""
    if device.type != "cuda":
        return BATCH_SIZE

    print("===> Autotuning batch size (VRAM-greedy):", candidates)
    criterion = nn.BCEWithLogitsLoss()

    for bs in candidates:
        try:
            m = create_model(model_name).to(device)
            m = _maybe_channels_last_model(m, device)
            opt = optimizer_factory(m)

            test_loader = DataLoader(
                train_subset,
                batch_size=bs,
                shuffle=True,
                num_workers=0,  # keep the test simple & robust
                pin_memory=True,
            )
            images, labels = next(iter(test_loader))
            images = _to_device_batch(images, device)
            y = get_binary_label_from_imagefolder_label(labels, fish_idx).float().to(device)

            opt.zero_grad(set_to_none=True)
            with _autocast_ctx():
                logits = m(images).squeeze(1)
                loss = criterion(logits, y)

            if USE_SCALER and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            del m, opt, images, labels, y, logits, loss, test_loader
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(f"===> Autotune picked batch_size={bs}")
            return bs

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda out of memory" in msg:
                print(f"    batch_size={bs} -> OOM, trying smaller...")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise

    print(f"===> Autotune: all candidates OOM; falling back to batch_size={BATCH_SIZE}")
    return BATCH_SIZE


# %% [markdown]
# ## 8) Main: train + save best + inference

# %%
def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"TRAIN_DIR not found: {TRAIN_DIR.resolve()}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"TEST_DIR not found: {TEST_DIR.resolve()}")

    # Full dataset (for class_to_idx + labels)
    full_ds = torchvision.datasets.ImageFolder(root=str(TRAIN_DIR), transform=val_tfms)
    class_to_idx = full_ds.class_to_idx
    print("Classes found:", class_to_idx)

    if "fish" not in class_to_idx:
        raise ValueError(
            f"Could not find a 'fish' folder under {TRAIN_DIR}. Found: {list(class_to_idx.keys())}"
        )
    fish_idx = class_to_idx["fish"]

    targets = np.array(full_ds.targets)
    binary_targets = (targets == fish_idx).astype(np.int64)

    train_idx, val_idx = stratified_split_indices(binary_targets, train_frac=0.8, seed=SEED)

    # Separate datasets for different transforms
    train_ds = torchvision.datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_tfms)
    val_ds   = torchvision.datasets.ImageFolder(root=str(TRAIN_DIR), transform=val_tfms)

    train_subset = Subset(train_ds, train_idx)
    val_subset   = Subset(val_ds, val_idx)

    train_y = binary_targets[train_idx]
    val_y   = binary_targets[val_idx]
    print(f"Train size: {len(train_subset)}  (fish: {int(train_y.sum())}, non-fish: {int((train_y==0).sum())})")
    print(f"Val size  : {len(val_subset)}    (fish: {int(val_y.sum())}, non-fish: {int((val_y==0).sum())})")

    # Optimizer factory (reused by batch autotune)
    def make_optimizer(m: nn.Module):
        return torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # VRAM-greedy batch size
    batch_size = BATCH_SIZE
    if AUTO_TUNE_BATCH_SIZE and device.type == "cuda":
        batch_size = autotune_batch_size(
            train_subset=train_subset,
            fish_idx=fish_idx,
            model_name=MODEL_NAME,
            optimizer_factory=make_optimizer,
            candidates=list(BATCH_SIZE_CANDIDATES),
        )

    # DataLoaders
    train_loader = make_loader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader   = make_loader(val_subset, batch_size=batch_size, shuffle=False)

    print("Using batch_size:", batch_size)
    print("NUM_WORKERS:", NUM_WORKERS, "| pin_memory:", device.type == "cuda")

    # Model
    model = create_model(MODEL_NAME).to(device)
    model = _maybe_channels_last_model(model, device)

    # Optional torch.compile (PyTorch 2+)
    if USE_TORCH_COMPILE and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=TORCH_COMPILE_MODE)
            print("torch.compile enabled | mode:", TORCH_COMPILE_MODE)
        except Exception as e:
            print("torch.compile failed; continuing uncompiled:", repr(e))

    optimizer = make_optimizer(model)

    print(model.__class__.__name__)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Warm-up one batch (helps compile/cudnn autotune)
    print("Trying to fetch one batch...")
    images, labels = next(iter(train_loader))
    print("Batch OK:", images.shape, labels.shape, labels[:10])

    # Train with best-checkpoint saving
    best_metric = float("-inf")
    best_epoch = -1

    print("===> Starting: training loop")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n====> Epoch {epoch}/{NUM_EPOCHS} (MODEL={MODEL_NAME})")

        t0 = time.time()
        train_loss = run_train_one_epoch(model, train_loader, optimizer, fish_idx, device, epoch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"===> Train done | loss={train_loss:.4f} | time={time.time()-t0:.1f}s")

        print("===> Starting: validation")
        val_metrics, _, _ = run_eval(model, val_loader, fish_idx, device)

        print(f"Val loss: {val_metrics['loss']:.4f}")
        print(f"Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"F1 score (fish=1, thr=0.5): {val_metrics['f1_fish']:.4f}")
        print(f"Precision (fish=1): {val_metrics['precision_fish']:.4f}")
        print(f"Recall    (fish=1): {val_metrics['recall_fish']:.4f}")
        print(f"Precision (non-fish=0): {val_metrics['precision_nonfish']:.4f}")
        print(f"Recall    (non-fish=0): {val_metrics['recall_nonfish']:.4f}")
        print(f"AUROC: {val_metrics['auroc']}")
        print(f"AUPRC: {val_metrics['auprc']}")
        print(f"Confusion (tp, tn, fp, fn): ({val_metrics['tp']}, {val_metrics['tn']}, {val_metrics['fp']}, {val_metrics['fn']})")

        metric_value = select_best_metric(val_metrics, BEST_METRIC_NAME)

        if metric_value > best_metric:
            best_metric = metric_value
            best_epoch = epoch

            torch.save(model.state_dict(), CKPT_PATH)

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

    print("\n===> Training finished")
    print("Best epoch:", best_epoch, "| best", BEST_METRIC_NAME, "=", best_metric)
    print("Best weights file:", CKPT_PATH.resolve())

    # Inference on test images -> predictions.csv
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {CKPT_PATH.resolve()}")

    print("===> Starting: loading best weights")
    model = create_model(MODEL_NAME).to(device)
    model = _maybe_channels_last_model(model, device)

    try:
        state_dict = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(CKPT_PATH, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded weights from:", CKPT_PATH.resolve())

    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    test_paths: List[str] = []
    for e in exts:
        test_paths.extend(glob(str(TEST_DIR / e)))
    test_paths = sorted(test_paths)
    print(f"===> Starting: predicting on {len(test_paths)} test images")

    def load_and_preprocess(img_path: str):
        img = Image.open(img_path).convert("RGB")
        x = val_tfms(img)
        return x

    rows = [("img_name", "probability")]

    with torch.no_grad():
        for i, p in enumerate(test_paths, start=1):
            x = load_and_preprocess(p).unsqueeze(0)
            x = _to_device_batch(x, device)

            with _autocast_ctx():
                logit = model(x).squeeze()
                prob = torch.sigmoid(logit).float().item()

            img_name = Path(p).name
            rows.append((img_name, f"{prob:.10f}"))

            if i == 1 or i % 200 == 0:
                print(f"    predicted {i}/{len(test_paths)}")

    PRED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PRED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    print("===> Done: wrote", PRED_CSV_PATH.resolve())


if __name__ == "__main__":
    main()
