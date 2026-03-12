import os
from pathlib import Path
from glob import glob
import csv
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# ===== User config =====
DATA_ROOT = Path("C:/Users/leino/Documents/JYU Opinnot/TIES4700 - Deep Learning/RODI-DATA")
TEST_DIR  = DATA_ROOT / "Test"

MODEL_NAME = "convnext_base"
IMG_SIZE = 224

OUT_DIR = Path("Competion/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = OUT_DIR / f"mode_{MODEL_NAME}.pt"
PRED_CSV_PATH = OUT_DIR / "predictions.csv"

USE_TF32 = True
USE_AMP = True
USE_CHANNELS_LAST = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

print("TEST_DIR :", TEST_DIR.resolve())
print("MODEL_NAME:", MODEL_NAME)
print("CKPT_PATH:", CKPT_PATH.resolve())
print("PRED_CSV_PATH:", PRED_CSV_PATH.resolve())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device selected ->", device)

if device.type == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(USE_TF32)
        torch.backends.cudnn.allow_tf32 = bool(USE_TF32)
    except Exception as e:
        print("TF32 flags not available:", e)

    try:
        torch.set_float32_matmul_precision("high" if USE_TF32 else "highest")
    except Exception:
        pass


def get_amp_dtype() -> Optional[torch.dtype]:
    if not (USE_AMP and device.type == "cuda"):
        return None
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


AMP_DTYPE = get_amp_dtype()
print("AMP enabled:", bool(AMP_DTYPE), "| dtype:", AMP_DTYPE)


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


val_tfms = transforms.Compose([
    PadToSquare(fill=0),
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


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
    if name in {"resnet18", "resnet-18", "resnet50", "resnet-50"}:
        return _make_resnet_binary(name.replace("-", ""))
    if name.startswith("convnext"):
        return _make_convnext_binary(name)
    raise ValueError(f"Unknown model_name={model_name!r}.")


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


def load_and_preprocess(img_path: str):
    img = Image.open(img_path).convert("RGB")
    x = val_tfms(img)
    return x


def main():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"TEST_DIR not found: {TEST_DIR.resolve()}")
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

    exts = ("*.jpg", "*.jpeg", "*.png")
    test_paths: List[str] = []
    for e in exts:
        test_paths.extend(glob(str(TEST_DIR / e)))
    test_paths = sorted(test_paths)
    print(f"===> Starting: predicting on {len(test_paths)} test images")

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