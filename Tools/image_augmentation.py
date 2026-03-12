"""
Augment a small class folder (e.g., "fish") by generating new images
with: horizontal flip, ±15° rotation, brightness/contrast jitter,
and small translation/affine transforms.

Requires:
  pip install albumentations opencv-python

Usage:
  python augment_fish.py --in_dir data/fish --out_dir data/fish_aug --target 3000
"""

import os
import cv2
import math
import glob
import uuid
import random
import argparse
from typing import List

import albumentations as A


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(files))


def build_augmenter(seed: int = 0) -> A.Compose:
    # Albumentations uses Python/random + numpy RNG internally; setting seed helps reproducibility.
    random.seed(seed)

    # Small translation/affine: translate_percent ±5% and mild scaling/shear.
    # Rotation limited to ±15 degrees.
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.05, 0.05),  # ±5% translation
                rotate=(-10, 10),
                shear=(-5, 5), # mild shear
                interpolation=cv2.INTER_LINEAR,
                mode=cv2.BORDER_REFLECT_101,
                p=1,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,  # ~±15%
                contrast_limit=0.15,    # ~±15%
                p=1,
            ),
        ]
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_image(out_path: str, image_bgr) -> None:
    # Create parent dirs if needed
    ensure_dir(os.path.dirname(out_path))
    # Write with OpenCV
    ok = cv2.imwrite(out_path, image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")


def augment_to_target(
    in_dir: str,
    out_dir: str,
    target_count: int,
    seed: int = 0,
    keep_originals: bool = True,
) -> None:
    ensure_dir(out_dir)

    src_paths = list_images(in_dir)
    if not src_paths:
        raise ValueError(f"No images found in: {in_dir}")

    augmenter = build_augmenter(seed=seed)

    # Optionally copy originals into out_dir first
    existing = list_images(out_dir)
    if keep_originals and not existing:
        for p in src_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            base = os.path.splitext(os.path.basename(p))[0]
            ext = os.path.splitext(p)[1].lower()
            out_path = os.path.join(out_dir, f"{base}__orig{ext}")
            save_image(out_path, img)

    # Recompute current count in out_dir
    current_paths = list_images(out_dir)
    current_count = len(current_paths)

    if current_count >= target_count:
        print(f"Already have {current_count} images in {out_dir} (>= target {target_count}). Done.")
        return

    needed = target_count - current_count
    print(f"Found {len(src_paths)} source images in {in_dir}.")
    print(f"Currently {current_count} images in {out_dir}. Need to generate {needed} more...")

    # Cycle through sources (randomly) until we hit target
    rng = random.Random(seed)
    for i in range(needed):
        src = rng.choice(src_paths)
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            continue

        augmented = augmenter(image=img)["image"]

        base = os.path.splitext(os.path.basename(src))[0]
        ext = os.path.splitext(src)[1].lower()
        # Unique filename to avoid collisions
        out_name = f"{base}__aug__{uuid.uuid4().hex[:10]}{ext}"
        out_path = os.path.join(out_dir, out_name)
        save_image(out_path, augmented)

        if (i + 1) % 200 == 0 or (i + 1) == needed:
            print(f"Generated {i+1}/{needed}")

    final_count = len(list_images(out_dir))
    print(f"Done. Output folder now has {final_count} images: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="C:\\Users\\leino\\Documents\\JYU Opinnot\\TIES4700 - Deep Learning\\RODI-DATA\\Train_iso\\shadow", help="Input folder containing fish images")
    parser.add_argument("--out_dir", default="C:\\Users\\leino\\Documents\\JYU Opinnot\\TIES4700 - Deep Learning\\RODI-DATA\\Train_iso_aug\\shadow", help="Output folder to write augmented images")
    parser.add_argument("--target", type=int, default=15000, help="Target total number of images in out_dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--no_keep_originals",
        action="store_true",
        help="If set, do NOT copy originals into out_dir before augmenting",
    )
    args = parser.parse_args()

    augment_to_target(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        target_count=args.target,
        seed=args.seed,
        keep_originals=not args.no_keep_originals,
    )


if __name__ == "__main__":
    main()