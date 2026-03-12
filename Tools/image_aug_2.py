"""
Augment a class folder (e.g., "fish") by generating new images until a target total.
Output naming:
  <folder>_<input_index>_org.jpg
  <folder>_<input_index>_arg_<copy_index>.jpg

Requires:
  pip install albumentations opencv-python

Usage:
  python augment_class.py --in_dir dataset/train/fish --out_dir dataset/train/fish_aug --target 3000
"""

import os
import cv2
import glob
import random
import argparse
from typing import List, Dict

import albumentations as A

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(files))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_augmenter(seed: int = 0) -> A.Compose:
    random.seed(seed)
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


def save_jpg(out_path: str, image_bgr, quality: int = 95) -> None:
    ensure_dir(os.path.dirname(out_path))
    ok = cv2.imwrite(out_path, image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
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

    class_name = os.path.basename(os.path.normpath(in_dir))  # folder name like "fish" or "leave"
    src_paths = list_images(in_dir)
    if not src_paths:
        raise ValueError(f"No images found in: {in_dir}")

    # Map each source image to a stable 1-based index (based on sorted file list)
    src_index: Dict[str, int] = {p: i + 1 for i, p in enumerate(src_paths)}

    augmenter = build_augmenter(seed=seed)
    rng = random.Random(seed)

    # Track how many augmented copies we've already created per input index
    # This ensures naming like fish_7_arg_1, fish_7_arg_2, ...
    aug_counter: Dict[int, int] = {i + 1: 0 for i in range(len(src_paths))}

    # Optionally write originals using your naming convention
    if keep_originals:
        for p in src_paths:
            idx = src_index[p]
            out_path = os.path.join(out_dir, f"{class_name}_{idx}_org.jpg")
            if os.path.exists(out_path):
                continue  # don't overwrite if rerun
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            save_jpg(out_path, img)

    # Count current jpg files in out_dir (including originals + augmented)
    current_count = len(glob.glob(os.path.join(out_dir, "*.jpg")))
    if current_count >= target_count:
        print(f"[{class_name}] Already have {current_count} images in {out_dir} (>= target {target_count}). Done.")
        return

    needed = target_count - current_count
    print(f"[{class_name}] Found {len(src_paths)} source images.")
    print(f"[{class_name}] Currently {current_count} images in out_dir. Need to generate {needed} augmented images...")

    # Evenly distribute augmentations across inputs
    N = len(src_paths)
    base = needed // N
    rem = needed % N

    for src_i, src in enumerate(src_paths, start=1):
        idx = src_index[src]
        k = base + (1 if src_i <= rem else 0)  # first 'rem' images get one extra

        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            continue

        for _ in range(k):
            augmented = augmenter(image=img)["image"]
            aug_counter[idx] += 1
            copy_idx = aug_counter[idx]
            out_path = os.path.join(out_dir, f"{class_name}_{idx}_arg_{copy_idx}.jpg")
            save_jpg(out_path, augmented)

    final_count = len(glob.glob(os.path.join(out_dir, "*.jpg")))
    print(f"[{class_name}] Done. Output folder now has {final_count} images: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="C:\\Users\\leino\\Documents\\JYU Opinnot\\TIES4700 - Deep Learning\\RODI-DATA\\Train\\shadow", help="Input folder containing insect images")
    parser.add_argument("--out_dir", default="C:\\Users\\leino\\Documents\\JYU Opinnot\\TIES4700 - Deep Learning\\RODI-DATA\\Train_aug\\shadow", help="Output folder to write augmented images")
    parser.add_argument("--target", type=int, default=15000, help="Target total number of images in out_dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_keep_originals", action="store_true", help="Do NOT export originals to out_dir")
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