# Splits RODI-DATA/Train into DATA_DIR/train and DATA_DIR/val preserving class folders.
import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

SRC = Path("RODI-DATA/Train")      # source with class subfolders
# set DATA_DIR in config to this folder ("RODI-DATA_split")
DST = Path("RODI-DATA_split")
TEST_SIZE = 0.2
SEED = 42

random.seed(SEED)

for cls_dir in SRC.iterdir():
    if not cls_dir.is_dir():
        continue
    imgs = list(cls_dir.glob("*.*"))
    if not imgs:
        continue
    train_imgs, val_imgs = train_test_split(
        [str(p) for p in imgs], test_size=TEST_SIZE, random_state=SEED)
    for sub, paths in [("train", train_imgs), ("val", val_imgs)]:
        out_dir = DST / sub / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy(p, out_dir / Path(p).name)
print("Done. Set DATA_DIR to", DST)
