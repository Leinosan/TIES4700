from PIL import Image
import os

root = "RODI-DATA_split/train"

for folder, _, files in os.walk(root):
    for f in files:
        path = os.path.join(folder, f)
        try:
            img = Image.open(path)
            img.verify()
        except Exception as e:
            print("Corrupted:", path, e)
