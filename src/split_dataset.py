import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(BASE_DIR, "datasets_balanced")
TRAIN_DIR  = os.path.join(BASE_DIR, "datasets", "train")
VAL_DIR    = os.path.join(BASE_DIR, "datasets", "val")

CLASS_FOLDERS = ["3", "14", "17", "18", "33"]
SPLIT_RATIO = 0.8

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

print("=== SPLIT DATASET TRAIN & VALIDATION ===\n")

for cls in CLASS_FOLDERS:
    src_dir  = os.path.join(SOURCE_DIR, cls)
    train_cls_dir = os.path.join(TRAIN_DIR, cls)
    val_cls_dir   = os.path.join(VAL_DIR, cls)

    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(val_cls_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    random.shuffle(files)

    split_index = int(len(files) * SPLIT_RATIO)
    train_files = files[:split_index]
    val_files   = files[split_index:]

    for file in train_files:
        shutil.copy(
            os.path.join(src_dir, file),
            os.path.join(train_cls_dir, file)
        )

    for file in val_files:
        shutil.copy(
            os.path.join(src_dir, file),
            os.path.join(val_cls_dir, file)
        )

    print(f"Kelas {cls}")
    print(f"  Train : {len(train_files)}")
    print(f"  Val   : {len(val_files)}\n")

print("✅ SPLIT TRAIN–VALIDATION SELESAI")
