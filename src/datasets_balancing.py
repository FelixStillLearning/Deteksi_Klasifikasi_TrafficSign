import os
import shutil
import random

# ================================
# PATH
# ================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(BASE_DIR, "datasets", "raw")
TARGET_DIR = os.path.join(BASE_DIR, "datasets_balanced")

CLASS_FOLDERS = ["3", "14", "17", "18", "33"]
TARGET_PER_CLASS = 689   # ambil dari kelas terkecil

# ================================
# BUAT FOLDER TARGET
# ================================
os.makedirs(TARGET_DIR, exist_ok=True)

print("=== MULAI UNDERSAMPLING DATASET ===\n")

for cls in CLASS_FOLDERS:
    src_class_dir = os.path.join(SOURCE_DIR, cls)
    dst_class_dir = os.path.join(TARGET_DIR, cls)

    os.makedirs(dst_class_dir, exist_ok=True)

    files = [f for f in os.listdir(src_class_dir)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    print(f"Kelas {cls} - Data awal: {len(files)}")

    if len(files) < TARGET_PER_CLASS:
        print(f" WARNING: Data kelas {cls} kurang dari {TARGET_PER_CLASS}\n")
        selected_files = files
    else:
        selected_files = random.sample(files, TARGET_PER_CLASS)

    for file in selected_files:
        src = os.path.join(src_class_dir, file)
        dst = os.path.join(dst_class_dir, file)
        shutil.copy(src, dst)

    print(f"Kelas {cls} - Setelah balancing: {len(selected_files)}\n")

print(" UNDERSAMPLING SELESAI")
