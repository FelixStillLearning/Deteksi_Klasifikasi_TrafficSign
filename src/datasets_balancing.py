import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ================================
# KONFIGURASI
# ================================

DATASET_PATH = "../datasets"          # Dataset asli
OUTPUT_PATH = "../datasets_balanced" # Dataset hasil balancing
TARGET_PER_CLASS = 1410
IMAGE_SIZE = 30

CLASS_FOLDERS = ["3", "14", "17", "18", "33"]

# ================================
# DATA AUGMENTATION SETTING
# ================================

datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False,
    fill_mode='nearest'
)

# ================================
# PROSES BALANCING
# ================================

os.makedirs(OUTPUT_PATH, exist_ok=True)

for folder in CLASS_FOLDERS:
    input_folder = os.path.join(DATASET_PATH, folder)
    output_folder = os.path.join(OUTPUT_PATH, folder)
    os.makedirs(output_folder, exist_ok=True)

    images = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    print(f"\nKelas {folder} | Data awal: {len(images)}")

    count = 0

    # ---------- SALIN DATA ASLI ----------
    for img_name in images:
        if count >= TARGET_PER_CLASS:
            break

        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        save_path = os.path.join(output_folder, f"orig_{count}.jpg")
        cv2.imwrite(save_path, img)
        count += 1

    # ---------- TAMBAH DATA DENGAN AUGMENTATION ----------
    if count < TARGET_PER_CLASS:
        for img_name in images:
            if count >= TARGET_PER_CLASS:
                break

            img_path = os.path.join(input_folder, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = np.expand_dims(img, axis=0)

            aug_iter = datagen.flow(img, batch_size=1)

            for _ in range(3):
                if count >= TARGET_PER_CLASS:
                    break

                augmented_img = next(aug_iter)[0]
                augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)

                save_path = os.path.join(output_folder, f"aug_{count}.jpg")
                cv2.imwrite(save_path, augmented_img)
                count += 1

    print(f"Kelas {folder} | Data akhir: {count}")

print("\n✅ DATASET BERHASIL DI-SAMARATAKAN KE 1410 GAMBAR PER KELAS")
print("📁 Lokasi hasil: datasets_balanced/")
