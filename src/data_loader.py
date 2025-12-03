import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from config import IMG_SIZE, DATA_PATH, CLASS_FOLDERS, NUM_CLASSES


def load_data():
    X = []
    y = []

    print("Memuat dataset...")

    for idx, folder in enumerate(CLASS_FOLDERS):
        folder_path = os.path.join(DATA_PATH, folder)

        if not os.path.exists(folder_path):
            print(f"Folder tidak ditemukan: {folder_path}")
            continue

        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, img_name)

                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0   # Normalisasi [0,1]

                    X.append(img)
                    y.append(idx)

                except:
                    print("Gagal membaca:", img_path)

    X = np.array(X)
    y = np.array(y)

    # One-hot encoding label
    y = to_categorical(y, NUM_CLASSES)

    # Split data dengan stratify
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Data Training :", X_train.shape)
    print("Data Validation:", X_val.shape)

    return X_train, X_val, y_train, y_val
