import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from config import TRAIN_PATH, VAL_PATH, IMG_SIZE, CLASS_FOLDERS

def load_images_from_folder(base_path):
    X = []
    y = []

    for idx, cls in enumerate(CLASS_FOLDERS):
        folder = os.path.join(base_path, cls)

        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(folder, file)

                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0

                X.append(img)
                y.append(idx)

    X = np.array(X)
    y = to_categorical(y, num_classes=len(CLASS_FOLDERS))

    return X, y


def load_data():
    print("Memuat dataset TRAIN...")
    X_train, y_train = load_images_from_folder(TRAIN_PATH)

    print("Memuat dataset VALIDATION...")
    X_val, y_val = load_images_from_folder(VAL_PATH)

    print("Data Training :", X_train.shape)
    print("Data Validation:", X_val.shape)

    return X_train, X_val, y_train, y_val
