import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

from config import IMG_SIZE, CLASS_FOLDERS, BASE_DIR

# ================================
# PATH TEST SET
# ================================
TEST_DIR = os.path.join(BASE_DIR, "datasets", "test")
TEST_CSV = os.path.join(TEST_DIR, "Test.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_sign_model.h5")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def main():
    print("=== TEST MODEL DENGAN DATA KAGGLE (VERSI FOLDER KAMU) ===\n")

    # ================================
    # 1. LOAD CSV TEST
    # ================================
    if not os.path.exists(TEST_CSV):
        print("ERROR: Test.csv tidak ditemukan di folder test!")
        return

    df = pd.read_csv(TEST_CSV)
    print("Total data di Test.csv:", len(df))

    # ================================
    # 2. FILTER HANYA 5 KELAS
    # ================================
    target_classes = [3, 14, 17, 18, 33]
    df = df[df["ClassId"].isin(target_classes)]
    print("Jumlah data test setelah filter 5 kelas:", len(df))

    # ================================
    # 3. LOAD GAMBAR SESUAI CSV
    # ================================
    X_test = []
    y_test = []

    for _, row in df.iterrows():
        filename = os.path.basename(row["Path"])
        img_path = os.path.join(TEST_DIR, filename)

        class_id = row["ClassId"]

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X_test.append(img)
        y_test.append(target_classes.index(class_id))  # 0–4

    # ================================
    # 4. KONVERSI KE NUMPY & ONE-HOT
    # ================================
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    if X_test.shape[0] == 0:
        print("ERROR: Tidak ada satupun data test yang berhasil dimuat!")
        return

    y_test = to_categorical(y_test, num_classes=5)

    print("Jumlah data test yang berhasil dimuat:", X_test.shape[0])
    print("Shape X_test:", X_test.shape)
    print("Shape y_test:", y_test.shape)

    # ================================
    # 5. LOAD MODEL
    # ================================
    print("\nMemuat model...")
    model = load_model(MODEL_PATH)

    # ================================
    # 6. EVALUASI MODEL
    # ================================
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

    print("\n=== HASIL EVALUASI DATA TEST ===")
    print("Test Accuracy :", test_acc)
    print("Test Loss     :", test_loss)

    # ================================
    # 7. PREDIKSI
    # ================================
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # ================================
    # 8. CONFUSION MATRIX
    # ================================
    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_FOLDERS,
                yticklabels=CLASS_FOLDERS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - DATA TEST")
    plt.tight_layout()

    os.makedirs(MODELS_DIR, exist_ok=True)
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix_test.jpg")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print("Confusion matrix test disimpan di:", cm_path)

    # ================================
    # 9. CLASSIFICATION REPORT
    # ================================
    report = classification_report(
        y_true,
        y_pred_classes,
        target_names=CLASS_FOLDERS
    )

    print("\n=== CLASSIFICATION REPORT (TEST) ===\n")
    print(report)

    report_path = os.path.join(MODELS_DIR, "classification_report_test.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print("Classification report test disimpan di:", report_path)


if __name__ == "__main__":
    main()
