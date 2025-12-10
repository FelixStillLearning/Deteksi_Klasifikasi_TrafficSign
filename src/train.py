import os
import csv
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import seaborn as sns

from data_loader import load_data
from model_builder import build_model
from config import EPOCHS, BATCH_SIZE, CLASS_FOLDERS, BASE_DIR

# Path untuk menyimpan model
MODELS_DIR = os.path.join(BASE_DIR, "models")


def main():
    # ================================
    # 1. LOAD DATA
    # ================================
    X_train, X_val, y_train, y_val = load_data()


    # ================================
    # 2. BUILD MODEL
    # ================================
    model = build_model()
    model.summary()

    # ================================
    # 3. CALLBACKS
    # ================================
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )

    callbacks = [early_stop, lr_scheduler]

    # ================================
    # 4. DATA AUGMENTATION
    # ================================
    train_datagen = ImageDataGenerator(
    rotation_range=5,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05
    )

    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE
    )

    # ================================
    # 5. TRAINING
    # ================================
    history = model.fit(
        train_generator,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # ================================
    # 6. EVALUASI MODEL
    # ================================
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print("Akurasi Validation :", val_acc)
    print("Loss Validation    :", val_loss)

    # ================================
    # 7. CONFUSION MATRIX
    # ================================
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_FOLDERS,
                yticklabels=CLASS_FOLDERS)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix_val.jpg")
    plt.savefig(cm_path, dpi=300)

    plt.show()

    print("Confusion matrix disimpan di:", cm_path)

    # ================================
    # 8. CLASSIFICATION REPORT
    # ================================
    report = classification_report(
        y_true,
        y_pred_classes,
        target_names=CLASS_FOLDERS
    )
    print("\n=== CLASSIFICATION REPORT ===\n")
    print(report)

    # ================================
    # 9. EXPORT HISTORY KE CSV
    # ================================
    os.makedirs(MODELS_DIR, exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(MODELS_DIR, "training_history.csv"), index=False)
    print("History training disimpan ke training_history.csv")

    # ================================
    # 10. SIMPAN MODEL
    # ================================
    model.save(os.path.join(MODELS_DIR, "traffic_sign_model.h5"))
    print("Model disimpan sebagai traffic_sign_model.h5")

    # ================================
    # 11. PLOT GRAFIK AKURASI & LOSS
    # ================================
    plt.figure(figsize=(12, 5))

    # Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Grafik Akurasi')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Grafik Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
