# ================================
# CONFIGURASI GLOBAL PROJECT
# ================================
import os

IMG_SIZE = 30          # Ukuran gambar input (30x30)
EPOCHS = 30            # Jumlah epoch training
BATCH_SIZE = 32        # Batch size
NUM_CLASSES = 5        # Jumlah kelas


# Menggunakan path relatif dari root project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "datasets", "train")
VAL_PATH   = os.path.join(BASE_DIR, "datasets", "val")
TEST_PATH  = os.path.join(BASE_DIR, "datasets", "test")


# Daftar kelas yang digunakan
CLASS_FOLDERS = ["3", "14", "17", "18", "33"]
