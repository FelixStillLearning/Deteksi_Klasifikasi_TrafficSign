# ================================
# CONFIGURASI GLOBAL PROJECT
# ================================
import os

IMG_SIZE = 30          # Ukuran gambar input (30x30)
EPOCHS = 30            # Jumlah epoch training
BATCH_SIZE = 32        # Batch size
NUM_CLASSES = 5        # Jumlah kelas

# Path dataset (dataset yang SUDAH balanced)
# Menggunakan path relatif dari root project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "datasets_balanced")

# Daftar kelas yang digunakan
CLASS_FOLDERS = ["3", "14", "17", "18", "33"]
