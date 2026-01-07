import os

# ================================
# PATH KE DATASET RAW
# ================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "datasets_balanced")

CLASS_FOLDERS = ["3", "14", "17", "18", "33"]
total_semua = 0

print("JUMLAH DATASET RAW PER KELAS:\n")

for folder in CLASS_FOLDERS:
    folder_path = os.path.join(DATASET_PATH, folder)

    if os.path.exists(folder_path):
        jumlah_file = len([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        print(f"Kelas {folder} : {jumlah_file} gambar")
        total_semua += jumlah_file
    else:
        print(f"Kelas {folder} : Folder tidak ditemukan!")

print("\nTOTAL SELURUH DATASET RAW :", total_semua, "gambar")
