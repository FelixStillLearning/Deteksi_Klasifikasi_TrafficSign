import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from config import BASE_DIR, IMG_SIZE, CLASS_FOLDERS

class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Classification - JST")
        self.root.geometry("900x600")
        self.root.resizable(False, False)
        
        # Load model
        model_path = os.path.join(BASE_DIR, "models", "traffic_sign_model.h5")
        try:
            self.model = load_model(model_path)
            print(f"Model loaded: {model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal load model: {e}")
            self.root.destroy()
            return
        
        # Class names mapping
        self.class_map = {
            0: ("3",  "Speed Limit 60"),
            1: ("14", "Stop"),
            2: ("17", "No Entry"),
            3: ("18", "General Caution"),
            4: ("33", "Turn Right Ahead")
        }
        
        # Variables
        self.current_image = None
        self.current_video_path = None
        
        # Create UI
        self.create_widgets()
    
    def create_widgets(self):
        # Set warna tema
        BG_DARK = "#1a1a1a"
        BG_LIGHT = "#f5f5f5"
        COLOR_PRIMARY = "#2196F3"
        COLOR_SUCCESS = "#4CAF50"
        COLOR_DANGER = "#f44336"
        
        self.root.config(bg=BG_LIGHT)
        
        # ===== TOP FRAME: TITLE =====
        title_frame = tk.Frame(self.root, bg=BG_DARK, height=70)
        title_frame.pack(fill="x")
        
        title_label = tk.Label(
            title_frame, 
            text="🚦 Traffic Sign Classification System", 
            font=("Segoe UI", 20, "bold"),
            bg=BG_DARK, 
            fg="white"
        )
        title_label.pack(pady=20)
        
        # ===== MIDDLE FRAME: BUTTONS =====
        button_frame = tk.Frame(self.root, bg=BG_LIGHT, height=80)
        button_frame.pack(fill="x", padx=20, pady=15)
        
        btn_style = {
            "font": ("Segoe UI", 10, "bold"),
            "width": 16,
            "height": 2,
            "fg": "white",
            "relief": "flat",
            "cursor": "hand2",
            "bd": 0
        }
        
        btn_upload_img = tk.Button(
            button_frame, 
            text="📁 Upload Gambar", 
            command=self.upload_image,
            bg=COLOR_PRIMARY,
            **btn_style
        )
        btn_upload_img.pack(side="left", padx=10)
        
        btn_upload_video = tk.Button(
            button_frame, 
            text="🎥 Upload Video", 
            command=self.upload_video,
            bg=COLOR_PRIMARY,
            **btn_style
        )
        btn_upload_video.pack(side="left", padx=10)
        
        btn_info = tk.Button(
            button_frame, 
            text="ℹ️ Info Model", 
            command=self.show_model_info,
            bg=COLOR_SUCCESS,
            **btn_style
        )
        btn_info.pack(side="left", padx=10)
        
        # ===== CONTENT FRAME =====
        content_frame = tk.Frame(self.root, bg=BG_LIGHT)
        content_frame.pack(fill="both", expand=True, padx=20, pady=15)
        
        # LEFT: Image Preview
        left_frame = tk.Frame(content_frame, bg="white", relief="solid", borderwidth=1)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 15))
        
        preview_label = tk.Label(
            left_frame, 
            text="Preview Gambar / Video", 
            font=("Segoe UI", 12, "bold"), 
            bg="white",
            fg=BG_DARK
        )
        preview_label.pack(pady=10)
        
        self.canvas = tk.Canvas(
            left_frame, 
            width=380, 
            height=380, 
            bg="#e8e8e8", 
            relief="solid", 
            borderwidth=1,
            highlightthickness=0
        )
        self.canvas.pack(pady=10, padx=10)
        
        # RIGHT: Results
        right_frame = tk.Frame(content_frame, bg="white", relief="solid", borderwidth=1)
        right_frame.pack(side="right", fill="both", expand=True, padx=(15, 0))
        
        result_label = tk.Label(
            right_frame, 
            text="📊 Hasil Prediksi", 
            font=("Segoe UI", 12, "bold"), 
            bg="white",
            fg=BG_DARK
        )
        result_label.pack(pady=10)
        
        self.result_frame = tk.Frame(right_frame, bg="#f9f9f9", relief="solid", borderwidth=1)
        self.result_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.result_text = tk.Label(
            self.result_frame, 
            text="📝 Belum ada hasil\n\nUpload gambar atau video\nuntuk melihat hasil prediksi",
            font=("Segoe UI", 11),
            bg="#f9f9f9",
            fg="#666666",
            justify="center",
            wraplength=250,
            pady=20
        )
        self.result_text.pack(fill="both", expand=True, padx=15, pady=15)
        
        # ===== BOTTOM FRAME: STATUS BAR =====
        status_frame = tk.Frame(self.root, bg=BG_DARK, height=35)
        status_frame.pack(fill="x", side="bottom")
        
        self.status_label = tk.Label(
            status_frame, 
            text="✓ Status: Ready", 
            font=("Segoe UI", 9),
            bg=BG_DARK,
            fg="#4CAF50",
            anchor="w"
        )
        self.status_label.pack(fill="x", padx=15, pady=8)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        
        if file_path:
            self.current_image = file_path
            self.current_video_path = None
            self.display_image(file_path)
            self.status_label.config(text=f"✓ Status: Gambar loaded - {os.path.basename(file_path)}", fg="#4CAF50")
            
            # Auto predict setelah upload
            self.predict_image(file_path)
    
    def upload_video(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        
        if file_path:
            self.current_video_path = file_path
            self.current_image = None
            
            # Display first frame
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Save temp first frame
                temp_path = "temp_frame.jpg"
                cv2.imwrite(temp_path, frame)
                self.display_image(temp_path)
                os.remove(temp_path)
                
            self.status_label.config(text=f"✓ Status: Video loaded - {os.path.basename(file_path)}", fg="#4CAF50")
            
            # Auto predict video
            self.predict_video(file_path)
    
    def display_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((380, 380), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        self.canvas.delete("all")
        self.canvas.create_image(190, 190, image=img_tk)
        self.canvas.image = img_tk  # Keep reference
    
    def predict_image(self, image_path):
        self.status_label.config(text="Status: Processing...")
        self.root.update()
        
        try:
            # Preprocess
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Predict
            predictions = self.model.predict(img, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = predictions[class_idx] * 100

            class_id, class_full_name = self.class_map[class_idx]
            
            # Display results
            result_text = f"✅ Prediksi:\n\n"
            result_text += f"Class ID: {class_id}\n"
            result_text += f"Sign: {class_full_name}\n"
            result_text += f"Confidence: {confidence:.2f}%\n\n"
            result_text += "Top 3 Predictions:\n"

            top3_idx = np.argsort(predictions)[-3:][::-1]
            for idx in top3_idx:
                cid, cname = self.class_map[idx]
                result_text += f"  {cname} ({cid}): {predictions[idx]*100:.1f}%\n"
            
            self.result_text.config(text=result_text, fg="#2196F3")
            self.status_label.config(text=f"✓ Status: Prediksi selesai - {class_full_name}", fg="#4CAF50")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal prediksi: {e}")
            self.status_label.config(text="⚠ Status: Error saat prediksi", fg="#f44336")
    
    def predict_video(self, video_path):
        self.status_label.config(text="⏳ Status: Processing video...", fg="#FF9800")
        self.root.update()
        
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Sample every 30 frames
            sample_rate = 30
            predictions_list = []
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_rate == 0:
                    # Preprocess
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)
                    
                    # Predict
                    predictions = self.model.predict(img, verbose=0)[0]
                    class_idx = np.argmax(predictions)
                    predictions_list.append(class_idx)
                
                frame_idx += 1
            
            cap.release()
            
            # Most common prediction
            if predictions_list:
                from collections import Counter
                most_common = Counter(predictions_list).most_common(1)[0]
                class_idx = most_common[0]
                occurrences = most_common[1]
                class_name = self.class_names[class_idx]
                class_full_name = self.class_info.get(class_name, class_name)
                
                result_text = f"✅ Hasil Video:\n\n"
                result_text += f"Class ID: {class_name}\n"
                result_text += f"Sign: {class_full_name}\n"
                result_text += f"Deteksi: {occurrences}/{len(predictions_list)} frames\n"
                result_text += f"Total Frames: {frame_count}\n"
                result_text += f"FPS: {fps}\n"
                
                self.result_text.config(text=result_text, fg="green")
                self.status_label.config(text=f"Status: Video selesai - {class_full_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal proses video: {e}")
            self.status_label.config(text="⚠ Status: Error saat proses video", fg="#f44336")
    
    def show_model_info(self):
        info = f"Model Information\n"
        info += f"================\n\n"
        info += f"Architecture: CNN\n"
        info += f"Input Size: {IMG_SIZE}x{IMG_SIZE}x3\n"
        info += f"Classes: {len(CLASS_FOLDERS)}\n"
        info += f"Class Names: {', '.join(CLASS_FOLDERS)}\n\n"
        info += f"Training Accuracy: 99.93%\n"
        
        messagebox.showinfo("Model Info", info)


def main():
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
