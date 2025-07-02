import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

# GUI root
root = tk.Tk()
root.title("Chest X-ray Classifier")
root.geometry("700x500")
root.configure(bg="#f0f4f8")

style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 12), padding=10)
style.configure("TLabel", font=("Segoe UI", 14), background="#f0f4f8")
style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"), background="#f0f4f8")

# Load model on demand
model_cache = {}

# Model preprocessing map
model_preprocessors = {
    'vgg16': vgg16_preprocess,
    'resnet50': resnet50_preprocess
}

# Predict function
def predict(model_path, model_type):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    img = Image.open(file_path).resize((224, 224))
    img_tk = ImageTk.PhotoImage(img)

    pred_win = tk.Toplevel(root)
    pred_win.title("Prediction")
    pred_win.configure(bg="#ffffff")

    ttk.Label(pred_win, text="Selected Chest X-ray:", style="Header.TLabel").pack(pady=10)
    tk.Label(pred_win, image=img_tk, bg="white").pack()
    pred_win.image = img_tk

    img_array = np.array(img.convert("RGB"))
    img_array = np.expand_dims(img_array, axis=0)

    preproc = model_preprocessors[model_type]
    img_array = preproc(img_array)

    if model_path not in model_cache:
        try:
            model_cache[model_path] = load_model(model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return

    model = model_cache[model_path]
    prediction = model.predict(img_array)[0]
    label = "Pneumonia" if prediction[0] > 0.5 else "Normal"
    confidence = float(prediction[0]) if label == "Pneumonia" else 1 - float(prediction[0])

    ttk.Label(pred_win, text=f"Prediction: {label}", style="Header.TLabel").pack(pady=(20, 5))
    ttk.Label(pred_win, text=f"Confidence: {confidence * 100:.2f}%", font=("Segoe UI", 16)).pack(pady=5)

# Model selection window
def open_model_options(model_type):
    opt_win = tk.Toplevel(root)
    opt_win.title(f"{model_type.upper()} Models")
    opt_win.geometry("400x300")
    opt_win.configure(bg="#eef2f7")

    ttk.Label(opt_win, text=f"{model_type.upper()} Model Variants", style="Header.TLabel").pack(pady=20)

    model_variants = [
        ("Before BG - Before Aug", f"{model_type}_beforeBG_beforeAug.h5"),
        ("Before BG - After Aug", f"{model_type}_beforeBG_afterAug.h5"),
        ("After BG - Before Aug", f"{model_type}_afterBG_beforeAug.h5"),
        ("After BG - After Aug", f"{model_type}_afterBG_afterAug.h5"),
    ]

    for desc, file_name in model_variants:
        model_path = os.path.join("models", file_name)
        ttk.Button(opt_win, text=desc, width=40,
                   command=lambda p=model_path, t=model_type: predict(p, t)).pack(pady=8)

# Main UI
ttk.Label(root, text="Chest X-ray Diagnosis System", style="Header.TLabel").pack(pady=30)

ttk.Button(root, text="VGG16 Models", width=30, command=lambda: open_model_options('vgg16')).pack(pady=15)
ttk.Button(root, text="ResNet50 Models", width=30, command=lambda: open_model_options('resnet50')).pack(pady=15)

# Footer (optional)
ttk.Label(root, text="Developed for Deep Learning Project", font=("Segoe UI", 10)).pack(side="bottom", pady=10)

root.mainloop()
