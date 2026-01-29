import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import ast
import os
from task1 import get_grocery_category


model = tf.keras.models.load_model('grocery_model.h5')
if os.path.exists('class_indices.txt'):
    with open('class_indices.txt', 'r') as f:
        idx_to_class = ast.literal_eval(f.read())
    print("Model Loaded ")



def predict_image_class(filepath):
    # Preprocessing with fixes
    img = Image.open(filepath).convert('RGB')
    img = img.resize((180, 180))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = np.max(preds[0]) * 100
    
    return idx_to_class.get(int(class_idx), "Unknown"), confidence


class Window:
    def __init__(self, root):
        self.root = root
        self.root.title("Task 3")
        self.root.geometry("500x600")
        tk.Label(root, text="Task 3", font=("Arial", 16)).pack(pady=10)

        # Image Area
        self.canvas = tk.Canvas(root, width=300, height=300, bg="lightgray")
        self.canvas.pack(pady=10)
        self.canvas.create_text(150, 150, text="upload a picture ", fill="gray")
        self.btn_upload = tk.Button(root, text="Upload Photo", command=self.upload_image)
        self.btn_upload.pack(pady=5)

        # Results
        self.lbl_detected = tk.Label(root, text="Detected: ...", font=("Arial", 12))
        self.lbl_detected.pack()
        
        self.lbl_category = tk.Label(root, text="Category: ...", font=("Arial", 12))
        self.lbl_category.pack()
        tk.Label(root, text="Shopping List:").pack(pady=(20, 5))
        self.list_box = tk.Listbox(root, height=8, width=40)
        self.list_box.pack()
        self.btn_add = tk.Button(root, text="Add to List", command=self.add_to_list, state="disabled")
        self.btn_add.pack(pady=10)

        self.current_item = None
        self.current_cat = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        

        load = Image.open(file_path).resize((300, 300))
        self.photo = ImageTk.PhotoImage(load)
        self.canvas.create_image(150, 150, image=self.photo)

        self.root.update()
        
        # Predict then cateegories 
        detected_item, confidence = predict_image_class(file_path)
        category = get_grocery_category(detected_item)
        
        # Update
        self.current_item = detected_item
        self.current_cat = category
        self.lbl_detected.config(text=f"Detected: {detected_item.upper()} ({confidence:.1f}%)")
        self.lbl_category.config(text=f"Category: {category}")
        self.btn_add.config(state="normal")
        self.add_to_list()

    def add_to_list(self):
        if self.current_item:
            entry = f"{self.current_item.upper():<15} [{self.current_cat}]"
            self.list_box.insert(tk.END, entry)
            self.list_box.yview(tk.END)
            self.btn_add.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = Window(root)
    root.mainloop()
