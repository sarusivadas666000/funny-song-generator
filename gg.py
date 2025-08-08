import cv2
import numpy as np
import tensorflow as tf
import pygame
import os
import threading
import tkinter as tk
from PIL import Image, ImageTk
import re  # For regex to clean labels

# Paths
MODEL_PATH = "models/flower_model.tflite"
LABELS_PATH = "models/flower_labels.txt"
AUDIO_DIR = "assets/audio"

# Load and clean labels
with open(LABELS_PATH, "r") as f:
    raw_labels = [line.strip() for line in f.readlines()]

# Remove leading numbers/spaces and lowercase
labels = [re.sub(r'^\d+\s*', '', lbl).lower() for lbl in raw_labels]

# Audio mapping
AUDIO_MAP = {label: os.path.join(AUDIO_DIR, f"{label}.mp3") for label in labels}
AUDIO_MAP["default"] = os.path.join(AUDIO_DIR, "default.mp3")

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Init audio
pygame.mixer.init()

def play_audio(label):
    clean_label = re.sub(r'^\d+\s*', '', label).lower()
    path = AUDIO_MAP.get(clean_label, AUDIO_MAP["default"])
    if os.path.exists(path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    else:
        print(f"Audio not found for label: {clean_label}")

# Start webcam
cap = cv2.VideoCapture(0)
last_label = None

# Tkinter GUI
root = tk.Tk()
root.title("Flower Music App")

video_label = tk.Label(root)
video_label.pack()

detected_label = tk.Label(root, text="", font=("Arial", 18), fg="green")
detected_label.pack()

def detect_flower():
    global last_label
    ret, frame = cap.read()
    if not ret:
        root.after(10, detect_flower)
        return

    # Process for model
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    index = np.argmax(predictions)
    raw_label = raw_labels[index]  # Original label from file
    clean_label = re.sub(r'^\d+\s*', '', raw_label).lower()
    confidence = predictions[index]

    # Show video in GUI
    cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(cv2_image)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Update label text
    detected_label.config(text=f"{clean_label} ({confidence:.2f})")

    # Play music if detected & changed
    if clean_label != last_label and confidence >= 1.0:
        threading.Thread(target=play_audio, args=(clean_label,)).start()
        last_label = clean_label

    root.after(10, detect_flower)

root.after(0, detect_flower)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
