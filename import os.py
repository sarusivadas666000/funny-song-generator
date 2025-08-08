import os
import time
import threading
import numpy as np
import cv2

# Try to load TensorFlow Lite runtime first
try:
    from tflite_runtime.interpreter import Interpreter
    print("Using tflite_runtime interpreter")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("Using tensorflow.lite interpreter")

# Load audio player
try:
    from playsound import playsound
except ImportError:
    raise RuntimeError("Please install playsound (pip install playsound==1.3.0)")

# Play audio in the same thread so we can wait for it to finish
def play_audio_blocking(path):
    if os.path.exists(path):
        print(f"Playing audio: {path}")
        playsound(path)
    else:
        print(f"Audio not found for this label: {path}")

# Load label list from text file
def load_labels(path):
    return [l.strip() for l in open(path, 'r').read().splitlines() if l.strip()]

# Load TFLite model
def load_tflite(model_path):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()

# Run classification on frame
def classify(interp, inp_details, out_details, frame, normalize='-1'):
    h, w = inp_details[0]['shape'][1:3]
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp_dtype = inp_details[0]['dtype']
    if inp_dtype == np.uint8:
        data = np.expand_dims(img.astype(np.uint8), axis=0)
    else:
        f = img.astype(np.float32)
        if normalize == '-1':
            data = (np.expand_dims(f, 0) - 127.5) / 127.5
        else:
            data = np.expand_dims(f / 255.0, 0)
    interp.set_tensor(inp_details[0]['index'], data)
    interp.invoke()
    out = np.squeeze(interp.get_tensor(out_details[0]['index']))
    idx = int(np.argmax(out))
    return idx, float(out[idx])

# Paths
MODEL_DIR = "models"
FLOWER_MODEL = os.path.join(MODEL_DIR, "flower_model.tflite")
FLOWER_LABEL = os.path.join(MODEL_DIR, "flower_labels.txt")
FACE_MODEL   = os.path.join(MODEL_DIR, "face_model.tflite")
FACE_LABEL   = os.path.join(MODEL_DIR, "face_labels.txt")

# Audio mapping
AUDIO_MAP = {
    "sunflower": "assets/audio/sunflower.wav",
    "rose": "assets/audio/rose.mp3",
    "daisy": "assets/audio/daisy.wav",
    "dandelion": "assets/audio/dandelion.mp3",
    "beard": "assets/audio/beard.mp3",
    "glasses": "assets/audio/glasses.mp3"
}

# Modes
MODE_FLOWER = "FLOWER"
MODE_FACE = "FACE"
mode = MODE_FLOWER
interp, inp_details, out_details = load_tflite(FLOWER_MODEL)
labels = load_labels(FLOWER_LABEL)
print(f"Loaded {mode} model with labels: {labels}")

# Config
CONF_THRESHOLD = 0.95      # confidence threshold
REQUIRED_COUNT = 30      # consecutive detections needed

# Detection tracking
last_label = None
confidence_count = 0
flower_detected = False

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press 'm' to toggle mode, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"Mode: {mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("LOL Lens", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('m'):
        mode = MODE_FACE if mode == MODE_FLOWER else MODE_FLOWER
        if mode == MODE_FLOWER:
            interp, inp_details, out_details = load_tflite(FLOWER_MODEL)
            labels = load_labels(FLOWER_LABEL)
        else:
            interp, inp_details, out_details = load_tflite(FACE_MODEL)
            labels = load_labels(FACE_LABEL)
        print("Switched to", mode, "model:", labels)
        last_label = None
        confidence_count = 0

    if not flower_detected and mode == MODE_FLOWER:
        idx, conf = classify(interp, inp_details, out_details, frame, normalize='-1')
        raw_label = labels[idx].strip() if idx < len(labels) else "unknown"
        # Remove number prefix if present
        label_parts = raw_label.split()
        if label_parts and label_parts[0].isdigit():
            label = " ".join(label_parts[1:]).strip().lower()
        else:
            label = raw_label.lower()

        print(f"Detected: {label} confidence: {conf:.3f}")

        if conf >= CONF_THRESHOLD:
            if label == last_label:
                confidence_count += 1
            else:
                last_label = label
                confidence_count = 1
            print(f"Consecutive high-confidence count: {confidence_count}")
        else:
            confidence_count = 0

        if confidence_count >= REQUIRED_COUNT:
            print(f"Flower '{label}' detected with high confidence {REQUIRED_COUNT} times — stopping detection.")
            if label in AUDIO_MAP:
                play_audio_blocking(AUDIO_MAP[label])  # Play audio and wait for it
            else:
                print(f"No matching audio for label: {label}")
            flower_detected = True
            break  # Exit after audio

cap.release()
cv2.destroyAllWindows()
print("Exiting...")
