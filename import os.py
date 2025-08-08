import os
import time
import threading
import numpy as np
import cv2

try:
    from tflite_runtime.interpreter import Interpreter
    print("Using tflite_runtime interpreter")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("Using tensorflow.lite interpreter")

try:
    from playsound import playsound
except ImportError:
    raise RuntimeError("Please install playsound (pip install playsound==1.3.0)")

def play_audio(path):
    if os.path.exists(path):
        threading.Thread(target=lambda: playsound(path), daemon=True).start()
    else:
        print("Audio not found:", path)

def load_labels(path):
    return [l.strip() for l in open(path, 'r').read().splitlines() if l.strip()]

def load_tflite(model_path):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()

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

AUDIO_DIR = "assets/audio"
AUDIO_MAP = {
    "sunflower": os.path.join(AUDIO_DIR, "sunflower.mp3"),
    "rose": os.path.join(AUDIO_DIR, "rose.mp3"),
    "daisy": os.path.join(AUDIO_DIR, "daisy.mp3"),
    "bald": os.path.join(AUDIO_DIR, "bald.mp3"),
    "beard": os.path.join(AUDIO_DIR, "beard.mp3"),
    "glasses": os.path.join(AUDIO_DIR, "glasses.mp3"),
    "unknown": os.path.join(AUDIO_DIR, "default.mp3")
}

# State
MODE_FLOWER = "FLOWER"
MODE_FACE = "FACE"
mode = MODE_FLOWER
interp, inp_details, out_details = load_tflite(FLOWER_MODEL)
labels = load_labels(FLOWER_LABEL)
print(f"Loaded {mode} model with labels: {labels}")
CONF_THRESHOLD = 0.4
DETECT_INTERVAL = 1.5
last_detect = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press 'm' to toggle mode, 'd' to detect now, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("LOL Lens", frame)
    key = cv2.waitKey(1) & 0xFF

    now = time.time()
    do_detect = False

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
        last_detect = 0
    elif key == ord('d'):
        do_detect = True

    if now - last_detect > DETECT_INTERVAL:
        do_detect = True

    if do_detect:
        idx, conf = classify(interp, inp_details, out_details, frame, normalize='-1')
        label = labels[idx] if idx < len(labels) else "unknown"
        print("Detected:", label, "confidence:", conf)

        if conf >= CONF_THRESHOLD:
            play_audio(AUDIO_MAP.get(label, AUDIO_MAP["unknown"]))
            last_detect = now
        else:
            print("Low confidence; skipping")

cap.release()
cv2.destroyAllWindows()
print("Exiting...")
hi