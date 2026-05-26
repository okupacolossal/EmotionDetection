"""
Real-time emotion detection via webcam.

Uses:
  - OpenCV Haar Cascade  → face detection
  - EmotionCNN (PyTorch) → emotion classification

Usage (from project root):
    python detect.py
    python detect.py --weights weights/best_model.pth --camera 0

Controls:
    Q  — quit
"""

import argparse
import cv2
import torch
import numpy as np

from src.model   import CNN
from src.dataset import EMOTIONS, IMG_SIZE


# ── Config ────────────────────────────────────────────────────────────────────

# Colour per emotion  (BGR)
EMOTION_COLOURS = {
    'Angry':    (0,   0,   220),
    'Happy':    (0,   220, 0  ),
    'Fear':     (220, 0,   220),
    'Sad':      (220, 100, 0  ),
    'Surprise': (0,   200, 220),
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICK  = 2
TEXT_SCALE = 0.75
TEXT_THICK = 2


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_face(face_bgr: np.ndarray) -> torch.Tensor:
    """
    BGR face crop  →  (1, 1, 48, 48) float tensor in [0, 1].
    Mirrors the pipeline in scripts/prepare_dataset.py exactly.
    """
    gray    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    tensor  = torch.tensor(resized, dtype=torch.float32) / 255.0
    return tensor.unsqueeze(0).unsqueeze(0)   # → (1, 1, 48, 48)


# ── Overlay helpers ───────────────────────────────────────────────────────────

def draw_prediction(frame, x, y, w, h, emotion: str, confidence: float):
    colour = EMOTION_COLOURS.get(emotion, (255, 255, 255))

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, BOX_THICK)

    # Label background
    label      = f'{emotion}  {confidence:.0%}'
    (tw, th), _ = cv2.getTextSize(label, FONT, TEXT_SCALE, TEXT_THICK)
    label_y    = max(y - 10, th + 10)
    cv2.rectangle(frame, (x, label_y - th - 6), (x + tw + 6, label_y + 4), colour, -1)

    # Label text  (dark lettering on coloured background)
    cv2.putText(frame, label, (x + 3, label_y), FONT, TEXT_SCALE,
                (20, 20, 20), TEXT_THICK, cv2.LINE_AA)


def draw_fps(frame, fps: float):
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 28), FONT, 0.65,
                (200, 200, 200), 1, cv2.LINE_AA)


def draw_confidence_bars(frame, probs: torch.Tensor, x_offset=10, y_offset=60):
    """Mini bar chart of all class probabilities in the top-left corner."""
    bar_max_w = 120
    bar_h     = 14
    gap       = 5

    for i, (emo, prob) in enumerate(zip(EMOTIONS, probs.tolist())):
        top = y_offset + i * (bar_h + gap)
        # background track
        cv2.rectangle(frame, (x_offset, top), (x_offset + bar_max_w, top + bar_h),
                      (60, 60, 60), -1)
        # filled bar
        filled = int(prob * bar_max_w)
        colour = EMOTION_COLOURS.get(emo, (180, 180, 180))
        cv2.rectangle(frame, (x_offset, top), (x_offset + filled, top + bar_h),
                      colour, -1)
        # label
        cv2.putText(frame, f'{emo[:3]} {prob:.0%}',
                    (x_offset + bar_max_w + 6, top + bar_h - 2),
                    FONT, 0.42, (220, 220, 220), 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device  : {device}')
    print(f'Weights : {args.weights}')
    print(f'Camera  : {args.camera}')
    print('Press Q to quit.\n')

    # ── Load model ──────────────────────────────────────────────────────────
    model = CNN().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # ── Face detector ───────────────────────────────────────────────────────
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f'Failed to load Haar cascade from: {cascade_path}')

    # ── Camera ──────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {args.camera}')

    tick      = cv2.getTickCount
    fps       = 0.0
    fps_alpha = 0.1   # exponential smoothing factor

    while True:
        t_start = tick()
        ret, frame = cap.read()
        if not ret:
            print('[WARN] Empty frame — retrying...')
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor  = 1.3,
            minNeighbors = 5,
            minSize      = (48, 48),
        )

        # ── For each detected face ─────────────────────────────────────────
        last_probs = None
        for (x, y, w, h) in faces:
            face_crop = frame[y: y + h, x: x + w]
            tensor    = preprocess_face(face_crop).to(device)

            with torch.no_grad():
                logits = model(tensor)                       # (1, 5)
                probs  = torch.softmax(logits, dim=1)[0]    # (5,)
                pred   = torch.argmax(probs).item()

            emotion    = EMOTIONS[pred]
            confidence = probs[pred].item()
            last_probs = probs.cpu()

            draw_prediction(frame, x, y, w, h, emotion, confidence)

        # Confidence bars for the last (or only) detected face
        if last_probs is not None:
            draw_confidence_bars(frame, last_probs)

        # FPS counter
        elapsed = (tick() - t_start) / cv2.getTickFrequency()
        fps     = fps_alpha * (1.0 / max(elapsed, 1e-6)) + (1 - fps_alpha) * fps
        draw_fps(frame, fps)

        cv2.imshow('Emotion Detection  |  Q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Real-time emotion detection')
    p.add_argument('--weights', default='weights/best_model.pth',
                   help='Path to trained model weights')
    p.add_argument('--camera',  default=0, type=int,
                   help='Camera index (0 = default webcam)')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
