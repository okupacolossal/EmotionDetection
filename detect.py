"""
Emotion Detection — Real-Time Webcam Inference
───────────────────────────────────────────────
First run:  guided capture session → fine-tune on your face → save personal model
Later runs: loads your personal model directly, skips setup.

Run:  python detect.py
Reset personal model:  python detect.py --reset
"""

import sys
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import os
import time
from pathlib import Path
from collections import deque
from train_pytorch import CNN

# ── Config ────────────────────────────────────────────────────────────────────

EMOTIONS        = ['Angry', 'Happy', 'Fear', 'Sad', 'Surprise']
IMG_SIZE        = 48
SMOOTH_FRAMES   = 15

PERSONAL_DIR    = Path('personal_data')
PERSONAL_MODEL  = Path('best_model_personal.pth')
BASE_MODEL      = Path('best_model.pth')

CAPTURE_COUNTDOWN = 3    # seconds "get ready" before each emotion
CAPTURE_SECONDS   = 5    # seconds of auto-capture per emotion
CAPTURE_FPS       = 10   # how many frames/sec to save during capture

FINETUNE_EPOCHS   = 40
FINETUNE_LR       = 5e-5  # small — preserve base knowledge, adapt to your face

EMOTION_COLORS = {
    'Angry':    (0,   0,  220),
    'Happy':    (0,  200,   0),
    'Fear':     (180,  0,  180),
    'Sad':      (220,  0,   0),
    'Surprise': (0,  180,  255),
}

# ── Face detector (shared everywhere) ────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_face(gray_eq):
    """Return (x,y,w,h) of largest face, or None."""
    faces = face_cascade.detectMultiScale(
        gray_eq, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])   # largest by area

def crop_face(gray, face_rect):
    """Crop, resize and equalise a face crop — matches training preprocessing."""
    x, y, w, h = face_rect
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = cv2.equalizeHist(face)
    return face

# ── Guided capture ─────────────────────────────────────────────────────────────

def run_capture_session(cap):
    """
    For each emotion:
      - 3 s countdown on screen
      - 5 s of auto-capture (one frame every ~100 ms)
      - saves 48×48 grayscale face crops to personal_data/<Emotion>/
    """
    PERSONAL_DIR.mkdir(exist_ok=True)
    window = 'Personal Training Setup'

    for emotion in EMOTIONS:
        folder = PERSONAL_DIR / emotion
        folder.mkdir(exist_ok=True)
        color  = EMOTION_COLORS[emotion]

        # ── countdown ──────────────────────────────────────────────────────────
        deadline = time.time() + CAPTURE_COUNTDOWN
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            remaining = int(deadline - time.time()) + 1

            overlay = frame.copy()
            h, w = overlay.shape[:2]

            # dim background
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, overlay)

            cv2.putText(overlay, 'Get ready to show:',
                        (40, h//2 - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
            cv2.putText(overlay, emotion.upper(),
                        (40, h//2),      cv2.FONT_HERSHEY_SIMPLEX, 3.2, color, 5)
            cv2.putText(overlay, f'Starting in  {remaining}',
                        (40, h//2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)

            cv2.imshow(window, overlay)
            cv2.waitKey(1)

        # ── capture ────────────────────────────────────────────────────────────
        target    = CAPTURE_SECONDS * CAPTURE_FPS
        captured  = 0
        skipped   = 0
        interval  = 1.0 / CAPTURE_FPS
        next_save = time.time()

        while captured < target:
            ret, frame = cap.read()
            if not ret:
                continue

            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)
            face    = detect_face(gray_eq)

            now = time.time()
            if face is not None and now >= next_save:
                crop = crop_face(gray, face)
                cv2.imwrite(str(folder / f'{captured:04d}.png'), crop)
                captured  += 1
                next_save  = now + interval
            elif face is None:
                skipped += 1

            # draw live preview
            overlay = frame.copy()
            fh, fw  = overlay.shape[:2]

            if face is not None:
                x, y, w, h = face
                cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)

            cv2.putText(overlay, f'CAPTURING: {emotion.upper()}',
                        (10, 35),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(overlay, f'Saved: {captured} / {target}',
                        (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            if face is None:
                cv2.putText(overlay, 'No face detected — move closer',
                            (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)

            # progress bar
            bar_filled = int(fw * captured / target)
            cv2.rectangle(overlay, (0, fh - 12), (fw, fh),         (50, 50, 50), -1)
            cv2.rectangle(overlay, (0, fh - 12), (bar_filled, fh), color,        -1)

            cv2.imshow(window, overlay)
            cv2.waitKey(1)

        print(f'  {emotion}: saved {captured} images ({skipped} frames had no face)')

    cv2.destroyWindow(window)
    print('Capture complete.\n')


# ── Fine-tune ──────────────────────────────────────────────────────────────────

class PersonalDataset(torch.utils.data.Dataset):
    def __init__(self, augment=True):
        self.samples  = []   # (image_array, label_idx)
        self.augment  = augment
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
        for idx, emotion in enumerate(EMOTIONS):
            folder = PERSONAL_DIR / emotion
            for img_path in folder.glob('*.png'):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.samples.append((img, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, label = self.samples[i]
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        if self.augment:
            tensor = self.transform(tensor)
        return tensor, label


def fine_tune(device):
    print('Fine-tuning model on your face data...')
    dataset = PersonalDataset(augment=True)
    if len(dataset) == 0:
        print('No personal images found — skipping fine-tune.')
        return

    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    model = CNN().to(device)
    model.load_state_dict(torch.load(BASE_MODEL, map_location=device))

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)

    best_loss = float('inf')

    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        total_loss = 0
        correct    = 0
        total      = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        avg_loss = total_loss / len(loader)
        acc      = correct / total
        print(f'  Epoch {epoch+1:>2}/{FINETUNE_EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.1%}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), PERSONAL_MODEL)

    print(f'Fine-tuning done. Personal model saved to {PERSONAL_MODEL}\n')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    reset = '--reset' in sys.argv
    if reset and PERSONAL_MODEL.exists():
        os.remove(PERSONAL_MODEL)
        print('Personal model removed. Will re-run setup.\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cap    = cv2.VideoCapture(0)

    # ── First-run setup ───────────────────────────────────────────────────────
    if not PERSONAL_MODEL.exists():
        print('No personal model found — starting guided capture.\n')
        print('You will be prompted to make each of these expressions:')
        for e in EMOTIONS:
            print(f'  • {e}')
        print()

        run_capture_session(cap)
        fine_tune(device)

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = PERSONAL_MODEL if PERSONAL_MODEL.exists() else BASE_MODEL
    print(f'Loading model: {model_path}')

    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── Detection loop ────────────────────────────────────────────────────────
    prob_buffer = deque(maxlen=SMOOTH_FRAMES)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        face    = detect_face(gray_eq)

        if face is None:
            prob_buffer.clear()
        else:
            x, y, w, h = face
            crop   = crop_face(gray, face)
            tensor = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0

            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

            prob_buffer.append(probs)
            avg_probs  = np.mean(prob_buffer, axis=0)
            pred       = int(np.argmax(avg_probs))
            emotion    = EMOTIONS[pred]
            confidence = float(avg_probs[pred])
            color      = EMOTION_COLORS[emotion]

            # face box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # main label
            cv2.putText(frame, f'{emotion}  {confidence:.0%}',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

            # per-emotion bar chart
            bar_x, bar_y = 10, 10
            bar_w_max    = 160
            bar_h, gap   = 16, 5
            for i, (prob, emo) in enumerate(zip(avg_probs, EMOTIONS)):
                emo_color  = EMOTION_COLORS[emo]
                filled     = int(bar_w_max * prob)
                by         = bar_y + i * (bar_h + gap)
                cv2.rectangle(frame, (bar_x, by),             (bar_x + bar_w_max, by + bar_h), (50, 50, 50), -1)
                if filled > 0:
                    cv2.rectangle(frame, (bar_x, by),         (bar_x + filled, by + bar_h),    emo_color,    -1)
                cv2.putText(frame, f'{emo:<8} {prob:.0%}',
                            (bar_x + bar_w_max + 6, by + bar_h - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, emo_color, 1)

        cv2.imshow('Emotion Detection  —  Q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
