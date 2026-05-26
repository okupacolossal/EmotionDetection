import cv2
import torch
import numpy as np
from collections import deque
from train_pytorch import CNN

EMOTIONS      = ['Angry', 'Happy', 'Fear', 'Sad', 'Surprise']
IMG_SIZE      = 48
SMOOTH_FRAMES = 15   # average probs over this many frames before deciding

# ── Load model ────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = CNN().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# ── Face detector ─────────────────────────────────────────────────────────────
# scaleFactor=1.1  → smaller steps, catches more face sizes
# minNeighbors=4   → slightly lenient, fewer missed detections
# minSize=(60,60)  → ignore tiny blobs that aren't faces

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ── Camera ────────────────────────────────────────────────────────────────────

cap        = cv2.VideoCapture(0)
prob_buffer = deque(maxlen=SMOOTH_FRAMES)   # rolling window of per-frame probs

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)   # normalise lighting before detection

    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor  = 1.05,
        minNeighbors = 3,
        minSize      = (60, 60),
    )

    if len(faces) == 0:
        prob_buffer.clear()   # reset when face disappears

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        if face.size == 0:
            continue
        face   = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face   = cv2.equalizeHist(face)   # normalise brightness to match training data
        tensor = torch.tensor(face, dtype=torch.float32) / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)   # (1, 1, 48, 48)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        prob_buffer.append(probs)

        # average over the last SMOOTH_FRAMES frames — kills jitter
        avg_probs  = np.mean(prob_buffer, axis=0)
        pred       = int(np.argmax(avg_probs))
        emotion    = EMOTIONS[pred]
        confidence = float(avg_probs[pred])

        # draw box + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
        cv2.putText(frame, f'{emotion}  {confidence:.0%}',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    cv2.imshow('Emotion Detection  —  Q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
