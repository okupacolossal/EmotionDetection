import cv2
import torch
import numpy as np
import mediapipe as mp
from train_pytorch import CNN

EMOTIONS = ['Angry', 'Happy', 'Fear', 'Sad', 'Surprise']
IMG_SIZE  = 48

# ── Load model ────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = CNN().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# ── Face detector (mediapipe — neural net based, much better than Haar) ───────

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# ── Camera ────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame = frame.shape[:2]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if results.detections:
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box

            # convert relative coords to pixel coords
            x = max(0, int(box.xmin * w_frame))
            y = max(0, int(box.ymin * h_frame))
            w = int(box.width  * w_frame)
            h = int(box.height * h_frame)
            x2 = min(w_frame, x + w)
            y2 = min(h_frame, y + h)

            # preprocess face crop → model input
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = gray[y:y2, x:x2]
            if face.size == 0:
                continue
            face   = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            tensor = torch.tensor(face, dtype=torch.float32) / 255.0
            tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)   # (1, 1, 48, 48)

            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1)[0]
                pred   = torch.argmax(probs).item()

            emotion    = EMOTIONS[pred]
            confidence = probs[pred].item()

            # draw box + label
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, f'{emotion}  {confidence:.0%}',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    cv2.imshow('Emotion Detection  —  Q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
