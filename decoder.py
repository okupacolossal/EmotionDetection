import os
import cv2
import numpy as np
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = 'data'
EMOTIONS = ['Angry', 'Happy', 'Fear', 'Sad', 'Surprise']
IMG_SIZE = 48

# ── Helpers ──────────────────────────────────────────────────────────────────
def one_hot(index, total):
    label = [0] * total
    label[index] = 1
    return label

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

# ── Loading ──────────────────────────────────────────────────────────────────
def load_images():
    inputs, targets = [], []

    for i, emotion in enumerate(EMOTIONS):
        label = one_hot(i, len(EMOTIONS))
        folder = os.path.join(DATA_DIR, emotion)
        files = [f for f in os.listdir(folder) if f.endswith(('jpg', 'png'))]

        for filename in tqdm(files, desc=f"Loading {emotion}"):
            img = preprocess_image(os.path.join(folder, filename))
            inputs.append(img)
            targets.append(label)

    return np.array(inputs), np.array(targets)

# ── Split ────────────────────────────────────────────────────────────────────
def split(X, y, train_ratio=0.70, val_ratio=0.85):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    train_end = int(train_ratio * len(X))
    val_end   = int(val_ratio  * len(X))

    return (
        X[indices[:train_end]],  y[indices[:train_end]],
        X[indices[train_end:val_end]], y[indices[train_end:val_end]],
        X[indices[val_end:]],    y[indices[val_end:]]
    )

# ── Main ─────────────────────────────────────────────────────────────────────
X, y = load_images()
X_train, y_train, X_val, y_val, X_test, y_test = split(X, y)

np.savez('dataset.npz',
         X_train=X_train, y_train=y_train,
         X_val=X_val,     y_val=y_val,
         X_test=X_test,   y_test=y_test)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")