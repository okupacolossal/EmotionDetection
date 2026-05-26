"""
Build dataset.npz from raw images.

Folder layout expected:
    data/
        Angry/    *.png / *.jpg
        Happy/    ...
        Fear/     ...
        Sad/      ...
        Surprise/ ...

Usage (from project root):
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --data data --out dataset.npz
"""

import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
EMOTIONS  = ['Angry', 'Happy', 'Fear', 'Sad', 'Surprise']
IMG_SIZE  = 48


# ── Helpers ───────────────────────────────────────────────────────────────────

def one_hot(index: int, total: int) -> list:
    label = [0] * total
    label[index] = 1
    return label


def preprocess_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img   # uint8 [0, 255] — normalisation is done at training time


# ── Loading ───────────────────────────────────────────────────────────────────

def load_images(data_dir: str):
    inputs, targets = [], []

    for i, emotion in enumerate(EMOTIONS):
        label  = one_hot(i, len(EMOTIONS))
        folder = os.path.join(data_dir, emotion)

        if not os.path.isdir(folder):
            print(f'[WARN] Missing folder: {folder} — skipping.')
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(('jpg', 'png'))]
        for filename in tqdm(files, desc=f'Loading {emotion:<10}'):
            img = preprocess_image(os.path.join(folder, filename))
            inputs.append(img)
            targets.append(label)

    return np.array(inputs, dtype=np.uint8), np.array(targets, dtype=np.uint8)


# ── Split ─────────────────────────────────────────────────────────────────────

def split(X, y, train_ratio=0.70, val_ratio=0.85):
    idx = np.random.permutation(len(X))
    t   = int(train_ratio * len(X))
    v   = int(val_ratio   * len(X))
    return (
        X[idx[:t]],  y[idx[:t]],
        X[idx[t:v]], y[idx[t:v]],
        X[idx[v:]],  y[idx[v:]],
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Prepare emotion dataset → .npz')
    p.add_argument('--data', default='data',        help='Root folder of raw images')
    p.add_argument('--out',  default='dataset.npz', help='Output .npz path')
    p.add_argument('--seed', type=int, default=42,  help='Random seed for split')
    args = p.parse_args()

    np.random.seed(args.seed)

    print(f'Reading images from: {args.data}')
    X, y = load_images(args.data)
    print(f'Total images loaded: {len(X)}')

    X_train, y_train, X_val, y_val, X_test, y_test = split(X, y)

    np.savez_compressed(
        args.out,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
    )

    print(f'\nSplit  →  train {len(X_train):>6} | val {len(X_val):>6} | test {len(X_test):>6}')
    print(f'Saved  →  {args.out}')


if __name__ == '__main__':
    main()
