# Real-Time Emotion Detection — Built from Scratch

> A full end-to-end deep learning project: custom CNN trained on 59k images, real-time webcam inference, and a **self-personalisation system** that fine-tunes the model to your face in under 2 minutes.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-red?logo=opencv&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What This Project Demonstrates

This project was built in three progressive stages — each one intentionally chosen to demonstrate a deeper understanding of how deep learning actually works, not just how to call a library.

| Stage | What I did | Why it matters |
|---|---|---|
| **Stage 1** | Implemented a full CNN in NumPy — conv layers, backprop, SGD — by hand | Proves I understand the maths, not just the API |
| **Stage 2** | Built a production PyTorch pipeline with GPU training, class balancing, and augmentation | Demonstrates real ML engineering: handling imbalanced data, regularisation, scheduling |
| **Stage 3** | Added a guided personalisation system that fine-tunes the model to any user's face | Shows understanding of transfer learning and domain shift |

---

## Demo

```
python detect.py
```

**First run:** A guided setup walks you through showing each expression on camera.
Your face data is collected, the model is fine-tuned, and the personalised model is saved.

**Every run after:** Loads your personalised model instantly and runs live detection.

```
python detect.py --reset    # redo the personalisation setup
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/okupacolossal/emotiondetection.git
cd emotiondetection
```

### 2. Install dependencies

**CPU only:**
```bash
pip install -r requirements.txt
```

**GPU (recommended — ~10× faster training):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. Run live detection
```bash
python detect.py
```

That's it. On first run the app will guide you through personalisation automatically.

---

## Personalisation System

Generic emotion models struggle with real webcam footage because they are trained on acted, studio-lit images that look nothing like a live camera feed. This is called **domain shift**.

To solve this, the app includes a guided fine-tuning flow:

```
First launch
    │
    ├─ No personal model found
    │       │
    │       ├─ Guided capture (≈ 40 seconds)
    │       │     For each of 5 emotions:
    │       │       • On-screen prompt: "Show ANGRY face"
    │       │       • 3s countdown → 5s of auto-capture
    │       │       • 50 labelled face crops saved automatically
    │       │
    │       └─ Fine-tuning (≈ 1–2 minutes)
    │             • Loads pretrained base model
    │             • Fine-tunes all layers at LR = 5×10⁻⁵
    │             • 40 epochs with augmentation (flip, rotation, affine)
    │             • Saves best_model_personal.pth
    │
    └─ Personal model found → load and run immediately
```

The fine-tuning uses a very small learning rate so the model retains its general knowledge from 59k training images while adapting to your specific face and lighting conditions.

---

## Training the Base Model

> Skip this if you just want to run detection — `best_model.pth` is already included.

### Prepare the dataset
```bash
python scripts/prepare_dataset.py
```
Reads images from `data/`, converts to grayscale 48×48, saves `dataset.npz` with 70/15/15 train/val/test splits.

### Train
```bash
python train_pytorch.py
```
Trains for 200 epochs, saves best weights to `best_model.pth`. GPU used automatically if available.

**Final validation accuracy: ~78% across 5 classes**

---

## Model Architecture

```
Input: (1, 48, 48) — grayscale face crop

Block 1:  Conv2d(1→32,  3×3, pad=1) → BatchNorm2d → ReLU → MaxPool   →  (32, 24, 24)
Block 2:  Conv2d(32→64, 3×3, pad=1) → BatchNorm2d → ReLU → MaxPool   →  (64, 12, 12)
Block 3:  Conv2d(64→128,3×3, pad=1) → BatchNorm2d → ReLU → MaxPool   →  (128, 6, 6)

Flatten → Linear(4608→128) → ReLU → Dropout(0.05) → Linear(128→5)
                                                              │
                                          [Angry, Happy, Fear, Sad, Surprise]
```

**Training techniques used:**

| Technique | Purpose |
|---|---|
| BatchNorm2d after every conv | Stabilises gradients, faster convergence |
| Class oversampling | Dataset had 2× more Happy than Surprise — all classes equalised to 12,866 each |
| Weighted CrossEntropyLoss | Rare classes (Fear, Angry) get stronger gradient signal |
| ReduceLROnPlateau scheduler | Halves LR when val loss stalls (patience=3, factor=0.5) |
| Data augmentation | Random horizontal flip + rotation |
| GPU via CUDA | RTX 3060 — ~1–2 min/epoch |

---

## Stage 1 — CNN from Scratch (NumPy only)

> `archive/cnn_numpy.py` — no PyTorch, no autograd. Just NumPy and maths.

Every component built by hand:

| Component | Implementation |
|---|---|
| Convolutional layer | Manual filter sliding, patch extraction, dot products |
| ReLU | Element-wise `max(0, x)` |
| Max pooling | 2×2 window with argmax tracking via `max_mask` |
| Flatten | Reshape `(C, H, W)` → `(N,)` |
| Fully connected layers | Matrix multiply + bias |
| Softmax | `exp(x) / sum(exp(x))` |
| Cross-entropy loss | `−log(p_true)` |
| Backpropagation | Full manual chain rule through conv, pool, FC1, FC2 |
| Mini-batch SGD | Gradient accumulation + weight update |

---

## Dataset

- **Source:** FER2013 (Facial Expression Recognition)
- **5 classes:** Angry, Happy, Fear, Sad, Surprise
- **~59,000** grayscale 48×48 images
- **Split:** 70% train / 15% val / 15% test
- Class imbalance handled via oversampling — all classes equalised to 12,866 training samples each

---

## Project Structure

```
emotiondetection/
│
├── detect.py               # Entry point — run this for live detection + personalisation
├── train_pytorch.py        # Training script + CNN architecture (CNN class imported by detect.py)
│
├── scripts/
│   └── prepare_dataset.py  # Converts raw images → dataset.npz
│
├── archive/
│   └── cnn_numpy.py        # Stage 1: full CNN from scratch, NumPy only
│
├── src/                    # Modular training utilities
│   ├── dataset.py
│   └── train.py
│
├── best_model.pth          # Pretrained base model weights (Git LFS)
├── dataset.npz             # Preprocessed training data (Git LFS)
├── requirements.txt
└── README.md
```

> `best_model_personal.pth` and `personal_data/` are generated locally and not committed.

---

## Tech Stack

| Tool | Role |
|---|---|
| **PyTorch** | Model architecture, GPU training, inference |
| **NumPy** | CNN from scratch — all maths by hand |
| **OpenCV** | Webcam capture, face detection, real-time display |
| **torchvision** | Data augmentation |
| **CUDA** | GPU-accelerated training and inference |

---

## Key Concepts Demonstrated

- **Convolutional Neural Networks** — architecture, receptive fields, feature maps
- **Backpropagation** — derived and implemented manually through conv, pool, and FC layers
- **Transfer learning / fine-tuning** — adapting a pretrained model to a new domain with limited data
- **Domain shift** — identifying and solving the gap between training data and real-world data
- **Class imbalance** — detection, oversampling, and weighted loss solutions
- **Batch Normalisation** — why it stabilises training
- **Real-time inference** — temporal smoothing for stable predictions
- **Production ML** — clean pipeline from raw data to live demo

---

## License

MIT — free to use, fork, and build on.
