# Real-Time Emotion Detection — Built from Scratch

A full end-to-end deep learning project detecting 5 facial emotions in real time via webcam.
Built in two stages: first a **pure NumPy CNN** (no ML libraries, full backprop by hand), then a full **PyTorch** pipeline with GPU training and live inference.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)
![NumPy](https://img.shields.io/badge/NumPy-from--scratch-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-red)

---

## What I Built & Skills Demonstrated

### Stage 1 — CNN from scratch with NumPy (`archive/cnn_numpy.py`)
> **No PyTorch. No autograd. Just NumPy and math.**

Built every component of a CNN by hand:

| Component | What I implemented |
|-----------|-------------------|
| Convolutional layer | Manual filter sliding, patch extraction, dot products |
| ReLU activation | Element-wise `max(0, x)` |
| Max pooling | 2x2 window with argmax tracking via `max_mask` |
| Flatten | Reshape `(C, H, W)` to `(N,)` for FC layers |
| Fully connected layers | Matrix multiply + bias |
| Softmax | `exp(x) / sum(exp(x))` |
| Cross-entropy loss | `-log(p_true)` |
| Backpropagation | Full manual chain rule through conv, pool, FC1, FC2 |
| Mini-batch SGD | Gradient accumulation across batch, then weight update |

This demonstrates a deep understanding of how neural networks actually work under the hood.

---

### Stage 2 — Production PyTorch Pipeline (`train_pytorch.py`)

Upgraded to PyTorch with a significantly deeper and more robust architecture:

**Model Architecture:**
```
Input: (1, 48, 48) grayscale face crop

Block 1: Conv2d(1->32,  3x3, pad=1) -> BatchNorm2d -> ReLU -> MaxPool  -> (32, 24, 24)
Block 2: Conv2d(32->64, 3x3, pad=1) -> BatchNorm2d -> ReLU -> MaxPool  -> (64, 12, 12)
Block 3: Conv2d(64->128,3x3, pad=1) -> BatchNorm2d -> ReLU -> MaxPool  -> (128, 6, 6)

Flatten -> Linear(4608->128) -> ReLU -> Dropout(0.05) -> Linear(128->5)
                                                                  |
                                                    [Angry, Happy, Fear, Sad, Surprise]
```

**Training improvements implemented:**

| Technique | Why |
|-----------|-----|
| BatchNorm2d after every conv | Stabilises gradients, faster convergence |
| 3 conv blocks instead of 2 | More capacity to learn subtle expressions |
| Class oversampling | Dataset had 2x more Happy than Surprise — all classes upsampled to equal count (12,866 each) |
| Weighted CrossEntropyLoss | Inverse-frequency weights per class — rare classes penalised more |
| ReduceLROnPlateau scheduler | Halves LR when val loss stalls (patience=3, factor=0.5) |
| GPU training via CUDA | RTX 3060 — ~1-2 min/epoch vs hours on CPU |
| Data augmentation | Random horizontal flip + rotation on training set |

**Final validation accuracy: ~78% across 5 classes**

---

### Stage 3 — Real-Time Webcam Inference (`detect.py`)

- OpenCV Haar cascade face detection with histogram equalisation for lighting robustness
- Preprocessing pipeline matches training exactly: grayscale -> equalizeHist -> resize 48x48 -> normalise [0,1]
- **Temporal smoothing:** rolling average of softmax probabilities over last 15 frames — eliminates per-frame jitter
- GPU inference via CUDA

---

## Project Structure

```
emotiondetection/
|
|-- archive/
|   +-- cnn_numpy.py          # Stage 1: full CNN from scratch — NumPy only, no frameworks
|
|-- data/                     # Raw images (Angry / Happy / Fear / Sad / Surprise)
|
|-- scripts/
|   +-- prepare_dataset.py    # Loads images, converts to grayscale 48x48, saves dataset.npz
|
|-- src/
|   |-- dataset.py            # PyTorch Dataset class with augmentation
|   +-- train.py              # Modular training utilities
|
|-- train_pytorch.py          # Main training script — run this to train
|-- detect.py                 # Real-time webcam emotion detection — run this to demo
|-- requirements.txt
+-- README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU (recommended):** For ~10x faster training on NVIDIA GPU:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

### 2. Prepare the dataset
```bash
python scripts/prepare_dataset.py
```
Reads images from `data/`, converts to grayscale 48x48, saves `dataset.npz` with train/val/test splits.

### 3. Train the model
```bash
python train_pytorch.py
```
Trains for 200 epochs, saves best weights to `best_model.pth`. GPU used automatically if available.

### 4. Run live detection
```bash
python detect.py
```
Opens your webcam. Press **Q** to quit.

---

## Dataset

- **5 emotion classes:** Angry, Happy, Fear, Sad, Surprise
- **~59,000 grayscale 48x48 images** split 70% train / 15% val / 15% test
- Class imbalance handled via oversampling — all classes equalised to 12,866 training samples each

---

## Tech Stack

| Tool | Used for |
|------|----------|
| **NumPy** | CNN from scratch — conv, pool, backprop, SGD, all by hand |
| **PyTorch** | Production model, GPU training, DataLoader pipeline |
| **OpenCV** | Face detection, webcam capture, frame preprocessing |
| **torchvision** | Data augmentation |
| **CUDA** | GPU-accelerated training and inference |

---

## Key Concepts Demonstrated

- Convolutional Neural Networks — architecture, receptive fields, feature maps
- Backpropagation through conv, pool, and FC layers — derived and implemented manually in NumPy
- Batch Normalisation and why it stabilises training
- Class imbalance — detection, oversampling, and weighted loss solutions
- Progression from NumPy prototype to framework-based production code
- Real-time inference with temporal smoothing for stable predictions
