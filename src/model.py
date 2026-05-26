import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Emotion CNN — 5-class grayscale classifier.

    Input:  (B, 1, 48, 48)
    Output: (B, 5) raw logits  [use with nn.CrossEntropyLoss]

    Architecture
    ────────────
    Conv1  (1→32,  3×3) → ReLU → MaxPool → 23×23
    Conv2  (32→64, 3×3) → ReLU → MaxPool → 10×10
    FC1    (6 400 → 128) → ReLU → Dropout(0.25)
    FC2    (128 → 5)      ← logits
    """

    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, 3)
        self.conv2   = nn.Conv2d(32, 64, 3)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1     = nn.Linear(10 * 10 * 64, 128)
        self.fc2     = nn.Linear(128, 5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # → (B, 32, 23, 23)
        x = self.pool(torch.relu(self.conv2(x)))   # → (B, 64, 10, 10)
        x = x.view(x.size(0), -1)                 # → (B, 6400)
        x = self.dropout(torch.relu(self.fc1(x))) # → (B, 128)
        x = self.fc2(x)                           # → (B, 5)
        return x
