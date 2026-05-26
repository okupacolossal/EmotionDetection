"""
Training script for EmotionCNN.

Usage (from project root):
    python -m src.train
    python -m src.train --epochs 100 --lr 0.0005 --batch 64
"""

import argparse
import torch
import torch.nn as nn

from src.model   import CNN
from src.dataset import EmotionDataset


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train EmotionCNN')
    p.add_argument('--epochs',   type=int,   default=200,    help='Number of training epochs')
    p.add_argument('--lr',       type=float, default=1e-3,   help='Adam learning rate')
    p.add_argument('--batch',    type=int,   default=32,     help='Batch size')
    p.add_argument('--npz',      type=str,   default='dataset.npz', help='Path to dataset.npz')
    p.add_argument('--weights',  type=str,   default='weights/best_model.pth',
                   help='Where to save the best model weights')
    return p.parse_args()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader = torch.utils.data.DataLoader(
        EmotionDataset('train', args.npz), batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = torch.utils.data.DataLoader(
        EmotionDataset('val',   args.npz), batch_size=args.batch, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    model     = CNN().to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += loss_fn(logits, labels).item()
                preds     = torch.argmax(logits, dim=1)
                correct  += (preds == labels).sum().item()
                total    += labels.size(0)

        # ── Log ──────────────────────────────────────────────────────────────
        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        accuracy    = correct / total

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), args.weights)
            marker = '  ✓ saved'
        else:
            marker = ''

        print(f'Epoch {epoch:>3}/{args.epochs} | '
              f'Train {train_loss:.4f}  Val {val_loss:.4f}  Acc {accuracy:.4f}{marker}')

    print(f'\nBest val accuracy: {best_accuracy:.4f}')
    print(f'Weights saved to : {args.weights}')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train(parse_args())
