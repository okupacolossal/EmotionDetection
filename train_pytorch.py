import torch
import torch.nn as nn
import torch.optim
import numpy as np
import torchvision.transforms as transforms

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)   # 1 grayscale input, 32 filters, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, 3)  # 32 input channels, 64 filters, 3x3 kernel
        self.dropout = nn.Dropout(0.05)
        self.pool = nn.MaxPool2d(2, 2)      # halves spatial dims
        self.fc1 = nn.Linear(10*10*64, 128) # flatten →  128 neurons
        self.fc2 = nn.Linear(128, 5)        # 128 → 5 emotion classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))   # conv + activate
        x = self.pool(x)                # downsample: 46x46 → 23x23
        x = torch.relu(self.conv2(x))   # conv 2 + activate [64 * 10 * 10]
        x = self.pool(x)
        x = x.view(x.size(0), -1)       # flatten, keeping batch dim
        x = torch.relu(self.fc1(x))     # FC + activate
        x = self.dropout(x)
        x = self.fc2(x)                 # raw logits — no softmax (CrossEntropyLoss handles it)
        return x


class LoadData(torch.utils.data.Dataset):

    def __init__(self, split):
        super().__init__()
        data = np.load('dataset.npz')

        # pick the right split
        if split == 'train':
            self.X = data['X_train']
            self.y = data['y_train']
            self.augment = True
        elif split == 'val':
            self.X = data['X_val']
            self.y = data['y_val']
            self.augment = False
        elif split == 'test':
            self.X = data['X_test']
            self.y = data['y_test']
            self.augment = False

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        image = (torch.tensor(self.X[idx]).float() / 255).unsqueeze(0)  # [0,255] → [0,1], add channel dim
        label = torch.argmax(torch.tensor(self.y[idx]).float()).long()   # OHE → class index

        if self.augment:
            transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            ])
            image = transform(image)
        return image, label


if __name__ == '__main__':
    # everything below only runs when you do `python train_pytorch.py` directly
    # if another file imports CNN from here, this block is skipped entirely~

    train_dataset = LoadData('train')
    val_dataset   = LoadData('val')
    test_dataset  = LoadData('test')

    # shuffle train so batches aren't always the same order; val/test order doesn't matter
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=32, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    print(device)
    model     = CNN().to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Adam: adaptive gradient descent — adjusts the lr per weight automatically

    best_accuracy = 0

    for i in range(200):
        train_loss = 0
        val_loss   = 0
        correct    = 0
        total      = 0

        # ── train ──────────────────────────────────────────────
        model.train()
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()           # clear last batch's gradients
            x = model(image)
            loss = loss_fn(x, label)
            loss.backward()                 # compute gradients
            optimizer.step()                # update weights
            train_loss += loss.item()

        # ── validate ───────────────────────────────────────────
        model.eval()
        with torch.no_grad():               # no gradients needed — faster + less memory
            for image, label in val_loader:
                image, label = image.to(device), label.to(device)
                x           = model(image)
                loss        = loss_fn(x, label)
                predictions = torch.argmax(x, dim=1)    # highest logit = predicted class
                correct     += (predictions == label).sum().item()
                total       += label.size(0)
                val_loss    += loss.item()

        # ── log ────────────────────────────────────────────────
        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        accuracy    = correct / total

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')    # save weights only, not full model

        print(f'Epoch {i+1:>2} | Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Acc: {accuracy:.4f}')
