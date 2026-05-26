import torch
import torch.nn as nn
import torch.optim
import numpy as np
import torchvision.transforms as transforms

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)    # 1 grayscale input, 32 filters, 3x3 kernel
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)   # 32 input channels, 64 filters, 3x3 kernel
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 64 input channels, 128 filters, 3x3 kernel
        self.bn3   = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.05)
        self.pool = nn.MaxPool2d(2, 2)                  # halves spatial dims
        self.fc1 = nn.Linear(6*6*128, 128)              # flatten → 128 neurons
        self.fc2 = nn.Linear(128, 5)                    # 128 → 5 emotion classes   

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))   # conv + BN + activate
        x = self.pool(x)                           # downsample: 48x48 → 24x24
        x = torch.relu(self.bn2(self.conv2(x)))   # conv 2 + BN + activate
        x = self.pool(x)                           # downsample: 24x24 → 12x12
        x = torch.relu(self.bn3(self.conv3(x)))   # conv 3 + BN + activate
        x = self.pool(x)                           # downsample: 12x12 → 6x6
        x = x.view(x.size(0), -1)                 # flatten, keeping batch dim
        x = torch.relu(self.fc1(x))               # FC + activate
        x = self.dropout(x)
        x = self.fc2(x)                           # raw logits — no softmax (CrossEntropyLoss handles it)
        return x


class LoadData(torch.utils.data.Dataset):

    def __init__(self, split):
        super().__init__()
        data = np.load('dataset.npz')

        # pick the right split
        if split == 'train':
            X_raw = data['X_train']
            y_raw = data['y_train']
            self.augment = True

            # oversample minority classes so every class has equal representation
            # this stops the model collapsing to the dominant class (Happy)
            labels     = np.argmax(y_raw, axis=1)
            max_count  = int(np.bincount(labels).max())
            X_balanced, y_balanced = [], []
            for cls in range(5):
                idx = np.where(labels == cls)[0]
                # repeat indices until we reach max_count, then slice exactly
                repeated = np.resize(idx, max_count)
                X_balanced.append(X_raw[repeated])
                y_balanced.append(y_raw[repeated])
            self.X = np.concatenate(X_balanced)
            self.y = np.concatenate(y_balanced)

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
    model = CNN().to(device)

    # class weights — inverse frequency so rare classes (Fear, Angry, Surprise)
    # get a stronger gradient signal than the dominant class (Happy)
    labels_train  = np.argmax(train_dataset.y, axis=1)
    class_counts  = np.bincount(labels_train, minlength=5).astype(float)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    class_weights = (class_weights / class_weights.sum() * 5).to(device)  # keep scale near 1

    loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
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
        
        scheduler.step(val_loss)

        print(f'Epoch {i+1:>2} | Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Acc: {accuracy:.4f}')
