import numpy as np
import torch
import torchvision.transforms as transforms

# ── Shared constants ──────────────────────────────────────────────────────────
EMOTIONS  = ['Angry', 'Happy', 'Fear', 'Sad', 'Surprise']
IMG_SIZE  = 48
NUM_CLASSES = len(EMOTIONS)   # 5


class EmotionDataset(torch.utils.data.Dataset):
    """
    Loads pre-built dataset.npz and serves (image, label) pairs.

    Parameters
    ----------
    split : 'train' | 'val' | 'test'
    npz_path : path to dataset.npz (default: 'dataset.npz' relative to cwd)
    """

    def __init__(self, split: str, npz_path: str = 'dataset.npz'):
        super().__init__()
        data = np.load(npz_path)

        if split == 'train':
            self.X, self.y = data['X_train'], data['y_train']
            self.augment = True
        elif split == 'val':
            self.X, self.y = data['X_val'], data['y_val']
            self.augment = False
        elif split == 'test':
            self.X, self.y = data['X_test'], data['y_test']
            self.augment = False
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'train', 'val', or 'test'.")

        self._augment_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])

    # ── Protocol ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        # [0,255] float → [0,1] tensor with channel dim  (1, H, W)
        image = (torch.tensor(self.X[idx]).float() / 255.0).unsqueeze(0)
        # one-hot → class index
        label = torch.argmax(torch.tensor(self.y[idx]).float()).long()

        if self.augment:
            image = self._augment_tf(image)

        return image, label
