from pathlib import Path
from typing import List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

# Build Albumentations pipeline from YAML list
A_MAP = {
    'RandomResizedCrop': A.RandomResizedCrop,
    'HorizontalFlip': A.HorizontalFlip,
    'VerticalFlip': A.VerticalFlip,
    'CenterCrop': A.CenterCrop,
    'RandomBrightnessContrast': A.RandomBrightnessContrast,
    'HueSaturationValue': A.HueSaturationValue,
    'GaussNoise': A.GaussNoise,
    'Resize': A.Resize,
}

def build_aug(list_cfg, img_size: int):
    if not list_cfg:
        list_cfg = []
    tfms = []
    # safety: inject resize if no crop
    has_crop = any(c['name'] in ('RandomResizedCrop','CenterCrop') for c in list_cfg)
    if not has_crop:
        tfms.append(A.Resize(img_size, img_size))
    for c in list_cfg:
        cls = A_MAP[c['name']]
        tfms.append(cls(**c.get('params', {})))
    tfms.append(A.Normalize())
    tfms.append(ToTensorV2())
    return A.Compose(tfms)

class EuroSATDataset(Dataset):
    def __init__(self, paths: List[Path], labels: List[int], aug=None):
        self.paths = paths
        self.labels = labels
        self.aug = aug

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = np.array(Image.open(p).convert('RGB'))
        if self.aug:
            img = self.aug(image=img)['image']
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, y

def load_paths_labels(root: Path):
    # assumes ImageFolder-like structure or flat images with labels.txt
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for c in classes:
        for p in (root / c).glob('*.jpg'):
            paths.append(p)
            labels.append(class_to_idx[c])
        for p in (root / c).glob('*.png'):
            paths.append(p)
            labels.append(class_to_idx[c])
    return paths, labels, classes

def stratified_split(paths, labels, val_size=0.15, test_size=0.0, seed=42):
    X = np.array(paths)
    y = np.array(labels)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_size+test_size, stratify=y, random_state=seed)
    if test_size > 0:
        rel = test_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=rel, stratify=y_tmp, random_state=seed)
    else:
        X_val, y_val = X_tmp, y_tmp
        X_test, y_test = [], []
    return list(X_train), list(y_train), list(X_val), list(y_val), list(X_test), list(y_test)