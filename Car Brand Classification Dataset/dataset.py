import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(img_size: int = 224, train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def build_loaders(root_dir: str, img_size: int = 224, batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    test_dir = os.path.join(root_dir, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=build_transforms(img_size, train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=build_transforms(img_size, train=False))
    test_dataset = datasets.ImageFolder(test_dir, transform=build_transforms(img_size, train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    num_classes = len(train_dataset.classes)
    return train_loader, val_loader, test_loader, num_classes


def unnormalize(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(1, 3, 1, 1)
    return t * std + mean
