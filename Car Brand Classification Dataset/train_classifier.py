import os
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from tqdm import tqdm

from dataset import build_loaders


def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def train_classifier(data_root: str, epochs: int = 10, batch_size: int = 32, lr: float = 3e-4, img_size: int = 224, save_path: str = 'classifier_resnet18.pt') -> Dict[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    train_loader, val_loader, test_loader, num_classes = build_loaders(data_root, img_size=img_size, batch_size=batch_size)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', ncols=100)
        running_loss = 0.0
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            pbar.set_postfix(loss=loss.item())

        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'model_state': model.state_dict(), 'num_classes': num_classes}, save_path)
        tqdm.write(f'Epoch {epoch}: val_acc={val_acc:.4f}')

    test_acc = evaluate(model, test_loader, device)
    tqdm.write(f'Test acc: {test_acc:.4f}')
    return {'val_acc': best_val, 'test_acc': test_acc}


if __name__ == '__main__':
    data_root = os.path.dirname(os.path.abspath(__file__))
    train_classifier(data_root)
