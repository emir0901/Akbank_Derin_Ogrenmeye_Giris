import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from dataset import build_loaders, unnormalize


class SimpleUNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, base, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(base * 4, base * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(base * 2, base, 3, padding=1), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        u2 = self.up2(e3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        y = self.out(d1)
        return torch.tanh(y)


def load_classifier(classifier_path: str, num_classes: int) -> nn.Module:
    from torchvision import models
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    state = torch.load(classifier_path, map_location='cpu')
    model.load_state_dict(state['model_state'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def train_obfuscator(data_root: str, classifier_path: str = 'classifier_resnet18.pt', epochs: int = 5, batch_size: int = 16, lr: float = 1e-4, img_size: int = 224, save_path: str = 'obfuscator.pt', lambda_uniform: float = 1.0, lambda_recon: float = 10.0) -> Dict[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    train_loader, val_loader, test_loader, num_classes = build_loaders(data_root, img_size=img_size, batch_size=batch_size)

    clf = load_classifier(os.path.join(data_root, classifier_path), num_classes).to(device)
    net = SimpleUNet(base=32).to(device)
    opt = Adam(net.parameters(), lr=lr)

    uniform_target = torch.full((batch_size, num_classes), 1.0 / num_classes)

    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        net.train()
        pbar = tqdm(train_loader, desc=f'Obf Epoch {epoch}/{epochs}', ncols=100)
        running = 0.0
        for images, _ in pbar:
            images = images.to(device)
            opt.zero_grad(set_to_none=True)

            delta = net(images)
            x_tilde = torch.clamp(unnormalize(images) + 0.1 * delta, 0.0, 1.0)
            x_tilde_norm = (x_tilde - torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

            with torch.no_grad():
                pass
            logits = clf(x_tilde_norm)
            probs = F.softmax(logits, dim=1)

            u = uniform_target[:probs.size(0), :].to(device)
            loss_uniform = F.kl_div((probs + 1e-8).log(), u, reduction='batchmean')
            loss_recon = F.l1_loss(x_tilde, torch.clamp(unnormalize(images), 0.0, 1.0))
            loss = lambda_uniform * loss_uniform + lambda_recon * loss_recon
            loss.backward()
            opt.step()

            running += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        # simple validation: measure confidence reduction on val set
        net.eval()
        with torch.no_grad():
            confs = []
            for images, _ in val_loader:
                images = images.to(device)
                delta = net(images)
                x_tilde = torch.clamp(unnormalize(images) + 0.1 * delta, 0.0, 1.0)
                x_tilde_norm = (x_tilde - torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                probs = F.softmax(clf(x_tilde_norm), dim=1)
                confs.append(probs.max(dim=1).values.mean().item())
            mean_conf = sum(confs) / max(len(confs), 1)
        if mean_conf < best_val:
            best_val = mean_conf
            torch.save({'model_state': net.state_dict()}, os.path.join(data_root, save_path))
        tqdm.write(f'Epoch {epoch}: val_max_conf={mean_conf:.4f}')

    return {'val_max_conf': best_val}


if __name__ == '__main__':
    data_root = os.path.dirname(os.path.abspath(__file__))
    train_obfuscator(data_root)
