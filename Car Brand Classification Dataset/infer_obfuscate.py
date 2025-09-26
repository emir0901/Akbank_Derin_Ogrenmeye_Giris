import os
import argparse

import torch
from PIL import Image
from torchvision import transforms

from train_obfuscator import SimpleUNet


def load_obfuscator(path: str, device: torch.device) -> SimpleUNet:
    net = SimpleUNet(base=32).to(device)
    state = torch.load(path, map_location=device)
    net.load_state_dict(state['model_state'])
    net.eval()
    return net


def process_image(img: Image.Image, net: SimpleUNet, device: torch.device) -> Image.Image:
    to_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        delta = net(x)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        x_unnorm = x * std + mean
        x_tilde = torch.clamp(x_unnorm + 0.1 * delta, 0.0, 1.0)
    to_pil = transforms.ToPILImage()
    return to_pil(x_tilde.squeeze(0).cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Girdi resim dosyasi veya klasor')
    parser.add_argument('--output', required=True, help='Cikti klasoru')
    parser.add_argument('--model', default='obfuscator.pt', help='Egitilmis obfuscator yolu')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    net = load_obfuscator(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model), device)

    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        for name in os.listdir(args.input):
            in_path = os.path.join(args.input, name)
            if not os.path.isfile(in_path):
                continue
            try:
                img = Image.open(in_path).convert('RGB')
            except Exception:
                continue
            out = process_image(img, net, device)
            out.save(os.path.join(args.output, name))
    else:
        img = Image.open(args.input).convert('RGB')
        out = process_image(img, net, device)
        base = os.path.splitext(os.path.basename(args.input))[0]
        out.save(os.path.join(args.output, f'{base}_obf.jpg'))


if __name__ == '__main__':
    main()
