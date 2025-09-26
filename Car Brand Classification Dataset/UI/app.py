import os
import io
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from train_obfuscator import SimpleUNet


@st.cache_resource
def load_model(model_path: str, device: torch.device) -> SimpleUNet:
    net = SimpleUNet(base=32).to(device)
    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state['model_state'])
    net.eval()
    return net


def obfuscate_image(img: Image.Image, net: SimpleUNet, device: torch.device) -> Image.Image:
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
    st.set_page_config(page_title="Car Brand Obfuscation", page_icon="🚗", layout="wide")
    st.title("Araba Marka Bilgisi Gizleme (Obfuscation)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    default_model = str(Path(__file__).resolve().parents[1] / 'obfuscator.pt')
    model_path = st.text_input("Model yolu", value=default_model)

    if not os.path.isfile(model_path):
        st.info("Model dosyası bulunamadı. Lütfen önce `train_obfuscator.py` ile modeli eğitin.")
        return

    net = load_model(model_path, device)

    tab1, tab2 = st.tabs(["Tek Görsel", "Klasör"])

    with tab1:
        file = st.file_uploader("Bir görsel yükleyin", type=["jpg", "jpeg", "png"]) 
        if file is not None:
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            out = obfuscate_image(img, net, device)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Orijinal")
                st.image(img, use_container_width=True)
            with col2:
                st.subheader("Obfuscate")
                st.image(out, use_container_width=True)
            buf = io.BytesIO()
            out.save(buf, format='JPEG')
            st.download_button("İndir", buf.getvalue(), file_name="obf.jpg", mime="image/jpeg")

    with tab2:
        folder = st.text_input("Klasör yolu", value=str(Path.cwd() / 'test'))
        if st.button("Klasörü İşle"):
            if not os.path.isdir(folder):
                st.error("Geçerli bir klasör girin")
            else:
                images = []
                for name in os.listdir(folder):
                    in_path = os.path.join(folder, name)
                    if not os.path.isfile(in_path):
                        continue
                    try:
                        img = Image.open(in_path).convert('RGB')
                    except Exception:
                        continue
                    out = obfuscate_image(img, net, device)
                    images.append((name, img, out))
                for name, img, out in images:
                    st.write(name)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img, use_container_width=True)
                    with col2:
                        st.image(out, use_container_width=True)


if __name__ == "__main__":
    main()
