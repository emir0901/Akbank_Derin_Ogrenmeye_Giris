# Araba Marka Bilgisi Gizleme (Obfuscation) Projesi

Bu proje, araba fotoğraflarındaki marka bilgisini makine öğrenmesi ile zor ayırt edilir hale getirmeyi amaçlar. İki aşamalı yaklaşım:
- Önce bir sınıflandırıcı (ResNet18) marka tanımayı öğrenir.
- Ardından bir obfuscator (basit U-Net) görüntüyü az değiştirerek sınıflandırıcının güvenini düşürmeyi (marka bilgisini silmeyi) öğrenir.

## Giriş
Kullanılan veri: klasör başına marka olacak şekilde `train/val/test` dizinleri. Model ve kodlar Python + PyTorch ile yazıldı.

## Kurulum
```bash
cd "Car Brand Classification Dataset"
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Eğitim
- Sınıflandırıcı:
```bash
python train_classifier.py
```
- Obfuscator:
```bash
python train_obfuscator.py
```

## Kullanım
- Komut satırı (tek görsel/klasör):
```bash
python infer_obfuscate.py --input path/to/image_or_folder --output out_dir --model obfuscator.pt
```
- UI (Streamlit):
```bash
streamlit run UI/app.py
```

## Metrikler
- Örnek (eğitim çıktısından): `val_acc ~ 0.73`, `test_acc ~ 0.72` (sınıflandırıcı)
- Obfuscator için izlenen metrik: doğrulama setinde sınıflandırıcı maksimum olasılığı; hedef daha düşük değerler (ör: 0.15 civarı).

## Ekler
- `UI/app.py`: Streamlit arayüzüyle görsel/klasör işleme.
- `supervised.ipynb` ve `unsupervised.ipynb`: çalışmanızı anlatmanız için şablon defterler.

## Sonuç ve Gelecek Çalışmalar
- Daha iyi görsel kalite için SSIM/VGG perceptual loss eklenebilir.
- Daha güçlü gizleme için farklı jeneratif mimariler denenebilir.

## Linkler
- Örnek şablon repo: [gokerguner/example-repo](https://github.com/gokerguner/example-repo)
