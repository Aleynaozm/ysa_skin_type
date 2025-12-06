# 🧴 Cilt Tipi Analizi Projesi

Bu proje, yapay zeka kullanarak cilt tiplerini (Kuru, Normal, Yağlı) analiz eden bir uygulamadır.

## 📋 Özellikler

- **4 Farklı Model:** Baseline CNN, VGG16, MobileNetV2, ResNet50
- **Veri Artırma:** Overfitting'i önlemek için data augmentation
- **Web Arayüzü:** Streamlit ile kullanıcı dostu arayüz
- **Detaylı Analiz:** Confusion matrix, classification report ve görselleştirmeler

## 🚀 Kurulum

### 1. Gereksinimler

```bash
pip install -r requirements.txt
```

### 2. Veri Seti

Kaggle'dan veri setini indirin:
- Dataset: `shakyadissanayake/oily-dry-and-normal-skin-types-dataset`
- Notebook içinde otomatik indirme kodu mevcuttur

### 3. Model Eğitimi

Google Colab'da `Untitled5.ipynb` dosyasını açın ve hücreleri sırayla çalıştırın:

1. Kütüphaneleri yükle
2. Veri setini indir
3. Veri setini hazırla
4. Modelleri oluştur
5. Modeli eğit
6. Modeli değerlendir

Eğitilen model `final_model_Baseline_CNN.h5` olarak kaydedilecektir.

### 4. Modeli Yerel Kullanım İçin Hazırlama

Eğitilen modeli `models/` klasörüne kopyalayın:

```bash
mkdir -p models
# Colab'dan indirdiğiniz modeli buraya kopyalayın
cp /content/final_model_Baseline_CNN.h5 models/
```

## 🖥️ Web Arayüzünü Çalıştırma

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak açılacaktır (genellikle `http://localhost:8501`)

## 📊 Kullanım

1. **Model Eğitimi (Colab):**
   - `Untitled5.ipynb` dosyasını Google Colab'da açın
   - Tüm hücreleri sırayla çalıştırın
   - Eğitilen modeli indirin

2. **Web Arayüzü:**
   - `streamlit run app.py` komutu ile uygulamayı başlatın
   - Cilt fotoğrafınızı yükleyin
   - "Analiz Et" butonuna tıklayın
   - Sonuçları görüntüleyin

## 📁 Proje Yapısı

```
ysa_skin_type-main/
├── Untitled5.ipynb          # Model eğitimi notebook'u
├── app.py                   # Streamlit web arayüzü
├── requirements.txt         # Python gereksinimleri
├── README.md               # Bu dosya
└── models/                  # Eğitilmiş modeller (oluşturulacak)
    └── final_model_Baseline_CNN.h5
```

## 🎯 Model Performansı

- **Baseline CNN:** Temel CNN mimarisi
- **VGG16:** Transfer learning ile feature extraction
- **MobileNetV2:** Fine-tuning ile optimize edilmiş
- **ResNet50:** Deep residual network

## ⚠️ Önemli Notlar

- Bu uygulama **eğitim amaçlıdır** ve tıbbi tavsiye yerine geçmez
- En iyi sonuçlar için net, iyi aydınlatılmış cilt fotoğrafları kullanın
- Model eğitimi GPU kullanımı ile daha hızlı olacaktır

## 📝 Lisans

Bu proje eğitim amaçlıdır.

## 👨‍💻 Geliştirici

Cilt tipi analizi projesi - Yapay Zeka ve Derin Öğrenme
