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
2. Kaggle API ayarlarını yap
3. Veri setini indir
4. Veri setini hazırla
5. Modelleri oluştur
6. Modeli eğit
7. Modeli değerlendir

Eğitilen model `final_model_Baseline_CNN.h5` olarak kaydedilecektir.

### 4. Modeli Yerel Kullanım İçin Hazırlama

Eğitilen modeli `models/` klasörüne kopyalayın:

```bash
mkdir -p models
# Colab'dan indirdiğiniz modeli buraya kopyalayın
cp /path/to/final_model_Baseline_CNN.h5 models/
```

## 🖥️ Web Arayüzünü Çalıştırma

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak açılacaktır (genellikle `http://localhost:8501`)

## 📊 Kullanım

1. **Model Eğitimi (Colab):**
   - `Untitled5.ipynb` dosyasını Google Colab'da açın
   - Kaggle API token'ınızı yükleyin (Cell 2)
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

## 📝 Notebook İçeriği

Notebook şu adımları içerir:
- Kütüphane yükleme ve ortam hazırlama
- Kaggle API kurulumu ve veri seti indirme
- Veri seti hazırlama ve görselleştirme
- 4 farklı model mimarisi oluşturma
- Model eğitimi ve callback'ler
- Model değerlendirme ve görselleştirme
- Model kaydetme ve indirme
- Tahmin fonksiyonu ve kullanım örnekleri

## ⚠️ Önemli Notlar

- Bu uygulama **eğitim amaçlıdır** ve tıbbi tavsiye yerine geçmez
- En iyi sonuçlar için net, iyi aydınlatılmış cilt fotoğrafları kullanın
- Model eğitimi GPU kullanımı ile daha hızlı olacaktır
- Colab'da GPU'yu Runtime > Change runtime type > GPU seçerek aktif edin

## 🔗 Bağlantılar

- [Kaggle Dataset](https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset)
- [Colab Notebook](https://colab.research.google.com/github/Aleynaozm/ysa_skin_type/blob/main/Untitled5.ipynb)

## 📝 Lisans

Bu proje eğitim amaçlıdır.

## 👨‍💻 Geliştirici

Cilt tipi analizi projesi - Yapay Zeka ve Derin Öğrenme
