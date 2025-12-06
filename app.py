"""
Cilt Tipi Analizi Web Arayüzü
Streamlit ile basit ve kullanıcı dostu arayüz
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Sayfa ayarları
st.set_page_config(
    page_title="Cilt Tipi Analizi",
    page_icon="🧴",
    layout="wide"
)

# Model ve sınıf isimleri
IMG_SIZE = (224, 224)
CLASS_NAMES = ['dry', 'normal', 'oily']

# Model yolu - önce models klasöründe, yoksa root'ta ara
MODEL_PATH = None
possible_paths = [
    'models/final_model_Baseline_CNN.h5',
    'final_model_Baseline_CNN.h5',
    'best_model_Baseline_CNN.h5',
    'models/best_model_Baseline_CNN.h5'
]

for path in possible_paths:
    if os.path.exists(path):
        MODEL_PATH = path
        break

# Türkçe sınıf isimleri
CLASS_NAMES_TR = {
    'dry': 'Kuru Cilt',
    'normal': 'Normal Cilt',
    'oily': 'Yağlı Cilt'
}

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 5px;
        display: flex;
        align-items: center;
        padding: 0 10px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_skin_model():
    """Modeli yükle (cache ile)"""
    try:
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            return model
        else:
            # Model bulunamadıysa hata göster ama uygulamayı durdurma
            return None
    except Exception as e:
        st.error(f"❌ Model yüklenirken hata: {str(e)}")
        return None

def preprocess_image(image):
    """Görüntüyü model için hazırla"""
    # RGB'ye çevir
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Boyutlandır
    image = image.resize(IMG_SIZE)
    
    # Array'e çevir ve normalize et
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_skin_type(image, model):
    """Cilt tipi tahmini yap"""
    # Görüntüyü ön işle
    img_array = preprocess_image(image)
    
    # Tahmin yap
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Tüm sınıflar için olasılıklar
    probabilities = {
        CLASS_NAMES[i]: float(predictions[0][i] * 100) 
        for i in range(len(CLASS_NAMES))
    }
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities
    }

# Ana başlık
st.markdown('<h1 class="main-header">🧴 Cilt Tipi Analizi</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Bilgi")
    st.markdown("""
    Bu uygulama, yapay zeka kullanarak cilt tipinizi analiz eder.
    
    **Desteklenen Cilt Tipleri:**
    - 🏜️ Kuru Cilt (Dry)
    - 🌸 Normal Cilt (Normal)
    - 💧 Yağlı Cilt (Oily)
    
    **Kullanım:**
    1. Cilt fotoğrafınızı yükleyin
    2. Analiz butonuna tıklayın
    3. Sonuçları görüntüleyin
    
    **Not:** En iyi sonuçlar için net, iyi aydınlatılmış cilt fotoğrafları kullanın.
    """)
    
    st.markdown("---")
    st.markdown("**Model:** Baseline CNN")
    st.markdown("**Görüntü Boyutu:** 224x224")

# Model yükleme
model = load_skin_model()

if model is None:
    st.error("❌ Model yüklenemedi!")
    st.info("""
    💡 **Model dosyası bulunamadı. Lütfen:**
    1. Modeli eğitin (Untitled5.ipynb dosyasını Colab'da çalıştırın)
    2. Eğitilen modeli `models/` klasörüne kopyalayın
    3. Model dosya adı: `final_model_Baseline_CNN.h5` veya `best_model_Baseline_CNN.h5`
    
    **Aranan konumlar:**
    - `models/final_model_Baseline_CNN.h5`
    - `final_model_Baseline_CNN.h5`
    - `models/best_model_Baseline_CNN.h5`
    - `best_model_Baseline_CNN.h5`
    """)
    st.stop()
else:
    st.success(f"✅ Model yüklendi: {MODEL_PATH}")

# Ana içerik
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Görüntü Yükle")
    
    # Dosya yükleme
    uploaded_file = st.file_uploader(
        "Cilt fotoğrafınızı seçin",
        type=['jpg', 'jpeg', 'png'],
        help="JPG, JPEG veya PNG formatında görüntü yükleyebilirsiniz"
    )
    
    if uploaded_file is not None:
        # Görüntüyü göster
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Görüntü", use_container_width=True)
        
        # Tahmin butonu
        if st.button("🔍 Analiz Et", type="primary", use_container_width=True):
            with st.spinner("🔄 Analiz yapılıyor..."):
                # Tahmin yap
                result = predict_skin_type(image, model)
                
                # Sonuçları session state'e kaydet
                st.session_state['prediction_result'] = result
                st.session_state['uploaded_image'] = image

with col2:
    st.header("📊 Analiz Sonuçları")
    
    if 'prediction_result' in st.session_state:
        result = st.session_state['prediction_result']
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        probabilities = result['probabilities']
        
        # Tahmin kutusu
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        
        # Ana tahmin
        st.markdown(f"### 🎯 Tahmin: **{CLASS_NAMES_TR[predicted_class]}**")
        st.markdown(f"### 📈 Güven: **{confidence:.2f}%**")
        
        # İlerleme çubuğu
        st.progress(confidence / 100)
        
        st.markdown("---")
        st.markdown("### 📊 Tüm Olasılıklar:")
        
        # Her sınıf için olasılık göster
        for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            class_tr = CLASS_NAMES_TR[class_name]
            is_predicted = (class_name == predicted_class)
            
            # Renk belirleme
            if is_predicted:
                color = "#1f77b4"
                icon = "✅"
            else:
                color = "#d3d3d3"
                icon = "  "
            
            st.markdown(f"{icon} **{class_tr}**: {prob:.2f}%")
            st.progress(prob / 100)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Öneriler
        st.markdown("---")
        st.markdown("### 💡 Öneriler:")
        
        if predicted_class == 'dry':
            st.info("""
            **Kuru Cilt İçin Öneriler:**
            - Nemlendirici kremler kullanın
            - Yumuşak temizleyiciler tercih edin
            - Güneş koruyucu kullanmayı unutmayın
            - Sıcak su yerine ılık su kullanın
            """)
        elif predicted_class == 'normal':
            st.success("""
            **Normal Cilt İçin Öneriler:**
            - Dengeli bir cilt bakım rutini uygulayın
            - Hafif nemlendiriciler kullanın
            - Düzenli temizlik yapın
            - Güneş koruyucu kullanın
            """)
        elif predicted_class == 'oily':
            st.warning("""
            **Yağlı Cilt İçin Öneriler:**
            - Yağsız, non-comedogenic ürünler kullanın
            - Düzenli temizlik yapın
            - Toner kullanmayı düşünün
            - Aşırı temizlikten kaçının
            """)
    else:
        st.info("👈 Sol taraftan bir görüntü yükleyip 'Analiz Et' butonuna tıklayın.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Cilt Tipi Analizi - Yapay Zeka Destekli</p>
        <p>⚠️ Bu uygulama eğitim amaçlıdır. Tıbbi tavsiye yerine geçmez.</p>
    </div>
    """,
    unsafe_allow_html=True
)

