# -*- coding: utf-8 -*-

"""Cilt Tipi Analizi - Derin Öğrenme Projesi (Final ve Karşılaştırmalı Çalışma)

Bu kod, dört farklı derin öğrenme modelini (Baseline CNN, VGG16, MobileNetV2, ResNet50) eğitip
performanslarını karşılaştırarak en iyi modeli seçmeyi hedefler.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 CİLT TİPİ ANALİZİ PROJESİ BAŞLATILIYOR")
print("="*80)
print("\n📦 Gerekli kütüphaneler yükleniyor...")

print(f"✅ TensorFlow versiyonu: {tf.__version__}")

np.random.seed(42)
tf.random.set_seed(42)
print("✅ Rastgelelik tohumları (seed) ayarlandı.")

print("\n📥 Kaggle veri seti indirilmesi için ortam hazırlanıyor...")

os.system("pip install -q kaggle")
os.system("mkdir -p ~/.kaggle")

print("\n⚠️ DİKKAT: Kaggle API kimlik doğrulaması için 'kaggle.json' dosyasını yüklemelisiniz.")

KAGGLE_DATASET = 'shakyadissanayake/oily-dry-and-normal-skin-types-dataset'
print(f"\n📥 Veri seti indiriliyor: {KAGGLE_DATASET}")
os.system(f"kaggle datasets download -d {KAGGLE_DATASET}")

print("📦 Veri seti açılıyor...")
BASE_DIR = '/content/skin_dataset'

if os.path.exists(BASE_DIR):
    shutil.rmtree(BASE_DIR)

os.system(f"unzip -qo oily-dry-and-normal-skin-types-dataset.zip -d {BASE_DIR}")
print("✅ Veri seti hazırlandı!")

BASE_DIR_PRE_SPLIT = f'{BASE_DIR}/Oily-Dry-Skin-Types'
TRAIN_DIR = f'{BASE_DIR_PRE_SPLIT}/train'
VAL_DIR = f'{BASE_DIR_PRE_SPLIT}/valid'
TEST_DIR = f'{BASE_DIR_PRE_SPLIT}/test'

print("\n📊 Veri seti DAĞILIMI KONTROLÜ:")
all_data = []
try:
    class_names_found = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
except FileNotFoundError:
    class_names_found = []

for split, path in [('Train', TRAIN_DIR), ('Validation', VAL_DIR), ('Test', TEST_DIR)]:
    if os.path.exists(path):
        total = 0
        row_data = {'Split': split}
        for class_name in class_names_found:
            class_path = os.path.join(path, class_name)
            count = len(os.listdir(class_path)) if os.path.isdir(class_path) else 0
            total += count
            row_data[class_name.capitalize()] = count
        row_data['Total'] = total
        all_data.append(row_data)

distribution_df = pd.DataFrame(all_data)
if not distribution_df.empty:
    print(distribution_df.to_string(index=False))

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 3
CLASS_NAMES = ['dry', 'normal', 'oily']
LEARNING_RATE = 0.0001

print(f"\n⚙️ Hiperparametreler: IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}, LR={LEARNING_RATE}")

print("\n🎨 Data Augmentation ve Jeneratörler yapılandırılıyor...")
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.15, zoom_range=0.15, horizontal_flip=True, fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

try:
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True, seed=42
    )
    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )
    print("✅ Jeneratörler hazır.")
except Exception as e:
    print(f"❌ Jeneratör hatası: {e}")

def create_baseline_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        layers.BatchNormalization(), layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(256, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def create_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False
    model = models.Sequential([
        base_model, layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(256, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def create_mobilenet_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
    model = models.Sequential([
        base_model, layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.4),
        layers.Dense(128, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def create_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    model = models.Sequential([
        base_model, layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

print("\n" + "="*80)
print("🤖 MODELLER OLUŞTURULUYOR VE DERLENİYOR")
print("="*80)

models_dict = {
    'Baseline_CNN': create_baseline_cnn(),
    'VGG16': create_vgg16_model(),
    'MobileNetV2': create_mobilenet_model(),
    'ResNet50': create_resnet_model()
}

for name, model in models_dict.items():
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"✅ Model {name} başarıyla tanımlandı ve derlendi.")

print("\n⚙️ CALLBACK'LER AYARLANIYOR")
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

checkpoint_dir = '/content/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

print("✅ Callback'ler hazır: Early Stopping (patience=10), ReduceLROnPlateau (patience=5)")

print("\n" + "="*80)
print("🚀 MODELLERİN EĞİTİMİ VE DEĞERLENDİRİLMESİ BAŞLIYOR")
print("="*80)
print("\n⚠️  Bu adım, dört modeli de sırayla eğiteceği için uzun sürecektir.")

results_df = pd.DataFrame(columns=['Model', 'Test_Accuracy', 'Test_Loss'])
global_history = {}

for MODEL_NAME, model_to_train in models_dict.items():
    print(f"\n--- 🎯 {MODEL_NAME} EĞİTİMİ BAŞLADI ---")
    
    checkpoint = ModelCheckpoint(
        f'{checkpoint_dir}/best_model_{MODEL_NAME}.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    callbacks = [early_stopping, reduce_lr, checkpoint]
    
    try:
        history = model_to_train.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_generator,
            validation_steps=val_generator.samples // BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        global_history[MODEL_NAME] = history
        
        model_to_train.save(f'/content/final_model_{MODEL_NAME}.h5')
        print(f"✅ {MODEL_NAME} eğitimi tamamlandı ve kaydedildi.")
    except Exception as e:
        print(f"❌ {MODEL_NAME} eğitimi sırasında hata oluştu: {e}")
        continue
    
    print(f"\n--- 📊 {MODEL_NAME} DEĞERLENDİRME ---")
    test_generator.reset()
    
    test_loss, test_accuracy = model_to_train.evaluate(test_generator, verbose=1)
    new_row = pd.DataFrame({
        'Model': [MODEL_NAME],
        'Test_Accuracy': [f"{test_accuracy:.4f}"],
        'Test_Loss': [f"{test_loss:.4f}"]
    })
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    y_pred = model_to_train.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + 1)
    y_pred_classes = np.argmax(y_pred, axis=1)[:len(test_generator.classes)]
    y_true = test_generator.classes
    
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix - {MODEL_NAME}')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.show()

print("\n" + "="*80)
print("🏆 TÜM MODELLERİN PERFORMANS ÖZETİ")
print("="*80)

results_df['Test_Accuracy'] = results_df['Test_Accuracy'].astype(float)
final_results = results_df.sort_values(by='Test_Accuracy', ascending=False)
print(final_results.to_string(index=False))

best_model_name = final_results.iloc[0]['Model']
best_accuracy = final_results.iloc[0]['Test_Accuracy']
best_model = models_dict[best_model_name]

best_model.save('/content/final_model_Best.h5')
print(f"\n--- 🎉 SEÇİLEN EN İYİ MODEL: {best_model_name} (Test Doğruluğu: {best_accuracy:.4f}) ---")
print(f"✅ En iyi model kaydedildi: /content/final_model_Best.h5")

if best_model_name in global_history:
    history = global_history[best_model_name]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title(f'{best_model_name} - Model Doğruluğu', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title(f'{best_model_name} - Model Kaybı', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\n" + "="*80)
print("🖼️ TEK GÖRÜNTÜ TAHMİN FONKSİYONU")
print("="*80)

def predict_skin_type(image_path, model):
    if not os.path.exists(image_path):
        return {"error": f"Hata: Görüntü dosyası bulunamadı: {image_path}"}
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        return {"error": f"Görüntü işlenirken hata: {e}"}
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_idx].capitalize()
    confidence = predictions[0][predicted_class_idx] * 100
    
    probabilities = {
        CLASS_NAMES[i].capitalize(): float(predictions[0][i] * 100)
        for i in range(len(CLASS_NAMES))
    }
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities
    }

print("✅ Tahmin fonksiyonu hazır!")
print(f"\n💡 Kullanım örneği: predict_skin_type('path/to/image.jpg', best_model)")

