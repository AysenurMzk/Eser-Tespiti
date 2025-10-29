import os
import json
import uuid
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import time

# Global değişkenlerin tanımlanması
_MODEL = None
_CLASS_LABELS = None

def load_trained_model(model_path=None, classes_path=None):
    """Eğitilmiş modeli ve sınıf etiketlerini yükler"""
    global _MODEL, _CLASS_LABELS

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Varsayılan dosya yolları
    model_path = model_path or os.path.join(base_dir, 'vangogh_fsiz.h5')
    classes_path = classes_path or os.path.join(base_dir, 'class_indices.json')

    # Dosya kontrolleri
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Sınıf etiketleri bulunamadı: {classes_path}")

    # Modeli yükle
    _MODEL = load_model(model_path)

    # Sınıf etiketlerini yükle
    with open(classes_path, 'r') as f:
        class_indices = json.load(f)
    _CLASS_LABELS = {v: k for k, v in class_indices.items()}

    return _MODEL, _CLASS_LABELS


def preprocess_image(img_path, target_size=(299, 299)):
    """Görseli modele uygun şekilde işler"""
    # OpenCV ile oku
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {img_path}")

    # BGR -> RGB dönüşümü
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Yeniden boyutlandırma ve normalizasyon
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def start_camera(camera_index=0):
    """Kamerayı başlatır (DirectShow ile)"""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # DirectShow kullanarak başlat
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {camera_index} açılamadı. DirectShow ile bağlantı kurulamadı.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Çözünürlük ayarı
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)  # FPS ayarı

    return cap


def save_frame(camera, save_dir, timeout=30):  # timeout süresini 30 saniyeye çıkardık
    """Güçlendirilmiş kare kaydetme fonksiyonu"""
    start_time = time.time()

    while time.time() - start_time < timeout:
        ret, frame = camera.read()
        if ret:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"capture_{int(time.time())}.jpg"
            path = os.path.join(save_dir, filename)

            if cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                return path
        time.sleep(0.1)

    raise RuntimeError("Timeout: Görüntü alınamadı")


def predict_image(img_path):
    """Görsel için tahmin yapar"""
    global _MODEL, _CLASS_LABELS

    if _MODEL is None or _CLASS_LABELS is None:
        raise RuntimeError("Model yüklenmemiş. Önce load_trained_model() çağrılmalı.")

    # Modelin beklediği boyutları al
    input_shape = _MODEL.input_shape[1:3]

    # Ön işleme ve tahmin
    processed_img = preprocess_image(img_path, target_size=input_shape)
    predictions = _MODEL.predict(processed_img)
    predicted_class = _CLASS_LABELS[np.argmax(predictions[0])]

    return predicted_class


def predict_from_camera(camera, save_dir='uploads'):
    """
    Kameradan bir kare alır, kaydeder ve tahmin yapar.
    """
    # Kameradan bir kare al
    image_path = save_frame(camera, save_dir)

    # Tahmin yapæ
    prediction = predict_image(image_path)
    return prediction


# Eğer class_indices.json dosyanız yoksa, şu komutla dosyayı oluşturabilirsiniz:
# create_class_indices_json()

import cv2
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Kamera {i} çalışıyor")
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"kamera_test_{i}.jpg", frame)
        cap.release()
    else:
        print(f"Kamera {i} çalışmıyor")
