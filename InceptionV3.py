import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# --------------------------
# 1. SİSTEM AYARLARI
# --------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow loglarını azalt
tf.config.threading.set_intra_op_parallelism_threads(4)  # Çok çekirdek desteği
tf.config.threading.set_inter_op_parallelism_threads(4)#tek işlemde 4 iş parcacığı kullansın istiyor

# --------------------------
# 2. VERİ HAZIRLIK
# --------------------------
# Parametreler
IMG_SIZE = (299, 299)  # InceptionV3 için ideal
BATCH_SIZE = 8  # CPU için güvenli boyut
NUM_CLASSES = 400  # Sınıf sayısı

# Veri artırma (CPU dostu)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,#piksel değerleri 1 0 arası normalize et krımızı mavi yeşil gör .daha hızlı öğrenmeyi sağlar
    rotation_range=20,  # Döndür
    width_shift_range=0.1,#yatay kaydır
    height_shift_range=0.1,#dik kaydır
    shear_range=0.1,#kesme
    zoom_range=0.1,
    horizontal_flip=True , #aynalama,
    fill_mode='reflect' #boş pikselleri doldurma
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

# Veri yükleyiciler (flow_from_directory)
train_dir = 'vangogh/train'
val_dir = 'vangogh/val'

#eğitim verilerini okur ve eğitim sürecinde kullanılmak üzere modelin girdi olarak alacağı verileri hazırlar.
train_generator = train_datagen.flow_from_directory(#veri setinin klasörden yüklemek için
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',#etiketlerin nasıl işlenecek cok sınıf için tek doğrı sınıf
    shuffle=True,#epoch başına eğtim verilerini karıitır
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Sınıf dağılımını kontrol
print("\nSınıf dağılımı:")
print(np.bincount(train_generator.classes))


# --------------------------
# 3. MODEL MİMARİSİ
# --------------------------
def build_cpu_friendly_model():
    base_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling='avg'
    )

    # Katman dondurma
    for layer in base_model.layers[:100]:
        layer.trainable = False#modelde yalnızxa son katmanları eğitir

    # Özelleştirilmiş başlık
    x = base_model.output #son katmanı aldık modelin
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)#her katman 512 nöronlu,overfittingi engelliyor,relu ile negatif değerleri 0 lıyor  daha fazla öğrenmek için
    x = BatchNormalization()(x)#öğrenmeyi hızlandırmk için normalize eder  varyansı 1 ortalamayı 0
    x = Dropout(0.3)(x)  # Daha düşük dropout
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


model = build_cpu_friendly_model()

# --------------------------
# 4. EĞİTİM SÜRECİ (Güvenli mod)
# --------------------------
# Optimizer ( için düşük LR)
optimizer = Adam(learning_rate=0.0001)

# Kayıp fonksiyonu
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)#Categorical Crossentropy kaybı, modelin tahmin ettiği sınıf olasılıklarının gerçek
                                                                 # etiketlerle ne kadar uyumsuz olduğunu ölçer

# Callback'ler (Kesintilere karşı korumalı)
callbacks = [
    ModelCheckpoint(
        'best_model_cpu.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,#10 epoch boyunca
        restore_best_weights=True,
        min_delta=0.001
    ),
    ReduceLROnPlateau(# vall iyleşmediğinde  daha küçük adımlarla iyleştirmk için lr kucultuyoruz
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

# Model derleme
model.compile(# eğitim sırasında kullanılan optimizayson , kayıp fonk ve doğruluk metriklerini belirler.
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)]#acurrsy hesaplanacak
)

# --------------------------
# 5. GÜVENLİ EĞİTİM (try-except bloğu)
# --------------------------
try:
    print("\nEğitim başlıyor...")
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        validation_data=val_generator,
        validation_steps=max(1, val_generator.samples // BATCH_SIZE),
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    print("\nSon değerlendirme (yüzde olarak):")
    loss, acc, top5 = model.evaluate(val_generator, verbose=1)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Top-1 Accuracy : {acc * 100:.2f}%")
    print(f"Validation Top-5 Accuracy : {top5 * 100:.2f}%")
    # Başarılı eğitim sonrası
    model.save('vangogh_fsiz.h5')
    print("Eğitim başarıyla tamamlandı!")

except KeyboardInterrupt:
    print("\nEğitim kullanıcı tarafından durduruldu. Son model kaydediliyor...")
    model.save('interrupted_cpu_model.h5')
except Exception as e:
    print(f"\nBeklenmeyen hata: {str(e)}")
    model.save('recovery_cpu_model.h5')
finally:
    print("Kaynaklar temizleniyor...")
    tf.keras.backend.clear_session()

# --------------------------
# 6. DEĞERLENDİRME (Opsiyonel)
# --------------------------
if 'history' in locals():
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()

    plt.savefig('training_history_cpu.png')
    plt.show()

    """
    C:\Users\aysen\PycharmProjects\flask_project\venv\Scripts\python.exe C:\Users\aysen\PycharmProjects\flask_project\InceptionV3.py 
2025-05-06 21:39:37.188806: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-05-06 21:39:39.184300: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Found 4800 images belonging to 400 classes.
Found 1200 images belonging to 400 classes.

Sınıf dağılımı:
[12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12]
2025-05-06 21:39:44.234430: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

Eğitim başlıyor...
Epoch 1/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.1435 - loss: 13.1206 - top_k_categorical_accuracy: 0.2410WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
600/600 ━━━━━━━━━━━━━━━━━━━━ 697s 1s/step - accuracy: 0.1438 - loss: 13.1182 - top_k_categorical_accuracy: 0.2414 - val_accuracy: 0.9808 - val_loss: 7.8047 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 2/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 0s 986ms/step - accuracy: 0.8416 - loss: 8.3136 - top_k_categorical_accuracy: 0.9152WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
600/600 ━━━━━━━━━━━━━━━━━━━━ 659s 1s/step - accuracy: 0.8416 - loss: 8.3127 - top_k_categorical_accuracy: 0.9152 - val_accuracy: 0.9958 - val_loss: 6.3173 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 3/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 0s 987ms/step - accuracy: 0.9156 - loss: 6.3031 - top_k_categorical_accuracy: 0.9463WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
600/600 ━━━━━━━━━━━━━━━━━━━━ 659s 1s/step - accuracy: 0.9156 - loss: 6.3025 - top_k_categorical_accuracy: 0.9463 - val_accuracy: 0.9975 - val_loss: 5.0668 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 4/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 0s 976ms/step - accuracy: 0.9330 - loss: 4.8982 - top_k_categorical_accuracy: 0.9659WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
600/600 ━━━━━━━━━━━━━━━━━━━━ 653s 1s/step - accuracy: 0.9330 - loss: 4.8978 - top_k_categorical_accuracy: 0.9659 - val_accuracy: 1.0000 - val_loss: 4.0106 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 5/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 653s 1s/step - accuracy: 0.9576 - loss: 3.8107 - top_k_categorical_accuracy: 0.9846 - val_accuracy: 1.0000 - val_loss: 3.2195 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 6/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 668s 1s/step - accuracy: 0.9700 - loss: 3.0103 - top_k_categorical_accuracy: 0.9893 - val_accuracy: 0.9992 - val_loss: 2.6945 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 7/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 648s 1s/step - accuracy: 0.9797 - loss: 2.4780 - top_k_categorical_accuracy: 0.9947 - val_accuracy: 1.0000 - val_loss: 2.3011 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 8/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 878s 1s/step - accuracy: 0.9840 - loss: 2.0967 - top_k_categorical_accuracy: 0.9969 - val_accuracy: 0.9983 - val_loss: 2.0609 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 9/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 929s 2s/step - accuracy: 0.9895 - loss: 1.8333 - top_k_categorical_accuracy: 0.9972 - val_accuracy: 1.0000 - val_loss: 1.9785 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 10/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 907s 2s/step - accuracy: 0.9900 - loss: 1.6729 - top_k_categorical_accuracy: 0.9987 - val_accuracy: 1.0000 - val_loss: 1.7802 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 11/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 667s 1s/step - accuracy: 0.9935 - loss: 1.5514 - top_k_categorical_accuracy: 0.9986 - val_accuracy: 0.9992 - val_loss: 1.6973 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 12/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 656s 1s/step - accuracy: 0.9913 - loss: 1.4735 - top_k_categorical_accuracy: 0.9985 - val_accuracy: 1.0000 - val_loss: 1.6555 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 13/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 762s 1s/step - accuracy: 0.9919 - loss: 1.4128 - top_k_categorical_accuracy: 0.9985 - val_accuracy: 1.0000 - val_loss: 1.6255 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 14/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 647s 1s/step - accuracy: 0.9955 - loss: 1.3665 - top_k_categorical_accuracy: 0.9997 - val_accuracy: 1.0000 - val_loss: 1.6054 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 15/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 639s 1s/step - accuracy: 0.9981 - loss: 1.3241 - top_k_categorical_accuracy: 0.9997 - val_accuracy: 1.0000 - val_loss: 1.6127 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 16/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 641s 1s/step - accuracy: 0.9943 - loss: 1.3049 - top_k_categorical_accuracy: 0.9995 - val_accuracy: 1.0000 - val_loss: 1.5506 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 17/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 639s 1s/step - accuracy: 0.9968 - loss: 1.2816 - top_k_categorical_accuracy: 0.9999 - val_accuracy: 0.9983 - val_loss: 1.4777 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 18/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 640s 1s/step - accuracy: 0.9971 - loss: 1.2708 - top_k_categorical_accuracy: 0.9998 - val_accuracy: 1.0000 - val_loss: 1.4933 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 19/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 638s 1s/step - accuracy: 0.9975 - loss: 1.2541 - top_k_categorical_accuracy: 0.9996 - val_accuracy: 1.0000 - val_loss: 1.5388 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04
Epoch 20/20
600/600 ━━━━━━━━━━━━━━━━━━━━ 639s 1s/step - accuracy: 0.9976 - loss: 1.2362 - top_k_categorical_accuracy: 0.9999 - val_accuracy: 0.9983 - val_loss: 1.5166 - val_top_k_categorical_accuracy: 1.0000 - learning_rate: 1.0000e-04

Son değerlendirme (yüzde olarak):
150/150 ━━━━━━━━━━━━━━━━━━━━ 66s 438ms/step - accuracy: 0.9994 - loss: 1.4746 - top_k_categorical_accuracy: 1.0000
Validation Loss: 1.4777
Validation Top-1 Accuracy : 99.83%
Validation Top-5 Accuracy : 100.00%
    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
Eğitim başarıyla tamamlandı!
Kaynaklar temizleniyor...
WARNING:tensorflow:From C:\Users\aysen\PycharmProjects\flask_project\venv\Lib\site-packages\keras\src\backend\common\global_state.py:82: The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.

WARNING:tensorflow:From C:\Users\aysen\PycharmProjects\flask_project\venv\Lib\site-packages\keras\src\backend\common\global_state.py:82: The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.
"""