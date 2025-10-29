"""import os
import shutil
import random

# Kaynak klasör ve hedef klasör
kaynak_klasor = "van_gogh_resimleri_standart"
hedef_klasor = "van_gogh_resimleri_train_val_split"

# Eğitim ve doğrulama oranları
train_ratio = 0.8
val_ratio = 0.2

# Eğitim ve doğrulama klasörlerini oluştur
train_dir = os.path.join(hedef_klasor, "train")
val_dir = os.path.join(hedef_klasor, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Türlerin bulunduğu alt klasörleri al
for style in os.listdir(kaynak_klasor):
    style_path = os.path.join(kaynak_klasor, style)
    if os.path.isdir(style_path):
        # Her tür için eğitim ve doğrulama klasörlerini oluştur
        os.makedirs(os.path.join(train_dir, style), exist_ok=True)
        os.makedirs(os.path.join(val_dir, style), exist_ok=True)

        # Resimleri al
        images = os.listdir(style_path)
        random.shuffle(images)  # Resimleri karıştır

        # Eğitim ve doğrulama resimlerini ayır
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Resimleri eğitim setine kopyala
        for img in train_images:
            src = os.path.join(style_path, img)
            dst = os.path.join(train_dir, style, img)
            shutil.copy(src, dst)

        # Resimleri doğrulama setine kopyala
        for img in val_images:
            src = os.path.join(style_path, img)
            dst = os.path.join(val_dir, style, img)
            shutil.copy(src, dst)

print("Veri kümesi başarıyla eğitim ve doğrulama setlerine ayrıldı.")"""
"""import os
import shutil

# Kaynak ve hedef klasörler
kaynak_klasor = "van_gogh_resimleri_train_val_split"
hedef_klasor = "van_gogh_resimleri_train_val_split2"

# Eğitim ve doğrulama oranları
train_ratio = 0.8
val_ratio = 0.2

# Yeni hedef klasörleri oluştur
train_dir = os.path.join(hedef_klasor, "train")
val_dir = os.path.join(hedef_klasor, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Türlerin bulunduğu alt klasörleri al
for split in ['train', 'val']:
    for style in os.listdir(os.path.join(kaynak_klasor, split)):
        style_path = os.path.join(kaynak_klasor, split, style)

        if os.path.isdir(style_path):
            # Resimleri al
            images = os.listdir(style_path)

            for img in images:
                # Yeni klasöre uygun bir isimle kopyalama
                img_path = os.path.join(style_path, img)
                if split == 'train':
                    dst_dir = os.path.join(train_dir, img.split('.')[0])  # Sanat eserinin ismi (resmin adı)
                else:
                    dst_dir = os.path.join(val_dir, img.split('.')[0])  # Sanat eserinin ismi (resmin adı)

                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(img_path, os.path.join(dst_dir, img))  # Resmi yeni oluşturulan klasöre kopyala"""

print("Veri kümesi başarıyla eğitim ve doğrulama setlerine ayrıldı ve yeni klasöre kopyalandı.")
"""import os
import shutil
import random

# Kaynak klasör ve hedef klasörler
source_dir = 'van_gogh_resimleri_2'  # Resimlerin bulunduğu klasör
target_dir = 'van_gogh_resimleri_train_val_split2'  # Eğitim ve doğrulama için kullanılacak klasör

# Eğitim ve doğrulama oranları
train_ratio = 0.8 # 1359 dosya 1729 klosor
val_ratio = 0.2 # 369 dosya 1729 klosor

# Eğitim ve doğrulama klasörlerini oluştur
train_dir = os.path.join(target_dir, 'train')
val_dir = os.path.join(target_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Kaynak klasördeki resimleri al
for image_name in os.listdir(source_dir):
    if image_name.endswith(".jpg"):  # Eğer dosya bir resimse (jpg uzantılı)
        # Her resim için klasör adı, resmin adından alınıyor (örneğin: "0001_starry_night.jpg" => "0001_starry_night")
        image_class = image_name.split('.')[0]  # Resim adı, dosya uzantısı hariç
        image_path = os.path.join(source_dir, image_name)

        # Eğitim ve doğrulama klasörleri için her resim için klasör oluşturuluyor
        label_train_dir = os.path.join(train_dir, image_class)
        label_val_dir = os.path.join(val_dir, image_class)

        # Klasörleri oluşturuyoruz
        os.makedirs(label_train_dir, exist_ok=True)
        os.makedirs(label_val_dir, exist_ok=True)

        # Eğitim ve doğrulama verisini ayırma
        if random.random() < train_ratio:
            shutil.copy(image_path, os.path.join(label_train_dir, image_name))  # Resmi eğitim setine kopyala
        else:
            shutil.copy(image_path, os.path.join(label_val_dir, image_name))  # Resmi doğrulama setine kopyala
            """

print("Veri başarıyla eğitim ve doğrulama setlerine ayrıldı.")
"""import os
import shutil
import random

target_dir = 'van_gogh_resimleri_train_val_split2'  # Eğitim ve doğrulama için kullanılacak klasör

# Eğitim ve doğrulama oranları
train_ratio = 0.8
val_ratio = 0.2

# Eğitim ve doğrulama klasörlerini oluştur
train_dir = os.path.join(target_dir, 'train')
val_dir = os.path.join(target_dir, 'val')

# Eğitim setinden %20'sini silme işlemi
train_classes = os.listdir(train_dir)
for train_class in train_classes:
    train_class_dir = os.path.join(train_dir, train_class)
    if not os.path.isdir(train_class_dir):
        continue  # Eğer klasör değilse atla
    train_images = os.listdir(train_class_dir)
    num_train_images = len(train_images)
    num_to_delete_train = int(num_train_images * 0.2)  # %20'sini sil

    # Rastgele seçilen resimleri sil
    images_to_delete_train = random.sample(train_images, num_to_delete_train)
    for image_name in images_to_delete_train:
        image_path = os.path.join(train_class_dir, image_name)

        # Dosyanın gerçekten mevcut olup olmadığını kontrol et
        if os.path.exists(image_path):
            try:
                os.remove(image_path)  # Silme işlemi
                print(f"Silindi: {image_path}")
            except Exception as e:
                print(f"Silinemedi: {image_path}, Hata: {e}")

    # Eğer klasörde başka resim kalmadıysa, klasörü sil
    if len(os.listdir(train_class_dir)) == 0:
        try:
            os.rmdir(train_class_dir)  # Klasörü sil
            print(f"Boş olan klasör silindi: {train_class_dir}")
        except Exception as e:
            print(f"Silinemedi: {train_class_dir}, Hata: {e}")

# Doğrulama setinden %80'ini silme işlemi
val_classes = os.listdir(val_dir)
for val_class in val_classes:
    val_class_dir = os.path.join(val_dir, val_class)
    if not os.path.isdir(val_class_dir):
        continue  # Eğer klasör değilse atla
    val_images = os.listdir(val_class_dir)
    num_val_images = len(val_images)
    num_to_delete_val = int(num_val_images * 0.8)  # %80'ini sil

    # Rastgele seçilen resimleri sil
    images_to_delete_val = random.sample(val_images, num_to_delete_val)
    for image_name in images_to_delete_val:
        image_path = os.path.join(val_class_dir, image_name)

        # Dosyanın gerçekten mevcut olup olmadığını kontrol et
        if os.path.exists(image_path):
            try:
                os.remove(image_path)  # Silme işlemi
                print(f"Silindi: {image_path}")
            except Exception as e:
                print(f"Silinemedi: {image_path}, Hata: {e}")

    # Eğer klasörde başka resim kalmadıysa, klasörü sil
    if len(os.listdir(val_class_dir)) == 0:
        try:
            os.rmdir(val_class_dir)  # Klasörü sil
            print(f"Boş olan klasör silindi: {val_class_dir}")
        except Exception as e:
            print(f"Silinemedi: {val_class_dir}, Hata: {e}")"""

print("Veri başarıyla eğitim ve doğrulama setlerine ayrıldı ve gereksiz veriler silindi.")

"""import os
import shutil

# Kaynak ve hedef dizinleri belirtelim
source_dir = 'van_gogh_resimleri_2/'  # Resimlerin bulunduğu dizin
train_dir = 'van_gogh_resimleri_train_val_split2/train/'  # Eğitim verisi için hedef klasör
val_dir = 'van_gogh_resimleri_train_val_split2/val/'  # Doğrulama verisi için hedef klasör

# Eğitim ve doğrulama setleri için bir oran belirleyelim (örneğin, %80 eğitim, %20 doğrulama)
train_ratio = 0.8

# Tüm resim dosyalarını alalım
image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]

# Resimleri eğitim ve doğrulama setlerine ayıralım
num_train = int(len(image_files) * train_ratio)
train_files = image_files[:num_train]
val_files = image_files[num_train:]

# Hedef dizinleri kontrol et ve oluştur
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# Resimleri ayırarak uygun klasöre taşıyalım
for file in train_files:
    # Dosya ismini eserno_eseradı şeklinde alalım (ilk _ karakterine göre bölüyoruz)
    parts = file.split('_', 1)  # İlk _'yi ayır
    if len(parts) != 2:
        print(f"Dosya ismi beklenen formatta değil: {file}")
        continue

    eser_numarasi, eser_adi = parts
    eser_adi = eser_adi.replace('.jpg', '').strip()  # Uzantıyı kaldır ve boşlukları temizle

    # Hedef klasörde uygun alt klasörleri oluştur
    target_folder = os.path.join(train_dir, eser_adi)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Resmi hedef klasöre taşıyalım
    shutil.move(os.path.join(source_dir, file), os.path.join(target_folder, file))

for file in val_files:
    # Dosya ismini eserno_eseradı şeklinde alalım (ilk _ karakterine göre bölüyoruz)
    parts = file.split('_', 1)  # İlk _'yi ayır
    if len(parts) != 2:
        print(f"Dosya ismi beklenen formatta değil: {file}")
        continue

    eser_numarasi, eser_adi = parts
    eser_adi = eser_adi.replace('.jpg', '').strip()  # Uzantıyı kaldır ve boşlukları temizle

    # Hedef klasörde uygun alt klasörleri oluştur
    target_folder = os.path.join(val_dir, eser_adi)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Resmi hedef klasöre taşıyalım
    shutil.move(os.path.join(source_dir, file), os.path.join(target_folder, file))"""

print("Resimler başarıyla taşındı ve eğitim-doğrulama setlerine ayrıldı.")
"""import os
import shutil
import random

# Kaynak ve hedef dizinleri belirtelim
source_dir = 'van_gogh_resimleri_2'  # Resimlerin bulunduğu dizin
train_dir = 'van_gogh_resimleri_train_val_split2/train/'  # Eğitim verisi için hedef klasör
val_dir = 'van_gogh_resimleri_train_val_split2/val/'  # Doğrulama verisi için hedef klasör

# Eğitim ve doğrulama setleri için bir oran belirleyelim (örneğin, %80 eğitim, %20 doğrulama)
train_ratio = 0.8

# Tüm resim dosyalarını alalım
image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]

# Resimleri sınıflara göre ayıralım
class_dict = {}

for file in image_files:
    parts = file.split('_', 1)  # Dosya ismini eserno_eseradı şeklinde ayırıyoruz
    if len(parts) != 2:
        print(f"Dosya ismi beklenen formatta değil: {file}")
        continue

    eser_numarasi, eser_adi = parts
    eser_adi = eser_adi.replace('.jpg', '').strip()  # Uzantıyı kaldır ve boşlukları temizle

    if eser_adi not in class_dict:
        class_dict[eser_adi] = []
    class_dict[eser_adi].append(file)

# Hedef dizinleri kontrol et ve oluştur
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# Her sınıf için eğitim ve doğrulama setlerine verileri ayıralım
for class_name, files in class_dict.items():
    # Eğitim ve doğrulama dosyaları arasındaki oranı ayarlayalım
    random.shuffle(files)

    # Burada resimleri doğru şekilde ayırıyoruz, ancak her sınıfın aynı sayıda resme sahip olmasını sağlıyoruz.
    num_train = int(len(files) * train_ratio)
    num_val = len(files) - num_train

    train_files = files[:num_train]
    val_files = files[num_train:]

    # Sınıf için hedef klasörler oluştur
    class_train_dir = os.path.join(train_dir, class_name)
    class_val_dir = os.path.join(val_dir, class_name)

    if not os.path.exists(class_train_dir):
        os.makedirs(class_train_dir)
    if not os.path.exists(class_val_dir):
        os.makedirs(class_val_dir)

    # Eğitim ve doğrulama setlerine resimleri taşıyalım
    for file in train_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(class_train_dir, file))

    for file in val_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(class_val_dir, file))"""

print("Veri kümesi eğitim ve doğrulama setlerine başarıyla ayrıldı.")
