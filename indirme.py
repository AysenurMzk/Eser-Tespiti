#klorleme 400 tane secme
"""import os
import random
import shutil
import json

# Yolları belirle
kaynak_klasor = 'vg2_klosorlu'
hedef_klasor = 'vangogh'
egitim_klasoru = os.path.join(hedef_klasor, 'train')
test_klasoru = os.path.join(hedef_klasor, 'val')
secim_dosyasi = 'secilen_klasorler.json'  # Seçilen klasörleri saklayacağımız JSON dosyası

# Hedef klasörler yoksa oluştur
os.makedirs(egitim_klasoru, exist_ok=True)
os.makedirs(test_klasoru, exist_ok=True)

# vg2_klosorlu klasöründeki tüm alt klasörleri listele
klasorler = [f for f in os.listdir(kaynak_klasor) if os.path.isdir(os.path.join(kaynak_klasor, f))]

# Daha önce seçilen klasörlerin kaydını kontrol et
if os.path.exists(secim_dosyasi):
    # Önceden seçilen klasörleri yükle
    with open(secim_dosyasi, 'r') as f:
        secilen_klasorler = json.load(f)
else:
    # Rastgele 400 klasör seç
    secilen_klasorler = random.sample(klasorler, 400)
    # Seçilen klasörleri kaydet
    with open(secim_dosyasi, 'w') as f:
        json.dump(secilen_klasorler, f)

# Seçilen her klasörü train ve test klasörlerine kopyala
for klasor in secilen_klasorler:
    # Klasörün yolunu belirle
    kaynak_klasor_yolu = os.path.join(kaynak_klasor, klasor)

    # Klasörün altındaki resimleri listele
    resim_dosyalari = [f for f in os.listdir(kaynak_klasor_yolu) if not f.startswith('.')]
    if not resim_dosyalari:
        continue

    # Train ve test klasörlerinde ilgili alt klasörü oluştur
    egitim_klasor_yolu = os.path.join(egitim_klasoru, klasor)
    test_klasor_yolu = os.path.join(test_klasoru, klasor)
    os.makedirs(egitim_klasor_yolu, exist_ok=True)
    os.makedirs(test_klasor_yolu, exist_ok=True)

    # Resimleri her iki klasöre de kopyala
    for resim_adi in resim_dosyalari:
        kaynak_resim_yolu = os.path.join(kaynak_klasor_yolu, resim_adi)

        # Resmin adını ve uzantısını al
        uzanti = os.path.splitext(resim_adi)[1]

        # Hedef yolunu oluştur
        egitim_hedef_yolu = os.path.join(egitim_klasor_yolu, resim_adi)
        test_hedef_yolu = os.path.join(test_klasor_yolu, resim_adi)

        # Resimleri train ve test klasörlerine kopyala
        shutil.copy(kaynak_resim_yolu, egitim_hedef_yolu)
        shutil.copy(kaynak_resim_yolu, test_hedef_yolu)

print("400 klasör train/ ve test/ klasörlerine kopyalandı, seçim kaydedildi.")
"""


#teste veri arttırma
"""import os
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


# Veri artırma işlemleri
def rotate_image(image, angle):
    return image.rotate(angle, expand=True)


def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def zoom_image(image, zoom_factor):
    w, h = image.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    zoomed = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return zoomed.crop((left, top, left + w, top + h))


def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def apply_gaussian_blur(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius))


def add_noise(image):
    np_image = np.array(image)
    noise = np.random.normal(0, 25, np_image.shape)
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return Image.fromarray(noisy_image.astype(np.uint8))


def horizontal_flip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def vertical_flip(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def color_jitter(image, brightness=0, contrast=0, saturation=0, hue=0):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1 + random.uniform(-brightness, brightness))

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1 + random.uniform(-contrast, contrast))

    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1 + random.uniform(-saturation, saturation))

    # Convert to HSV to apply hue shift
    image = image.convert('HSV')
    np_img = np.array(image, dtype=np.float32)  # Ensure the data type is float32

    # Hue shift
    np_img[..., 0] += random.uniform(-hue, hue)

    # Ensure the hue values are within the valid range (0 to 255)
    np_img[..., 0] = np_img[..., 0] % 256
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)  # Convert back to uint8

    return Image.fromarray(np_img, 'HSV').convert('RGB')


def random_crop(image, crop_size):
    w, h = image.size
    left = random.randint(0, w - crop_size)
    top = random.randint(0, h - crop_size)
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))


def resize_image(image, scale_factor):
    w, h = image.size
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    return image.resize((new_w, new_h), Image.LANCZOS)


def affine_transform(image):
    max_angle = 30
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle)


def shift_hue(image, delta):
    img = image.convert("HSV")
    np_img = np.array(img, dtype=np.float32)  # HSV'deki tüm veriler float32 tipine dönüştürülür

    # Hue kanalını kaydır
    np_img[..., 0] = (np_img[..., 0] + delta)  # Hue değerini kaydır

    # Hue değerlerini 0 ile 255 arasında sınırla
    np_img[..., 0] = np.clip(np_img[..., 0], 0, 255)  # Hue'yu 0-255 arası değerlerle sınırladık

    np_img = np.clip(np_img, 0, 255).astype(np.uint8)  # Tüm değerleri 0-255 arasında tutarak uint8 tipine dönüştür

    return Image.fromarray(np_img, "HSV").convert("RGB")


def random_erase(image, erase_prob=0.5):
    np_img = np.array(image)
    if random.random() < erase_prob:
        h, w, _ = np_img.shape
        top = random.randint(0, h // 2)
        left = random.randint(0, w // 2)
        bottom = random.randint(top + 1, h)
        right = random.randint(left + 1, w)
        np_img[top:bottom, left:right, :] = 0
    return Image.fromarray(np_img)


def random_rotate(image, max_angle=15):
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle)


# Yolları belirle
test_klasoru = 'vangogh/train'

# Test klasöründeki her klasöre veri artırma işlemi yap
for klasor in os.listdir(test_klasoru):
    klasor_yolu = os.path.join(test_klasoru, klasor)

    # Alt klasördeki resimleri listele
    if os.path.isdir(klasor_yolu):
        for resim_adi in os.listdir(klasor_yolu):
            if resim_adi.startswith('.'):
                continue

            # Resmin tam yolunu belirle
            resim_yolu = os.path.join(klasor_yolu, resim_adi)
            img = Image.open(resim_yolu)

            # Resmin ismini ve uzantısını al
            name, ext = os.path.splitext(resim_adi)

            # Klasördeki mevcut augmentasyonları kontrol et
            existing_files = os.listdir(klasor_yolu)
            suffixes = [f"{name}_{i}{ext}" for i in range(1, 1000)]  # 1000'e kadar olan augmentasyonlar
            existing_files = set(existing_files)

            augmentations = [
                ('rotate_90', rotate_image(img, 90)),
                ('flip', horizontal_flip(img)),
                ('zoom_1.2', zoom_image(img, 1.2)),
                ('brightness_1.5', adjust_brightness(img, 1.5)),
                ('contrast_1.5', adjust_contrast(img, 1.5)),
                ('gaussian_blur', apply_gaussian_blur(img, 2)),
                ('noise', add_noise(img)),
                ('vertical_flip', vertical_flip(img)),
                ('color_jitter', color_jitter(img, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
                ('random_crop', random_crop(img, 100)),
                ('resize_1.5', resize_image(img, 1.5)),
                ('affine_transform', affine_transform(img)),
                ('shift_hue', shift_hue(img, 10)),
                ('random_erase', random_erase(img))
            ]

            # Augmentasyonlardan her biri için yeni resim kaydet
            for idx, (suffix, augmented_img) in enumerate(augmentations, start=1):
                augmented_file_name = f"{name}_{suffix}{ext}"
                if augmented_file_name not in existing_files:
                    augmented_img.save(os.path.join(klasor_yolu, augmented_file_name))
                    existing_files.add(augmented_file_name)

print("Veri artırma işlemi tamamlandı. Yeni resimler test klasörüne kaydedildi.")
"""


#train
"""import os
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


# Veri artırma işlemleri
def rotate_image(image, angle):
    return image.rotate(angle, expand=True)


def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)  # Yatay aynalama (flip)


def zoom_image(image, zoom_factor):
    w, h = image.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    zoomed = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return zoomed.crop((left, top, left + w, top + h))  # Yakınlaştırma (zoom)


def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def apply_gaussian_blur(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius))


def add_noise(image):
    np_image = np.array(image)
    noise = np.random.normal(0, 25, np_image.shape)
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return Image.fromarray(noisy_image.astype(np.uint8))


def horizontal_flip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def vertical_flip(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def color_jitter(image, brightness=0, contrast=0, saturation=0, hue=0):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1 + random.uniform(-brightness, brightness))

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1 + random.uniform(-contrast, contrast))

    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1 + random.uniform(-saturation, saturation))

    # Convert to HSV to apply hue shift
    image = image.convert('HSV')
    np_img = np.array(image, dtype=np.float32)  # Ensure the data type is float32

    # Hue shift
    np_img[..., 0] += random.uniform(-hue, hue)

    # Ensure the hue values are within the valid range (0 to 255)
    np_img[..., 0] = np_img[..., 0] % 256
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)  # Convert back to uint8

    return Image.fromarray(np_img, 'HSV').convert('RGB')


def random_crop(image, crop_size):
    w, h = image.size
    left = random.randint(0, w - crop_size)
    top = random.randint(0, h - crop_size)
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))


def resize_image(image, scale_factor):
    w, h = image.size
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    return image.resize((new_w, new_h), Image.LANCZOS)


def affine_transform(image):
    max_angle = 30
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle)


def shift_hue(image, delta):
    img = image.convert("HSV")
    np_img = np.array(img, dtype=np.float32)  # HSV'deki tüm veriler float32 tipine dönüştürülür

    # Hue kanalını kaydır
    np_img[..., 0] = (np_img[..., 0] + delta)  # Hue değerini kaydır

    # Hue değerlerini 0 ile 255 arasında sınırla
    np_img[..., 0] = np_img[..., 0] % 256  # Hue'yu 0-255 arası değerlerle sınırladık

    np_img = np.clip(np_img, 0, 255).astype(np.uint8)  # Tüm değerleri 0-255 arasında tutarak uint8 tipine dönüştür

    return Image.fromarray(np_img, "HSV").convert("RGB")


def random_erase(image, erase_prob=0.5):
    np_img = np.array(image)
    if random.random() < erase_prob:
        h, w, _ = np_img.shape
        top = random.randint(0, h // 2)
        left = random.randint(0, w // 2)
        bottom = random.randint(top + 1, h)
        right = random.randint(left + 1, w)
        np_img[top:bottom, left:right, :] = 0
    return Image.fromarray(np_img)


def random_rotate(image, max_angle=15):
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle)


# Yolları belirle
train_klasoru = 'vangogh/train'

# Train klasöründeki her klasöre veri artırma işlemi yap
for klasor in os.listdir(train_klasoru):
    klasor_yolu = os.path.join(train_klasoru, klasor)

    # Alt klasördeki resimleri listele
    if os.path.isdir(klasor_yolu):
        for resim_adi in os.listdir(klasor_yolu):
            if resim_adi.startswith('.'):
                continue

            # Resmin tam yolunu belirle
            resim_yolu = os.path.join(klasor_yolu, resim_adi)
            img = Image.open(resim_yolu)

            # Resmin ismini ve uzantısını al
            name, ext = os.path.splitext(resim_adi)

            # Klasördeki mevcut augmentasyonları kontrol et
            existing_files = os.listdir(klasor_yolu)
            existing_files = set(existing_files)

            # Veri artırma işlemleri: Aynalama, Yakınlaştırma, vb. (şimdi 11 işlem var)
            augmentations = [
                ('rotate_90', rotate_image(img, 90)),
                ('flip', horizontal_flip(img)),
                ('zoom_1.2', zoom_image(img, 1.2)),
                ('brightness_1.5', adjust_brightness(img, 1.5)),
                ('contrast_1.5', adjust_contrast(img, 1.5)),
                ('gaussian_blur', apply_gaussian_blur(img, 2)),
                ('noise', add_noise(img)),
                ('vertical_flip', vertical_flip(img)),
                ('color_jitter', color_jitter(img, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
                ('random_crop', random_crop(img, 100)),
                ('resize_1.5', resize_image(img, 1.5))
            ]

            # Augmentasyonlardan her biri için yeni resim kaydet
            for idx, (suffix, augmented_img) in enumerate(augmentations, start=1):
                # Her artırma işleminde dosya adı orijinal dosya adının sonuna "_1", "_2" şeklinde eklenir
                augmented_file_name = f"{name}_{idx}{ext}"
                if augmented_file_name not in existing_files:
                    augmented_img.save(os.path.join(klasor_yolu, augmented_file_name))
                    existing_files.add(augmented_file_name)

print("Veri artırma işlemi tamamlandı. Yeni resimler train klasörüne kaydedildi.")
"""


#val de veri arttırma

"""import os
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


# Veri artırma işlemleri
def rotate_image(image, angle):
    return image.rotate(angle, expand=True)


def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)  # Yatay aynalama (flip)


def zoom_image(image, zoom_factor):
    w, h = image.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    zoomed = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return zoomed.crop((left, top, left + w, top + h))  # Yakınlaştırma (zoom)


# Yolları belirle
val_klasoru = 'vangogh/val'

# Veri artırma işlemleri: Her resme 11 farklı işlem uygulanacak
for klasor in os.listdir(val_klasoru):
    klasor_yolu = os.path.join(val_klasoru, klasor)

    # Alt klasördeki resimleri listele
    if os.path.isdir(klasor_yolu):
        for resim_adi in os.listdir(klasor_yolu):
            if resim_adi.startswith('.'):
                continue

            # Resmin tam yolunu belirle
            resim_yolu = os.path.join(klasor_yolu, resim_adi)
            img = Image.open(resim_yolu)

            # Resmin ismini ve uzantısını al
            name, ext = os.path.splitext(resim_adi)

            # Klasördeki mevcut augmentasyonları kontrol et
            existing_files = os.listdir(klasor_yolu)
            existing_files = set(existing_files)

            # Veri artırma işlemleri
            augmentations = [
                ('flip', flip_image(img)),  # Yatay aynalama
                ('zoom_1.2', zoom_image(img, 1.2))  # Yakınlaştırma
            ]

            # Augmentasyonları kaydet
            for idx, (suffix, augmented_img) in enumerate(augmentations, start=1):
                # Her artırma işleminde dosya adı orijinal dosya adının sonuna "_1", "_2" şeklinde eklenir
                augmented_file_name = f"{name}_{idx}{ext}"
                if augmented_file_name not in existing_files:
                    augmented_img.save(os.path.join(klasor_yolu, augmented_file_name))
                    existing_files.add(augmented_file_name)

print("Veri artırma işlemi tamamlandı. Yeni resimler val klasörüne kaydedildi.")
"""


"""import os
import time
import requests
from bs4 import BeautifulSoup
import re
from PIL import Image
from io import BytesIO
import egitim
# Ayarlar
kayit_klasoru = "van_gogh_resimleri_2"
genislik = 512
yukseklik = 512
os.makedirs(kayit_klasoru, exist_ok=True)

# HTTP Ayarları
basliklar = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

zaman_asimi = 30
maksimum_deneme = 3
bekleme = 2

# WikiArt'tan Van Gogh eserlerini al
wikiart_adres = "https://www.wikiart.org/en/vincent-van-gogh/all-works/text-list"
cevap = requests.get(wikiart_adres, headers=basliklar, timeout=zaman_asimi)

eser_listesi = []
if cevap.status_code == 200:
    sayfa = BeautifulSoup(cevap.text, "html.parser")
    for li in sayfa.find_all('li', class_='painting-list-text-row'):
        baglanti = li.find('a', href=True)
        if baglanti and "/en/vincent-van-gogh/" in baglanti['href']:
            adi = baglanti.text.strip()
            url = "https://www.wikiart.org" + baglanti['href']
            eser_listesi.append((adi, url))
else:
    print("WikiArt sayfası alınamadı!")
    exit()

baslangic = 0

def gecerli_mi(resim_yolu):
    try:
        with Image.open(resim_yolu) as resim:
            resim.load()
        return os.path.getsize(resim_yolu) > 1024
    except Exception as hata:
        print(f"Geçersiz resim bulundu ({resim_yolu}): {hata}")
        return False

def indir_ve_yeniden_boyutlandir(url, kayit_yolu):
    try:
        yanit = requests.get(url, headers=basliklar, stream=True, timeout=zaman_asimi)
        if yanit.status_code == 200:
            resim = Image.open(BytesIO(yanit.content))

            if resim.mode in ('RGBA', 'LA'):
                arkaplan = Image.new('RGB', resim.size, (255, 255, 255))
                arkaplan.paste(resim, mask=resim.split()[-1])
                resim = arkaplan

            resim = resim.resize((genislik, yukseklik), Image.LANCZOS)
            gecici_yol = kayit_yolu + '.temp'
            resim.save(gecici_yol, "JPEG", quality=90, optimize=True, progressive=True)

            if gecerli_mi(gecici_yol):
                if os.path.exists(kayit_yolu):
                    os.remove(kayit_yolu)
                os.rename(gecici_yol, kayit_yolu)
                print(f" Kaydedildi: {os.path.basename(kayit_yolu)}")
                return True
            else:
                os.remove(gecici_yol)
                print(f" Geçersiz resim silindi: {os.path.basename(kayit_yolu)}")
                return False
    except Exception as hata:
        print(f"Hata oluştu: {str(hata)}")
        return False

# Ana indirme döngüsü
for sira, (eser_adi, eser_url) in enumerate(eser_listesi[baslangic:], start=baslangic + 1):
    temiz_adi = re.sub(r'[^\w\s-]', '', eser_adi.split(',')[0].strip()).lower().replace(' ', '_')
    dosya_adi = f"{str(sira).zfill(4)}_{temiz_adi}.jpg"
    tam_yol = os.path.join(kayit_klasoru, dosya_adi)

    if os.path.exists(tam_yol):
        if gecerli_mi(tam_yol):
            print(f" Atlandı ({sira}/{len(eser_listesi)}): {dosya_adi} geçerli.")
            continue
        else:
            print(f" Bozuk dosya bulundu ve silindi: {dosya_adi}")
            os.remove(tam_yol)

    print(f"\n İndiriliyor ({sira}/{len(eser_listesi)}): {eser_adi}")

    for deneme in range(1, maksimum_deneme + 1):
        try:
            yanit = requests.get(eser_url, headers=basliklar, timeout=zaman_asimi)

            if yanit.status_code == 200:
                sayfa = BeautifulSoup(yanit.text, "html.parser")
                img_tag = sayfa.find("img", {"itemprop": "image"}) or \
                          sayfa.find("img", class_="ms-zoom-cursor") or \
                          sayfa.find("img", class_="masonry-image-img")

                if img_tag:
                    resim_url = img_tag.get('src') or img_tag.get('data-src')
                    if resim_url:
                        if not resim_url.startswith('http'):
                            resim_url = 'https:' + resim_url

                        print(f" Resim URL'si: {resim_url}")
                        if indir_ve_yeniden_boyutlandir(resim_url, tam_yol):
                            break
                else:
                    print(" Resim etiketi bulunamadı.")
            else:
                print(f"HTTP {yanit.status_code} hatası.")

        except requests.exceptions.RequestException as hata:
            print(f" İstek hatası: {str(hata)}")

        if deneme < maksimum_deneme:
            bekle = bekleme * deneme
            print(f" {bekle} saniye bekleniyor, yeniden deneniyor ({deneme}/{maksimum_deneme})...")
            time.sleep(bekle)

    time.sleep(bekleme)

print("\n İndirme işlemi tamamlandı!")
"""

"""import os
import shutil

# Kaynak klasör ve hedef klasör
source_folder = 'van_gogh_resimleri_2'
destination_folder = 'vg2_klosorlu'

# Hedef klasörü oluştur
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Kaynak klasördeki tüm dosyaları al
for file_name in os.listdir(source_folder):
    # Yalnızca .jpg uzantılı dosyaları seç
    if file_name.endswith('.jpg'):
        # Dosya ismini ayırarak sıra numarası ve eser adını ayır
        parts = file_name.split('_', 1)

        if len(parts) == 2:
            # Sıra numarası ve eser adı
            artwork_name = parts[1].split('.')[0]  # '.' karakterinden önceki kısmı al

            # Her resim için, sıra numarasını da içeren bir klasör oluştur
            artwork_folder = os.path.join(destination_folder, file_name.split('.')[0])

            # Eğer klasör yoksa oluştur
            if not os.path.exists(artwork_folder):
                os.makedirs(artwork_folder)

            # Dosyayı ilgili klasöre taşı
            source_file = os.path.join(source_folder, file_name)
            destination_file = os.path.join(artwork_folder, file_name)

            # Eğer dosya zaten hedef klasörde varsa, taşımaya gerek yok
            if not os.path.exists(destination_file):
                shutil.move(source_file, destination_file)

print("İşlem tamamlandı.")"""
