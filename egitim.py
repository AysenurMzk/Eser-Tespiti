import os
import shutil
import pandas as pd
import re
import difflib

# CSV dosyasının yolu
csv_yolu = "sanat_eserleri_listesi_guncel_duzeltilmis.csv"

# Kaynak klasör ve hedef klasör
kaynak_klasor = "van_gogh_resimleri_2"
hedef_klasor = "van_gogh_resimleri_standart"

# CSV'yi oku ve yalnızca "Vincent van Gogh" sanatçısının eserlerini al
df = pd.read_csv(csv_yolu)
df_vangogh = df[df["artist_name"].str.lower() == "vincent_van_gogh"]

# Dosya adını temizleyip standart hale getiren fonksiyon
def temizle_ve_standartlastir(text):
    text = text.lower()
    text = text.replace("vincent-van-gogh_", "")
    text = re.sub(r'[^a-z0-9]', '_', text)
    text = re.sub(r'_+', '_', text)
    text = text.strip("_")
    return text

# Klasördeki tüm dosya adlarını temizle
dosya_listesi = os.listdir(kaynak_klasor)
temiz_dosyalar = {
    f: temizle_ve_standartlastir(f.split(".")[0]) for f in dosya_listesi
}

# Eşik değeri
esik = 0.45  # Eşleşme için eşik değeri

# Eşleşmeyen dosyaları tutmak için liste
eslesmeyenler_listesi = []

# Her Van Gogh eserini kontrol et ve en iyi eşleşmeyi bul
"""for _, row in df_vangogh.iterrows():
    image_path = row["image_path"]
    style, filename = image_path.split("/", 1)
    hedef_isim = temizle_ve_standartlastir(filename.split(".")[0])

    # En benzer dosyayı bul (eşleşme puanı > 0.45 ise kabul ediyoruz)
    best_match = None
    best_score = 0

    for dosya_ad, temiz_ad in temiz_dosyalar.items():
        skor = difflib.SequenceMatcher(None, hedef_isim, temiz_ad).ratio()
        if skor > best_score:
            best_score = skor
            best_match = dosya_ad

    # Eğer en yüksek eşleşme puanı eşikten düşükse, eşleşmeyenler listesine ekle
    if best_score < esik:
        eslesmeyenler_listesi.append(hedef_isim)
        print(f"❌ Eşleşmedi: {hedef_isim} (en yüksek benzerlik: {best_score:.2f})")
        continue

    # Dosyayı doğru klasöre kopyala
    hedef_stil_klasoru = os.path.join(hedef_klasor, style)
    os.makedirs(hedef_stil_klasoru, exist_ok=True)
    shutil.copy(
        os.path.join(kaynak_klasor, best_match),
        os.path.join(hedef_stil_klasoru, best_match)
    )

    print(f"✅ Eşleşti: {hedef_isim} → {best_match} ({style})")"""

# Eşleşmeyen dosyaları masaüstüne kaydet
eslesmeyenler_dosya_yolu = r"C:\Users\aysen\Desktop\eslesmeyenler.txt"

with open(eslesmeyenler_dosya_yolu, "w") as dosya:
    for isim in eslesmeyenler_listesi:
        dosya.write(isim + "\n")

print(f"Eşleşmeyen dosyalar ve türler {eslesmeyenler_dosya_yolu} konumuna kaydedildi.")

# Eşleşmeyen dosyaların ve türlerin listesi
eslesmeyenler_listesi = [
    ("worn_out_1882", "Realism"),
    ("feet_1885_1", "Realism"),
    ("hand_1885_1", "Realism"),
    ("straw_hats_at_dusk_1885", "Post_Impressionism"),
    ("cows_1890_1", "Realism"),
    ("joseph_roulin_sitting_in_a_cane_chair_three_quarter_length_1888", "Post_Impressionism"),
    ("morning_going_to_work_1890", "Realism"),
    ("hands_1885_1_1", "Post_Impressionism"),
    ("can_books_wineglass_bread_and_arum_sketch_of_two_women_and_a_girl_1890_1", "Post_Impressionism"),
    ("hand_1885_1_1", "Realism"),
    ("feet_1885_1_1", "Post_Impressionism"),
    ("hands_1885_2_1", "Realism")
]

# Kaynak ve hedef klasörler
kaynak_klasor = "van_gogh_resimleri_2"
hedef_klasor = "van_gogh_resimleri_standart"

# Eşleşmeyen dosyaları türlerine göre doğru klasöre yerleştir
"""for dosya_ad, style in eslesmeyenler_listesi:
    # Klasördeki dosyayı bul
    dosya_bulundu = False
    for dosya in os.listdir(kaynak_klasor):
        if dosya_ad in dosya.lower():  # Dosya adını kontrol et
            # Hedef klasörü oluştur
            hedef_stil_klasoru = os.path.join(hedef_klasor, style)
            os.makedirs(hedef_stil_klasoru, exist_ok=True)

            # Dosyayı kopyala
            shutil.copy(
                os.path.join(kaynak_klasor, dosya),
                os.path.join(hedef_stil_klasoru, dosya)
            )
            print(f"✅ {dosya_ad} başarıyla {style} klasörüne kopyalandı.")
            dosya_bulundu = True
            break

    if not dosya_bulundu:
        print(f"❌ {dosya_ad} bulunamadı.")"""

print("\n🎉 Eşleşmeyen dosyalar türlerine göre klasöre yerleştirildi!")

