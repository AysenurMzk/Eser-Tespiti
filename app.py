import os
import cv2
import requests
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
import numpy as np
from flask_login import LoginManager, login_user, login_required, current_user, logout_user
from islemler import islemler
from veritabani import db, Kullanici, Baktiklarim
from resim_tahmin import load_trained_model, predict_image, start_camera, save_frame
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__, template_folder="sayfalar", static_folder="ekler")
CORS(app, supports_credentials=True)  # CORS ayarları eklendi
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'gizli_key' #oturumların güvenlik ayarı için
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Çerez güvenlik ayarı
app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS için

# Veritabanı bağlantısı başlatır ve Blueprint
db.init_app(app)
app.register_blueprint(islemler, url_prefix='/') #uygualamya dahil etme

# Flask-Login yapılandırması
login_manager = LoginManager()#OTURUM ACMA YÖNTEMİ için
login_manager.init_app(app)#flaska entegre eder
login_manager.login_view = 'islemler.giris_uye'#kullanıcı giriş yapmadığında yönlenririlecek sayfa adı

@login_manager.user_loader
def load_user(user_id):
    return Kullanici.query.get(int(user_id))#kimlik doğrulama ıd alır

# Model ve kamera
model, class_labels = load_trained_model()
camera = start_camera()

# Dosya yolları
UPLOAD_FOLDER = 'uploads'
ESERLER_TXT = 'eserler.txt'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def clear_uploads():
    """Uploads klasöründeki tüm dosyaları temizler"""
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Hata oluştu: {e}")

def goruntu():
    """Video feed için kareleri hazırlar."""
    while True:#göruntuyu freame olarak alıp jpeg formatına dönuşturur
        success, frame = camera.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        jpg = buffer.tobytes()
        yield (b'--frame\r\n'#her kare için
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')

def dogru_sayfa_basligi(eser_adi):
    """Wikipedia API ile en uygun başlığı bul."""
    url = "https://tr.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": eser_adi, "format": "json"}
    #apıye sorgu yappcz, aram sonucları listesini alcz ,yanıt json formatında olck
    r = requests.get(url, params=params)#get isteğiyle ğıdan veri alır
    if r.ok:
        data = r.json().get("query", {}).get("search", [])
        if data:
            return data[0]["title"]
    return eser_adi#en uygun baslığı bulur

def txt_dosyasindan_ara(eser_adi):#verilen eser adına göre txt de arama yapıyor
    """Eser bilgisini yerel txt dosyasından arar ve formatlar."""
    TRANSLATIONS = {
        "Original Title": "Orijinal Başlık",
        "Date": "Tarih",
        "Style": "Stil",
        "Genre": "Tür",
        "Media": "Teknik",
        "Location": "Konum",
        "Dimensions": "Boyutlar"
    }

    try:
        with open(ESERLER_TXT, 'r', encoding='utf-8') as file:
            content = file.read()#txt okuma modunda acıldı

        eserler = content.split('\n\n')

        for eser in eserler:
            if eser_adi.lower() in eser.lower():#buyuk kucuge uyumsuz
                lines = eser.split('\n')#ese rbilgileri satırla ayrıldı
                formatted_info = "<div class='eser-bilgisi'>" #eser bilgileri html formatında işlenir
                for line in lines:
                    line = line.strip()#satır basi sonundaki boşluğu temizle
                    if not line:
                        continue
                    if ':' in line:
                        key, value = line.split(':', 1)#satirda : varsa key
                        key = key.strip()
                        value = value.strip()
                        translated_key = TRANSLATIONS.get(key, key)#baslık turkçeye cevir
                        formatted_info += f"<p><strong>{translated_key}:</strong> {value}</p>"
                    else:
                        formatted_info += f"<p>{line}</p>"
                formatted_info += "</div>"
                return formatted_info
    except FileNotFoundError:
        print(f"{ESERLER_TXT} dosyası bulunamadı!")
    return None

def wikipedia_bilgisi(eser_adi):
    """Sadece Wikipedia'dan bilgi getirir."""
    title = dogru_sayfa_basligi(eser_adi)# Wikipedia sayfasının başlığını al
    page = f"https://tr.wikipedia.org/wiki/{title.replace(' ', '_')}"# Sayfanın URL'sini oluştur
    r = requests.get(page) # Wikipedia sayfasını al
    if not r.ok:
        return None

    soup = BeautifulSoup(r.text, 'html.parser')# Sayfa içeriğini BeautifulSoup ile parse et
    ps = soup.find_all('p')# Sayfadaki tüm <p> (paragraf) etiketlerini bul
    texts = [p.get_text().strip() for p in ps if len(p.get_text().strip()) > 50]

    if texts:# Eğer paragraflar varsa, ilk 3 paragrafı birleştir
        wikipedia_content = ""
        wikipedia_content += "\n\n".join(f"{text}" for text in texts[:3])
        wikipedia_content += "" # Formatlanmış içeriği döndür
        return wikipedia_content
    return None

@app.route('/check_auth')
def check_auth():
    return jsonify({
        'logged_in': current_user.is_authenticated,
        'user_id': current_user.id if current_user.is_authenticated else None
    })#kullanıcı giriş yapmışsa id dondurur yapmamamıssa none

# Rotalar
@app.route('/')
def ana_sayfa():
    return render_template("index.html")

@app.route('/giris2')
@login_required
def giris2():
    return render_template("giris2.html")

@app.route('/baktiklarim')
@login_required#giriş yapmak zorunlu
def baktiklarim():
    user_history = Baktiklarim.query.filter_by(user_id=current_user.id) \
        .order_by(Baktiklarim.created_at.desc()) \
        .all()
    return render_template("baktiklarim.html", history=user_history)

@app.route('/baktiklarim_ekle', methods=['POST'])
@login_required
def baktiklarim_ekle():
    if not current_user.is_authenticated:
        return jsonify({'status': 'error', 'message': 'Giriş yapmalısınız!'}), 401

    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'Geçersiz veri!'}), 400

    artwork_name = data.get('artwork_name')
    image_path = data.get('image_path', 'no-image')

    if not artwork_name:
        return jsonify({'status': 'error', 'message': 'Eser adı gereklidir!'}), 400

    try:
        yeni_bakilan = Baktiklarim(
            user_id=current_user.id,
            artwork_name=artwork_name,
            image_path=image_path,
            created_at=datetime.now()
        )
        db.session.add(yeni_bakilan)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Eser listenize eklendi!'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'Kayıt sırasında hata: {str(e)}'}), 500

@app.route('/video_feed')
def video_feed():
    return Response(goruntu(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/arama', methods=['POST'])#gelen post isteğiyle kullanıcı kullanıcı görseli yükledi mi ya da eser adı girdi mi
def arama():
    clear_uploads()#eski yüklenen dosyayı sil

    eser_adi = request.form.get('eser_adi', '').strip()
    file = request.files.get('file')
    predicted_class = None#göruntu tahmin sonuçlarını tut
    formatted_prediction = None #görtuntu tahmin sonuclarını tut
    bilgi = None
    from_file = False

    if file and file.filename != '':#dosya yuklendi mş
        filename = secure_filename(file.filename)#özel karakterler kaldır
        path = os.path.join(UPLOAD_FOLDER, filename)#yolu kaydet
        file.save(path)
        predicted_class = predict_image(path)#göruntu uzerinde tahmin yapmak için

        if predicted_class:
            formatted_prediction = format_artwork_name(predicted_class)#tahmin edilen sınıf isme göre biçimlendir
            txt_bilgi = txt_dosyasindan_ara(formatted_prediction)#bilgiyi ara
            wiki_bilgi = wikipedia_bilgisi(formatted_prediction + " tablosu")#wikipedia bilgisi al

            bilgi = ""
            if txt_bilgi:
                bilgi += txt_bilgi
            if wiki_bilgi:
                if txt_bilgi:
                    bilgi += "<hr>"
                bilgi += wiki_bilgi
            if not bilgi:
                bilgi = "Bu eser hakkında bilgi bulunamadı."

            eser_adi = formatted_prediction
            from_file = True

    if not from_file and eser_adi:
        bilgi = wikipedia_bilgisi(eser_adi + " tablosu")
        if not bilgi:
            bilgi = "Bu eser hakkında bilgi bulunamadı."

    return jsonify({ #bilgiyi json formatında tutar
        "status": "success",
        "bilgi": bilgi,
        "predicted_class": predicted_class,
        "formatted_prediction": formatted_prediction,
        "eser_adi": eser_adi,
        "from_file": from_file,
        "image_path": file.filename if file else None
    })

def format_artwork_name(raw_name):# sayıları ve _ siler
    """Eser adını daha okunabilir hale getirir."""
    if '_' in raw_name:
        parts = raw_name.split('_', 1)
        if parts[0].isdigit():
            formatted = parts[1].replace('_', ' ')
        else:
            formatted = raw_name.replace('_', ' ')
    else:
        formatted = raw_name
    return ' '.join(word.capitalize() for word in formatted.split())

@app.route('/predict_file', methods=['POST'])
def predict_file():#dosya göruntusunu alır ve tahmin yapar
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']#yuklenen dosyayı alır
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)#dosyanın kaydedeceği yol
    file.save(path)
    pred = predict_image(path)#tahmin yapar
    return jsonify({'predicted_class': pred, 'image_path': filename})
    #jsona döndurur
@app.route('/predict_camera', methods=['GET'])
def predict_camera():
    try:
        path = save_frame(camera, UPLOAD_FOLDER)
        if not os.path.exists(path):
            return jsonify({'error': 'Görüntü kaydedilemedi'}), 500

        img = cv2.imread(path)
        if img is None:
            return jsonify({'error': 'Görüntü okunamadı'}), 500

        predicted_class = predict_image(path)
        return jsonify({
            'predicted_class': predicted_class,
            'image_path': os.path.basename(path)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

with app.app_context():
    db.create_all()
    print("Veritabanı ve tablolar oluşturuldu!")

if __name__ == '__main__':
    app.run(debug=True)