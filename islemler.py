from flask import Blueprint, render_template, request, redirect, url_for, flash
#render_template html sablonlarını kontrol etmek için kullanılır ,request = kullanıcıdan gelen verileri okumak için
#redirect=kullanıcıyı başka bir sayfaya yönlendirmek için, url_for=dinamik url oluşturmak için
from werkzeug.security import generate_password_hash
from veritabani import db, Kullanici
from werkzeug.security import generate_password_hash, check_password_hash

islemler = Blueprint('islemler', __name__)

# Kullanıcı Kayıt Sayfası

@islemler.route('/uye_ol', methods=['GET', 'POST'])
def uye_ol():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            flash("TÜM ALANLARI DOLDURUN!", category='error')
            return redirect(url_for('islemler.uye_ol'))

        kullanici_varmi = Kullanici.query.filter_by(email=email).first()

        if kullanici_varmi:
            flash("BU E-POSTA ZATEN KULLANILIYOR!", category='error')
        else:
            hashed_password = generate_password_hash(password)  # Şifre hashlenerek kaydediliyor
            yeni_kullanici = Kullanici(email=email, password=hashed_password)
            db.session.add(yeni_kullanici)
            db.session.commit()
            flash("BAŞARIYLA KAYIT OLDUNUZ! GİRİŞE YÖNLENDİRİLİYORSUNUZ...", category='success')

            # Önce `uye_ol` sayfasına yönlendir, sonra otomatik `giris_uye`'ye yönlendirsin
            return redirect(url_for('islemler.uye_ol'))

    return render_template("uye_ol.html")


#  Kullanıcı Giriş Sayfası
@islemler.route('/giris_uye', methods=['GET', 'POST'])
def giris_uye():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        kullanici = Kullanici.query.filter_by(email=email).first()

        if kullanici and check_password_hash(kullanici.password, password):  # Hash kontrolü
            flash("GİRİŞ BAŞARILI!", category='success')
            return redirect(url_for('islemler.giris2'))  # Giriş başarılı olduğunda giris2.html'e yönlendir

        else:
            flash("GEÇERSİZ E-POSTA VEYA ŞİFRE!", category='error')

    return render_template("giris_uye.html")


# Şifre sıfırlama sayfası
@islemler.route('/sifremi_unuttum', methods=['GET', 'POST'])
def sifremi_unuttum():
    if request.method == 'POST':
        email = request.form.get('email')
        yeni_sifre = request.form.get('new_password')
        yeni_sifre_tekrar = request.form.get('new_password_repeat')

        if yeni_sifre != yeni_sifre_tekrar:
            flash(" Şifreler eşleşmiyor!", category='error')
            return redirect(url_for('islemler.sifremi_unuttum'))

        kullanici = Kullanici.query.filter_by(email=email).first()
        if kullanici:
            hashed_password = generate_password_hash(yeni_sifre)  # Şifre hashlenerek güncelleniyor
            kullanici.password = hashed_password
            db.session.commit()
            flash(" Şifreniz başarıyla güncellendi. Giriş yapabilirsiniz.", category='success')
            return redirect(url_for('islemler.giris_uye'))
        else:
            flash(" Bu e-posta ile kayıtlı kullanıcı bulunamadı!", category='error')

    return render_template("sifremi_unuttum.html")
# Giriş Başarılı Sayfası (giris2.html)
@islemler.route('/giris2')
def giris2():
    return render_template("giris2.html")  #  Kullanıcı giriş yaptıktan sonra açılacak sayfa

