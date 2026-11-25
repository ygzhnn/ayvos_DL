"""
OpenCV Giriş: cv2.imread, cv2.imshow, cv2.resize

Bu dosya üç şeyi öğretir:
  1) imread : diskteki bir resim dosyasını Python'a okumak
  2) resize : resmin boyutunu değiştirmek
  3) imshow : resmi ekranda pencerede göstermek

Bilgisayar için bir resim, sayılardan oluşan bir tablodur.
Resmin her noktasına "piksel" denir. Renkli bir resimde her pikselin 3 sayısı
vardır (mavi, yeşil, kırmızı miktarı) ve her sayı 0 ile 255 arasındadır.
0 = hiç yok, 255 = tam dolu. Örneğin (0, 0, 255) saf kırmızıdır.
"""
from pathlib import Path  # Dosya yollarıyla (klasör/dosya adresleri) çalışmayı kolaylaştırır

import cv2  # OpenCV: görüntü işleme kütüphanesi. Kodda hep "cv2" diye yazılır.

# ------------------------------------------------------------
# Dosya yollarını hazırlıyoruz
# ------------------------------------------------------------
# __file__ şu an çalışan dosyanın (opencv_giris.py) adresidir.
# .resolve()  -> tam adresi verir (C:\Users\...\opencv_giris.py gibi)
# .parent     -> bir üst klasöre çıkar (dosyanın içinde bulunduğu klasör)
# İki kere .parent dersek proje ana klasörüne ulaşırız.
# Bu sayede script'i hangi klasörden çalıştırırsan çalıştır dosyaları bulur.
ROOT_DIR = Path(__file__).resolve().parent.parent

# "/" işareti Path nesnelerinde klasör/dosya adlarını birleştirir.
INPUT_PATH = ROOT_DIR / "data" / "test.jpg"  # Okuyacağımız resim

# Çıktıları 01_opencv_giris/outputs klasörüne kaydedeceğiz
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Klasörü oluştur. exist_ok=True: "zaten varsa hata verme" demektir.
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1) cv2.imread: dosyadan görüntü okuma
# ============================================================
# imread (image read) bir resim dosyasını okur ve bir NumPy dizisi (sayı tablosu) döndürür.
# str(INPUT_PATH): OpenCV dosya yolunu yazı (string) olarak ister, Path nesnesini yazıya çeviriyoruz.
# İkinci parametre vermezsek varsayılan olarak renkli (3 kanallı) okur.
img = cv2.imread(str(INPUT_PATH))

# DİKKAT: imread dosyayı bulamazsa hata VERMEZ, sessizce None ("hiçbir şey") döndürür.
# Kontrol etmezsek program ilerler ve çok sonra alakasız bir yerde garip bir hata verir.
# O yüzden hemen burada kontrol ediyoruz.
if img is None:
    # raise: kasten hata fırlatır ve programı durdurur.
    raise FileNotFoundError(f"Görüntü okunamadı: {INPUT_PATH}")

# imread'e ikinci parametre vererek okuma şeklini seçebiliriz:
#   cv2.IMREAD_COLOR     -> renkli oku (3 kanal). Varsayılan budur.
#   cv2.IMREAD_GRAYSCALE -> gri tonlamalı oku (tek kanal), her piksel tek bir sayı (parlaklık)
#   cv2.IMREAD_UNCHANGED -> dosyada ne varsa olduğu gibi oku (PNG'de saydamlık kanalı dahil)
gray = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)

# .shape resmin boyutlarını verir.
# Renkli resim: (yükseklik, genişlik, kanal sayısı). Örn: (2160, 5120, 3)
# Gri resim   : (yükseklik, genişlik). Kanal yok çünkü piksel başına tek sayı var.
print(f"renkli shape: {img.shape}  (yükseklik, genişlik, kanal)")
print(f"gri shape   : {gray.shape}  (kanal boyutu yok)")

# .dtype sayıların türünü verir. uint8 = "0 ile 255 arası tam sayı" demektir.
print(f"dtype       : {img.dtype}  (0-255)")

# ÖNEMLİ: OpenCV renkleri RGB (kırmızı, yeşil, mavi) DEĞİL, BGR (mavi, yeşil, kırmızı) sırasıyla tutar.
# Bu yüzden img[y, x] bize [Mavi, Yeşil, Kırmızı] verir.
# Başka kütüphaneler (Matplotlib gibi) RGB bekler, bu yüzden renkler karışabilir.
y, x = 100, 200  # DİKKAT: dizide önce satır (y, yukarıdan aşağı), sonra sütun (x, soldan sağa) gelir
b, g, r = img[y, x]  # Bu pikselin üç sayısını üç değişkene açıyoruz (unpacking)
print(f"({x},{y}) pikseli -> B={b} G={g} R={r}")


# ============================================================
# 2) cv2.resize: boyut değiştirme
# ============================================================
# img.shape[:2] ifadesi shape'in ilk iki elemanını alır: yükseklik ve genişlik.
height, width = img.shape[:2]

# a) Oran vererek küçültme: fx yatayda, fy dikeyde ölçek çarpanıdır.
#    0.5 = yarı boyut. dsize=None demek "boyutu ben vermiyorum, çarpandan hesapla" demektir.
#    interpolation: yeni piksellerin eski piksellerden nasıl hesaplanacağını seçer.
#    Küçültürken INTER_AREA en iyi sonucu verir (komşu pikselleri ortalar).
half = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# b) Sabit boyut vererek yeniden boyutlandırma.
#    DİKKAT: dsize = (GENİŞLİK, YÜKSEKLİK) sırasıyla verilir.
#    Bu, shape'in (yükseklik, genişlik) sırasının TERSİDİR. Çok sık yapılan hatadır.
#    Burada resmi kare yaptığımız için en-boy oranı bozulur ve resim ezilir.
squashed = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)


# c) En-boy oranını koruyarak belirli bir genişliğe getiren kendi fonksiyonumuz.
#    "def" ile fonksiyon tanımlarız: bir kere yazarız, istediğimiz kadar çağırırız.
def resize_to_width(image, new_width):
    """Resmin oranını bozmadan genişliğini new_width yapar."""
    old_height, old_width = image.shape[:2]
    scale = new_width / old_width  # Ölçek çarpanı. Örn: 1280 / 5120 = 0.25
    new_height = int(round(old_height * scale))  # Yüksekliği de aynı oranda değiştir; tam sayıya yuvarla

    # Küçültüyorsak (scale < 1) INTER_AREA, büyütüyorsak INTER_LINEAR kullanıyoruz.
    method = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_width, new_height), interpolation=method)


# Resmi ekrana sığacak 1280 piksel genişliğe getir
screen_img = resize_to_width(img, 1280)

# .shape[1] genişlik, .shape[0] yükseklik. Yazdırırken "genişlik x yükseklik" biçiminde yazıyoruz.
print(f"orijinal : {width}x{height}")
print(f"yarım    : {half.shape[1]}x{half.shape[0]}")
print(f"ezik     : {squashed.shape[1]}x{squashed.shape[0]}  (oran bozuk)")
print(f"ekran    : {screen_img.shape[1]}x{screen_img.shape[0]}  (oran korunmuş)")

# cv2.imwrite (image write) bir resmi diske kaydeder. Dosya uzantısına (.jpg) göre format seçer.
# İlk parametre dosya yolu, ikincisi kaydedilecek resimdir.
cv2.imwrite(str(OUTPUT_DIR / "half.jpg"), half)
cv2.imwrite(str(OUTPUT_DIR / "gray.jpg"), gray)
cv2.imwrite(str(OUTPUT_DIR / "squashed.jpg"), squashed)
print(f"kaydedildi: {OUTPUT_DIR}")


# ============================================================
# 3) cv2.imshow: pencerede gösterme
# ============================================================
# imshow(pencere_adı, resim) bir pencere açıp resmi gösterir.
# AMA imshow tek başına pencereyi ekrana ÇİZMEZ. Çizme işi waitKey çağrılınca olur.
# waitKey(ms) "klavyeden tuş bekle" demektir:
#   waitKey(0) -> bir tuşa basılana kadar sonsuza kadar bekler
#   waitKey(1) -> en fazla 1 milisaniye bekler ve devam eder (video döngülerinde kullanılır)
# Dönüş değeri basılan tuşun sayısal kodudur. Hangi tuş olduğunu ord('q') ile karşılaştırırız.

# Orijinal resim 5120 piksel genişliğinde, ekrana sığmaz. imshow resmi otomatik küçültmez,
# kenarlar ekran dışına taşar. Bu yüzden göstermeden önce küçültüyoruz.
cv2.imshow("1 - Renkli (1280 px)", screen_img)
cv2.imshow("2 - Gri", resize_to_width(gray, 1280))
cv2.imshow("3 - Ezik 640x640", squashed)

key = cv2.waitKey(0)  # Bir tuşa basılana kadar bekle; basılan tuşun kodunu key'e koy
print(f"basılan tuş kodu: {key}")

# Alternatif: önce namedWindow ile pencereyi WINDOW_NORMAL modunda oluşturursak
# pencereyi fareyle büyütüp küçültebiliriz. Resmin pikselleri değişmez, sadece gösterim küçülür.
cv2.namedWindow("4 - Orijinal (boyutlanabilir)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("4 - Orijinal (boyutlanabilir)", 1280, 540)  # Pencere boyutu: genişlik, yükseklik
cv2.imshow("4 - Orijinal (boyutlanabilir)", img)
cv2.waitKey(0)

# Açık tüm pencereleri kapat. Programı temiz bitirmek için iyi alışkanlıktır.
cv2.destroyAllWindows()
