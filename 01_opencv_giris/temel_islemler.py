"""
OpenCV Giriş (devam): imread/imshow/resize dışındaki temel fonksiyonlar

  1) cvtColor        : renk uzayı dönüşümü (BGR, RGB, GRAY, HSV)
  2) split / merge   : kanalları ayırma ve birleştirme
  3) ROI / kırpma    : NumPy dilimleme ile bölge seçme
  4) flip / rotate   : çevirme ve 90° döndürme
  5) warpAffine      : istenen açıyla döndürme ve kaydırma
  6) copyMakeBorder  : kenar ekleme
  7) Çizim           : line, rectangle, circle, putText
  8) addWeighted     : iki görüntüyü karıştırma
  9) imwrite         : kalite ayarıyla kaydetme


Her pencerede bir tuşa basınca sonrakine geçer, ESC ile tümünü atlar.

Hatırlatma: renkli bir resim, her pikseli 3 sayıdan (Mavi, Yeşil, Kırmızı) oluşan,
0-255 arası sayılarla dolu bir tablodur. Buradaki işlemlerin hepsi bu tabloyu
değiştirmekten ibarettir.
"""
from pathlib import Path  # Dosya yolları için

import cv2  # OpenCV: görüntü işleme
import numpy as np  # NumPy: sayı tablolarıyla (diziler) hızlı çalışma. Kodda hep "np" diye yazılır.

# Proje ana klasörünü, girdi resmini ve çıktı klasörünü bul (açıklaması opencv_giris.py'de var)
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "data" / "test.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# True iken pencereler açılır. ESC'ye basınca False olur ve kalan pencereler atlanır.
SHOW_WINDOWS = True


def fit_to_screen(image, max_width=1000):
    """Resim ekrana sığmayacak kadar genişse, oranını bozmadan küçültür."""
    height, width = image.shape[:2]
    if width <= max_width:
        return image  # Zaten yeterince küçük, dokunma
    scale = max_width / width  # Ölçek çarpanı
    return cv2.resize(image, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)


def show(title, image):
    """Resmi bir pencerede gösterir, tuşa basılınca kapatır. ESC'ye basılırsa sonrakileri atlar."""
    global SHOW_WINDOWS  # Fonksiyonun dışındaki değişkeni değiştireceğimizi söylüyoruz
    if not SHOW_WINDOWS:
        return  # Pencereler kapatıldıysa hiçbir şey yapma
    cv2.imshow(title, fit_to_screen(image))
    if cv2.waitKey(0) == 27:  # 27, ESC tuşunun kodudur
        SHOW_WINDOWS = False
    cv2.destroyWindow(title)  # Sadece bu pencereyi kapat


# ------------------------------------------------------------
# Resmi oku ve çalışma boyutuna getir
# ------------------------------------------------------------
img = cv2.imread(str(INPUT_PATH))
if img is None:  # imread dosyayı bulamazsa hata vermez, None döndürür; o yüzden kontrol ediyoruz
    raise FileNotFoundError(f"Görüntü okunamadı: {INPUT_PATH}")

# 5120 piksellik resimle çalışmak yavaş ve gereksiz. İşlemleri 1280 piksel genişlikte yapalım.
img = fit_to_screen(img, 1280)
height, width = img.shape[:2]
print(f"çalışma boyutu: {width}x{height}")


# ============================================================
# 1) cv2.cvtColor: renk uzayı dönüşümü
# ============================================================
# "Renk uzayı", rengi sayılarla ifade etme yöntemidir. Aynı resim farklı yöntemlerle gösterilebilir:
#   BGR  : Mavi-Yeşil-Kırmızı miktarı (OpenCV'nin varsayılanı)
#   RGB  : Kırmızı-Yeşil-Mavi (Matplotlib ve çoğu başka kütüphane bunu bekler)
#   GRAY : Sadece parlaklık, tek sayı (siyah-beyaz fotoğraf gibi)
#   HSV  : Hue (ton/renk cinsi), Saturation (renk canlılığı), Value (parlaklık)
# cv2.COLOR_<ESKİ>2<YENİ> sabitleri hangi dönüşümü yapacağımızı söyler.
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # Mavi ile kırmızı kanalın yerini değiştirir
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 3 kanalı 1 kanala indirir
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# RGB resmi OpenCV'nin imshow'u ile gösterirsek, imshow BGR beklediği için
# kırmızı ile mavi yer değiştirmiş görünür (gökyüzü turuncu olabilir).
show("1a - RGB'yi imshow ile gostermek (yanlis renkler)", rgb)

# Gri resim tek kanallıdır, renkli resim 3 kanallıdır. İkisini yan yana koyabilmek
# (np.hstack = yatay birleştir) için gri resmi tekrar 3 kanala çeviriyoruz.
# Renkler geri gelmez, sadece her piksel için aynı sayı 3 kez tekrarlanır.
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
show("1b - Renkli | Gri", np.hstack([img, gray_3ch]))

# HSV, renge göre seçim yapmayı çok kolaylaştırır. BGR'de "turuncu" demek için üç kanalı
# birlikte düşünmek gerekir, HSV'de ise renk cinsi tek bir kanalda (H) durur.
# OpenCV'de H değeri 0-179 arasıdır (derece/2), S ve V 0-255 arasıdır.
# Yaklaşık H değerleri: kırmızı 0-10 ve 170-179, turuncu/ten 5-25, yeşil 35-85, mavi 100-125, mor 130-160
lower = np.array([5, 60, 40])    # Alt sınır: [H, S, V]. S ve V'yi sınırlamak gri/siyah/soluk pikselleri eler
upper = np.array([25, 255, 255])  # Üst sınır: [H, S, V]

# inRange: her piksele bakar. Üç değer de alt-üst sınır arasındaysa 255 (beyaz), değilse 0 (siyah) yazar.
# Sonuç, tek kanallı bir "maske"dir: seçilen yerler beyaz, diğerleri siyah.
mask = cv2.inRange(hsv, lower, upper)

# (mask > 0) her piksel için True/False üretir. .mean() bunların ortalamasını yani oranını verir.
print(f"ten/turuncu piksel oranı: %{(mask > 0).mean() * 100:.1f}")

# bitwise_and ile maskeyi resme uygularız: maskenin beyaz olduğu yerler korunur, kalanı siyah olur.
# (img, img: aynı resmi kendisiyle "ve"liyoruz, asıl iş mask parametresinde.)
selected = cv2.bitwise_and(img, img, mask=mask)
show("1c - HSV ile ten/turuncu tonlar", selected)
# Dene: lower/upper değerlerini yeşil ya da mavi aralığına çevirip sonuca bak


# ============================================================
# 2) cv2.split / cv2.merge: kanallar
# ============================================================
# split, 3 kanallı resmi 3 ayrı gri resme böler. Her biri o rengin "ne kadar var" haritasıdır.
# Parlak = o renk çok var, koyu = az var.
blue, green, red = cv2.split(img)  # BGR sırasında döner
print(f"B kanalı ortalaması: {blue.mean():.1f}, G: {green.mean():.1f}, R: {red.mean():.1f}")
show("2a - B | G | R kanallari (her biri gri)", np.hstack([blue, green, red]))

# merge, ayrı kanalları tekrar birleştirir. Burada mavi ve yeşili sıfırlayıp
# (np.zeros_like: aynı boyutta, içi sıfır dolu tablo) sadece kırmızıyı bırakıyoruz.
zeros = np.zeros_like(blue)
only_red = cv2.merge([zeros, zeros, red])  # Sıra yine B, G, R
show("2b - Sadece kirmizi kanal", only_red)

# Not: img[:, :, 2] ifadesi de kırmızı kanalı verir ve split'ten daha hızlıdır,
# çünkü kopya oluşturmaz.


# ============================================================
# 3) ROI / kırpma: NumPy dilimleme
# ============================================================
# ROI = Region Of Interest (ilgilendiğimiz bölge). Resim bir tablo olduğu için
# tablonun bir parçasını almak resmi kırpmak demektir: img[y1:y2, x1:x2]
# Önce satır aralığı (y, yukarıdan aşağı), sonra sütun aralığı (x, soldan sağa).
# "//" tam sayı bölmesidir (sonuç ondalıklı olmaz).
y1, y2 = height // 4, 3 * height // 4  # Yüksekliğin %25'inden %75'ine
x1, x2 = width // 4, 3 * width // 4    # Genişliğin %25'inden %75'ine
roi = img[y1:y2, x1:x2]

# DİKKAT: Dilimleme KOPYA oluşturmaz, orijinal resmin aynı belleğine bakan bir GÖRÜNÜM döndürür.
# roi'yi değiştirirsen img de değişir. Bağımsız kopya için .copy() kullanılır.
copy_img = img.copy()

# 255 - piksel değeri = negatif (ters renk). Sadece ortadaki bölgeye uyguluyoruz.
copy_img[y1:y2, x1:x2] = 255 - copy_img[y1:y2, x1:x2]
show("3 - Ortasi negatif (ROI)", copy_img)


# ============================================================
# 4) cv2.flip / cv2.rotate
# ============================================================
# flip: resmi aynalar. İkinci parametre yönü belirler:
#   1 -> yatay çevir (sol-sağ yer değiştirir, ayna gibi)
#   0 -> dikey çevir (baş aşağı)
#  -1 -> ikisi birden
flipped_h = cv2.flip(img, 1)
flipped_v = cv2.flip(img, 0)

# rotate: sadece 90'ın katları için hızlı döndürme.
# Diğer seçenekler: cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180
rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
print(f"90° döndürünce boyut: {rotated_90.shape[1]}x{rotated_90.shape[0]} (en ve boy yer değiştirdi)")
show("4a - Yatay | Dikey flip", np.hstack([flipped_h, flipped_v]))
show("4b - 90 derece saat yonu", rotated_90)


# ============================================================
# 5) cv2.warpAffine: istenen açıyla döndürme ve kaydırma
# ============================================================
# "Affine dönüşüm", resmi döndürme, kaydırma, ölçekleme gibi doğruları bozmayan
# dönüşümlerin genel adıdır. 2x3'lük bir sayı tablosu (matris) ile tanımlanır.
# Her yeni piksel konumu bu matrisle eski konumdan hesaplanır.

# Döndürme matrisini OpenCV bizim için hazırlar:
# (dönme merkezi, açı, ölçek). Açı pozitifse saat yönünün TERSİNE döner.
center = (width / 2, height / 2)  # Resmin ortası
rotation_matrix = cv2.getRotationMatrix2D(center, 30, 1.0)  # 30 derece, ölçek değişmesin
# warpAffine(resim, matris, (çıktı_genişliği, çıktı_yüksekliği)). Boyut yine (genişlik, yükseklik) sırasıyla!
rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
# Köşeler çerçevenin dışına taşar, boş kalan yerler siyah olur.

# Kaydırma matrisini kendimiz yazarız. Son sütun kaydırma miktarıdır:
# x yönünde +100 piksel (sağa), y yönünde +50 piksel (aşağı).
shift_matrix = np.float32([[1, 0, 100],
                           [0, 1, 50]])
shifted = cv2.warpAffine(img, shift_matrix, (width, height))
show("5 - 30 derece dondurme | Kaydirma", np.hstack([rotated, shifted]))


# ============================================================
# 6) cv2.copyMakeBorder: kenar ekleme
# ============================================================
# Resmin dört kenarına (üst, alt, sol, sağ) piksel ekler. Eklenen pikselleri doldurma yöntemini biz seçeriz.
border = 60  # Her kenara eklenecek piksel sayısı
# Üç parametre: üst, alt, sol, sağ kenar kalınlığı
const_border = cv2.copyMakeBorder(img, border, border, border, border,
                                  cv2.BORDER_CONSTANT, value=(0, 200, 255))  # Tek renkle doldur (BGR: sarı-turuncu)
reflect_border = cv2.copyMakeBorder(img, border, border, border, border,
                                    cv2.BORDER_REFLECT)    # Kenardaki pikselleri ayna gibi yansıt
replicate_border = cv2.copyMakeBorder(img, border, border, border, border,
                                      cv2.BORDER_REPLICATE)  # En kenardaki pikseli tekrar et
# Bu kenar modları ileride blur gibi filtrelerde önem kazanacak: filtre resmin kenarında
# kendi dışına taştığında, olmayan piksellerin yerine ne konacağını bunlar belirler.
show("6 - CONSTANT | REFLECT | REPLICATE", np.hstack([const_border, reflect_border, replicate_border]))


# ============================================================
# 7) Çizim: line, rectangle, circle, putText
# ============================================================
# Çizim fonksiyonları resmi YERİNDE değiştirir (yeni resim döndürmez, aynı resmin üstüne çizer).
# Orijinalin bozulmaması için önce kopyasını alıyoruz.
# Renkler BGR sırasındadır: (255, 0, 0) mavi, (0, 255, 0) yeşil, (0, 0, 255) kırmızı.
# Koordinatlar (x, y) şeklindedir. Dilimlemede [y, x] idi, burada tersi. Karıştırma!
drawing = img.copy()

# line(resim, başlangıç_noktası, bitiş_noktası, renk, kalınlık)
cv2.line(drawing, (0, 0), (width - 1, height - 1), (255, 0, 0), 3)

# rectangle(resim, sol_üst_köşe, sağ_alt_köşe, renk, kalınlık)
cv2.rectangle(drawing, (x1, y1), (x2, y2), (0, 255, 0), 4)

# Kalınlığa -1 verirsek şeklin içini tamamen boyar. Yazının arkasına zemin yapmak için kullanışlı.
cv2.rectangle(drawing, (20, 20), (220, 90), (0, 0, 0), -1)

# circle(resim, merkez, yarıçap, renk, kalınlık)
cv2.circle(drawing, (width // 2, height // 2), 80, (0, 0, 255), 5)

# putText(resim, yazı, sol_alt_başlangıç_noktası, yazı_tipi, boyut, renk, kalınlık, çizgi_türü)
# LINE_AA yazının kenarlarını yumuşatır, daha düzgün görünür.
cv2.putText(drawing, "OpenCV", (35, 70), cv2.FONT_HERSHEY_SIMPLEX,
            1.4, (255, 255, 255), 2, cv2.LINE_AA)
# Not: putText Türkçe karakterleri (ş, ğ, ı) basamaz, onların yerine "?" çıkar.
show("7 - Cizim fonksiyonlari", drawing)


# ============================================================
# 8) cv2.addWeighted: iki görüntüyü karıştırma
# ============================================================
# Formül: sonuç = resim1 * alfa + resim2 * beta + gamma
# Sonuç 255'i aşarsa otomatik 255'te kesilir (taşma olmaz).
# Yarı saydam bilgi paneli yapmanın standart yoludur: önce siyah bir şerit çizilmiş bir katman
# hazırlarız, sonra orijinalle karıştırırız.
overlay = img.copy()
cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)  # Üstte siyah şerit
# Siyah katmanın %60'ı ile orijinalin %40'ını karıştır. Şeridin altındaki görüntü hafifçe görünür kalır.
translucent = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
cv2.putText(translucent, f"{width}x{height}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (255, 255, 255), 2, cv2.LINE_AA)
show("8 - addWeighted ile yari saydam panel", translucent)

# Taşma (overflow) farkı: NumPy'da uint8 sayılar 255'i geçince başa sarar (250 + 10 = 4),
# cv2.add ise 255'te durur. Parlaklık artırırken bu fark büyük hataya yol açar.
a = np.array([[250]], dtype=np.uint8)
print(f"numpy 250+10 = {(a + 10)[0, 0]}  (taşma, başa sarar)")
print(f"cv2.add      = {cv2.add(a, 10)[0, 0]}  (255'te kırpar)")


# ============================================================
# 9) cv2.imwrite: kalite ayarıyla kaydetme
# ============================================================
# Dosya uzantısı formatı belirler. JPEG "kayıplı" sıkıştırır: dosya küçülür ama kalite düşer.
# Kalite 0-100 arasıdır. PNG ise "kayıpsızdır": kalite aynı kalır ama dosya büyüktür.
# Üçüncü parametre bir ayar listesidir: [ayar_adı, değeri]
cv2.imwrite(str(OUTPUT_DIR / "drawing_q95.jpg"), drawing, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite(str(OUTPUT_DIR / "drawing_q10.jpg"), drawing, [cv2.IMWRITE_JPEG_QUALITY, 10])
cv2.imwrite(str(OUTPUT_DIR / "drawing.png"), drawing)

# Dosya boyutlarını KB olarak yazdır ve farkı gör
for file_name in ["drawing_q95.jpg", "drawing_q10.jpg", "drawing.png"]:
    size_kb = (OUTPUT_DIR / file_name).stat().st_size / 1024  # Bayt cinsinden boyutu KB'a çevir
    print(f"{file_name:16s}: {size_kb:7.1f} KB")
# imwrite başarılıysa True döndürür. Klasör yoksa hata vermeden False döndürür.

cv2.destroyAllWindows()
