"""OpenCV Giriş (devam): imread/imshow/resize dışındaki temel fonksiyonlar"""
from pathlib import Path  # Dosya yolları için

import cv2  # OpenCV: görüntü işleme
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "data" / "test.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

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
    global SHOW_WINDOWS
    if not SHOW_WINDOWS:
        return
    cv2.imshow(title, fit_to_screen(image))
    if cv2.waitKey(0) == 27:  # 27, ESC tuşunun kodudur
        SHOW_WINDOWS = False
    cv2.destroyWindow(title)  # Sadece bu pencereyi kapat


# Resmi oku ve çalışma boyutuna getir
img = cv2.imread(str(INPUT_PATH))
if img is None:
    raise FileNotFoundError(f"Görüntü okunamadı: {INPUT_PATH}")

img = fit_to_screen(img, 1280)
height, width = img.shape[:2]
print(f"çalışma boyutu: {width}x{height}")


# 1) cv2.cvtColor: renk uzayı dönüşümü
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 3 kanalı 1 kanala indirir
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

show("1a - RGB'yi imshow ile gostermek (yanlis renkler)", rgb)

gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
show("1b - Renkli | Gri", np.hstack([img, gray_3ch]))

lower = np.array([5, 60, 40])
upper = np.array([25, 255, 255])  # Üst sınır: [H, S, V]

mask = cv2.inRange(hsv, lower, upper)

print(f"ten/turuncu piksel oranı: %{(mask > 0).mean() * 100:.1f}")

selected = cv2.bitwise_and(img, img, mask=mask)
show("1c - HSV ile ten/turuncu tonlar", selected)


# 2) cv2.split / cv2.merge: kanallar
blue, green, red = cv2.split(img)  # BGR sırasında döner
print(f"B kanalı ortalaması: {blue.mean():.1f}, G: {green.mean():.1f}, R: {red.mean():.1f}")
show("2a - B | G | R kanallari (her biri gri)", np.hstack([blue, green, red]))

zeros = np.zeros_like(blue)
only_red = cv2.merge([zeros, zeros, red])  # Sıra yine B, G, R
show("2b - Sadece kirmizi kanal", only_red)


# 3) ROI / kırpma: NumPy dilimleme
y1, y2 = height // 4, 3 * height // 4  # Yüksekliğin %25'inden %75'ine
x1, x2 = width // 4, 3 * width // 4    # Genişliğin %25'inden %75'ine
roi = img[y1:y2, x1:x2]

copy_img = img.copy()

copy_img[y1:y2, x1:x2] = 255 - copy_img[y1:y2, x1:x2]
show("3 - Ortasi negatif (ROI)", copy_img)


# 4) cv2.flip / cv2.rotate
flipped_h = cv2.flip(img, 1)
flipped_v = cv2.flip(img, 0)

rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
print(f"90° döndürünce boyut: {rotated_90.shape[1]}x{rotated_90.shape[0]} (en ve boy yer değiştirdi)")
show("4a - Yatay | Dikey flip", np.hstack([flipped_h, flipped_v]))
show("4b - 90 derece saat yonu", rotated_90)


# 5) cv2.warpAffine: istenen açıyla döndürme ve kaydırma

center = (width / 2, height / 2)  # Resmin ortası
rotation_matrix = cv2.getRotationMatrix2D(center, 30, 1.0)  # 30 derece, ölçek değişmesin
rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
# Köşeler çerçevenin dışına taşar, boş kalan yerler siyah olur.

shift_matrix = np.float32([[1, 0, 100],
                           [0, 1, 50]])
shifted = cv2.warpAffine(img, shift_matrix, (width, height))
show("5 - 30 derece dondurme | Kaydirma", np.hstack([rotated, shifted]))


# 6) cv2.copyMakeBorder: kenar ekleme
border = 60  # Her kenara eklenecek piksel sayısı
# Üç parametre: üst, alt, sol, sağ kenar kalınlığı
const_border = cv2.copyMakeBorder(img, border, border, border, border,
                                  cv2.BORDER_CONSTANT, value=(0, 200, 255))  # Tek renkle doldur (BGR: sarı-turuncu)
reflect_border = cv2.copyMakeBorder(img, border, border, border, border,
                                    cv2.BORDER_REFLECT)    # Kenardaki pikselleri ayna gibi yansıt
replicate_border = cv2.copyMakeBorder(img, border, border, border, border,
                                      cv2.BORDER_REPLICATE)  # En kenardaki pikseli tekrar et
show("6 - CONSTANT | REFLECT | REPLICATE", np.hstack([const_border, reflect_border, replicate_border]))


# 7) Çizim: line, rectangle, circle, putText
drawing = img.copy()

# line(resim, başlangıç_noktası, bitiş_noktası, renk, kalınlık)
cv2.line(drawing, (0, 0), (width - 1, height - 1), (255, 0, 0), 3)

# rectangle(resim, sol_üst_köşe, sağ_alt_köşe, renk, kalınlık)
cv2.rectangle(drawing, (x1, y1), (x2, y2), (0, 255, 0), 4)

cv2.rectangle(drawing, (20, 20), (220, 90), (0, 0, 0), -1)

# circle(resim, merkez, yarıçap, renk, kalınlık)
cv2.circle(drawing, (width // 2, height // 2), 80, (0, 0, 255), 5)

cv2.putText(drawing, "OpenCV", (35, 70), cv2.FONT_HERSHEY_SIMPLEX,
            1.4, (255, 255, 255), 2, cv2.LINE_AA)
show("7 - Cizim fonksiyonlari", drawing)


# 8) cv2.addWeighted: iki görüntüyü karıştırma
overlay = img.copy()
cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)  # Üstte siyah şerit
translucent = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
cv2.putText(translucent, f"{width}x{height}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (255, 255, 255), 2, cv2.LINE_AA)
show("8 - addWeighted ile yari saydam panel", translucent)

a = np.array([[250]], dtype=np.uint8)
print(f"numpy 250+10 = {(a + 10)[0, 0]}  (taşma, başa sarar)")
print(f"cv2.add      = {cv2.add(a, 10)[0, 0]}  (255'te kırpar)")


# 9) cv2.imwrite: kalite ayarıyla kaydetme
cv2.imwrite(str(OUTPUT_DIR / "drawing_q95.jpg"), drawing, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite(str(OUTPUT_DIR / "drawing_q10.jpg"), drawing, [cv2.IMWRITE_JPEG_QUALITY, 10])
cv2.imwrite(str(OUTPUT_DIR / "drawing.png"), drawing)

# Dosya boyutlarını KB olarak yazdır ve farkı gör
for file_name in ["drawing_q95.jpg", "drawing_q10.jpg", "drawing.png"]:
    size_kb = (OUTPUT_DIR / file_name).stat().st_size / 1024  # Bayt cinsinden boyutu KB'a çevir
    print(f"{file_name:16s}: {size_kb:7.1f} KB")

cv2.destroyAllWindows()
