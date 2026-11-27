"""OpenCV Giriş: cv2.imread, cv2.imshow, cv2.resize"""
from pathlib import Path

import cv2

# Dosya yollarını hazırlıyoruz
ROOT_DIR = Path(__file__).resolve().parent.parent

# "/" işareti Path nesnelerinde klasör/dosya adlarını birleştirir.
INPUT_PATH = ROOT_DIR / "data" / "test.jpg"  # Okuyacağımız resim

# Çıktıları 01_opencv_giris/outputs klasörüne kaydedeceğiz
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Klasörü oluştur. exist_ok=True: "zaten varsa hata verme" demektir.
OUTPUT_DIR.mkdir(exist_ok=True)


# 1) cv2.imread: dosyadan görüntü okuma
img = cv2.imread(str(INPUT_PATH))

if img is None:
    # raise: kasten hata fırlatır ve programı durdurur.
    raise FileNotFoundError(f"Görüntü okunamadı: {INPUT_PATH}")

gray = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)

print(f"renkli shape: {img.shape}  (yükseklik, genişlik, kanal)")
print(f"gri shape   : {gray.shape}  (kanal boyutu yok)")

print(f"dtype       : {img.dtype}  (0-255)")

y, x = 100, 200
b, g, r = img[y, x]
print(f"({x},{y}) pikseli -> B={b} G={g} R={r}")


# 2) cv2.resize: boyut değiştirme
height, width = img.shape[:2]

half = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

squashed = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)


def resize_to_width(image, new_width):
    """Resmin oranını bozmadan genişliğini new_width yapar."""
    old_height, old_width = image.shape[:2]
    scale = new_width / old_width  # Ölçek çarpanı. Örn: 1280 / 5120 = 0.25
    new_height = int(round(old_height * scale))

    method = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_width, new_height), interpolation=method)


# Resmi ekrana sığacak 1280 piksel genişliğe getir
screen_img = resize_to_width(img, 1280)

print(f"orijinal : {width}x{height}")
print(f"yarım    : {half.shape[1]}x{half.shape[0]}")
print(f"ezik     : {squashed.shape[1]}x{squashed.shape[0]}  (oran bozuk)")
print(f"ekran    : {screen_img.shape[1]}x{screen_img.shape[0]}  (oran korunmuş)")

cv2.imwrite(str(OUTPUT_DIR / "half.jpg"), half)
cv2.imwrite(str(OUTPUT_DIR / "gray.jpg"), gray)
cv2.imwrite(str(OUTPUT_DIR / "squashed.jpg"), squashed)
print(f"kaydedildi: {OUTPUT_DIR}")


# 3) cv2.imshow: pencerede gösterme

cv2.imshow("1 - Renkli (1280 px)", screen_img)
cv2.imshow("2 - Gri", resize_to_width(gray, 1280))
cv2.imshow("3 - Ezik 640x640", squashed)

key = cv2.waitKey(0)
print(f"basılan tuş kodu: {key}")

cv2.namedWindow("4 - Orijinal (boyutlanabilir)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("4 - Orijinal (boyutlanabilir)", 1280, 540)  # Pencere boyutu: genişlik, yükseklik
cv2.imshow("4 - Orijinal (boyutlanabilir)", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
