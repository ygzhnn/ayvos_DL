"""
OpenCV ile Gaussian Blur: cv2.GaussianBlur() nasıl çalışır, parametreleri ne yapar?

Çalıştırma (proje kökünden, venv açıkken):
    python 02_gaussian_blur/gaussian_blur.py
"""
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "data" / "test.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SHOW_WINDOWS = True


def fit_to_screen(image, max_width=1400):
    """Resim ekrana sığmayacak kadar genişse, oranını bozmadan küçültür."""
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    return cv2.resize(image, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)


def show(title, image):
    """Resmi pencerede gösterir, tuşa basılınca kapatır. ESC'ye basılırsa sonrakileri atlar."""
    global SHOW_WINDOWS
    if not SHOW_WINDOWS:
        return
    cv2.imshow(title, fit_to_screen(image))
    if cv2.waitKey(0) == 27:  # 27 = ESC tuşu
        SHOW_WINDOWS = False
    cv2.destroyWindow(title)


def add_label(image, text):
    """Resmin sol üstüne okunaklı bir etiket yazar (kopya üzerinde çalışır)."""
    labeled = image.copy()  # Orijinali bozmamak için kopya
    cv2.rectangle(labeled, (0, 0), (200, 32), (0, 0, 0), -1)  # Siyah zemin
    cv2.putText(labeled, text, (6, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return labeled


# Resmi oku ve çalışma boyutuna getir
img = cv2.imread(str(INPUT_PATH))
if img is None:
    raise FileNotFoundError(f"Görüntü okunamadı: {INPUT_PATH}")

img = fit_to_screen(img, 480)
print(f"çalışma boyutu: {img.shape[1]}x{img.shape[0]}")


# 1) Kernel nedir?
kernel_1d = cv2.getGaussianKernel(5, 1.0)  # (5 satır, 1 sütun) bir tablo döner
print("\n5 uzunluğunda Gaussian kernel (sigma=1):")
print(kernel_1d.ravel().round(4))

kernel_2d = np.outer(kernel_1d, kernel_1d)
print("\n5x5 Gaussian kernel (ortası en büyük, kenarlara doğru küçülüyor):")
print(kernel_2d.round(4))

print(f"kernel toplamı: {kernel_2d.sum():.4f}")


# 2) cv2.GaussianBlur: temel kullanım
blurred = cv2.GaussianBlur(img, (9, 9), 2)
show("1 - Orijinal | GaussianBlur (9x9, sigma=2)", np.hstack([img, blurred]))


# 3) ksize ve sigma kombinasyonlarını yan yana karşılaştırma (3x3 ızgara)
ksizes = [3, 9, 21]
sigmas = [1, 3, 8]

rows = []
for k in ksizes:
    row_images = []
    for s in sigmas:
        result = cv2.GaussianBlur(img, (k, k), s)
        row_images.append(add_label(result, f"ksize={k} sigma={s}"))
    rows.append(np.hstack(row_images))  # Bu satırdaki 3 resmi yatay birleştir
grid = np.vstack(rows)
show("2 - Izgara: satir = ksize (3, 9, 21), sutun = sigma (1, 3, 8)", grid)
cv2.imwrite(str(OUTPUT_DIR / "grid.jpg"), grid)


# 4) Sadece sigma vermek: ksize=(0, 0)
auto = cv2.GaussianBlur(img, (0, 0), 3)
manual = cv2.GaussianBlur(img, (19, 19), 3)  # 6 * 3 + 1 = 19: aynı sonucu bekliyoruz

diff = np.abs(auto.astype(np.int16) - manual.astype(np.int16))
print(f"\nksize=(0,0) ile (19,19) arasındaki en büyük piksel farkı: {diff.max()}")
show("3 - ksize=(0,0) sigma=3 | ksize=(19,19) sigma=3", np.hstack([auto, manual]))

only_ksize = cv2.GaussianBlur(img, (9, 9), 0)
print("ksize=(9,9), sigma=0 -> sigma otomatik hesaplanır:", round(0.3 * ((9 - 1) * 0.5 - 1) + 0.8, 2))


# 5) sigmaX ve sigmaY: yönlü blur
horizontal = cv2.GaussianBlur(img, (0, 0), sigmaX=8, sigmaY=0.1)  # Sadece yatayda bulanık
vertical = cv2.GaussianBlur(img, (0, 0), sigmaX=0.1, sigmaY=8)    # Sadece dikeyde bulanık
show("4 - Yatay blur | Dikey blur (hareket bulanıklığı gibi)",
     np.hstack([add_label(horizontal, "sigmaX=8 sigmaY~0"), add_label(vertical, "sigmaX~0 sigmaY=8")]))


# 6) Blur ne işe yarar? Gürültü azaltma
rng = np.random.default_rng(42)
noise = rng.normal(0, 25, img.shape)
noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

denoised = cv2.GaussianBlur(noisy, (5, 5), 1.5)
show("5 - Gurultulu | Gaussian blur ile temizlenmis",
     np.hstack([add_label(noisy, "gurultulu"), add_label(denoised, "temizlenmis")]))


# 7) Kenar modu: resmin kenarında ne olur?
default_border = cv2.GaussianBlur(img, (21, 21), 8)  # Varsayılan: BORDER_REFLECT_101
constant_border = cv2.GaussianBlur(img, (21, 21), 8, borderType=cv2.BORDER_CONSTANT)  # Dışarıyı siyah (0) say
show("6 - Varsayilan kenar | BORDER_CONSTANT (kenarlar koyulasir)",
     np.hstack([add_label(default_border, "varsayilan"), add_label(constant_border, "CONSTANT")]))

cv2.imwrite(str(OUTPUT_DIR / "blurred.jpg"), blurred)
cv2.imwrite(str(OUTPUT_DIR / "denoised.jpg"), denoised)
print(f"\nkaydedildi: {OUTPUT_DIR}")
cv2.destroyAllWindows()
