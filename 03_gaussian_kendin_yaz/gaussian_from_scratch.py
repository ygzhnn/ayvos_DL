"""
Gaussian Blur'u Kendin Yaz

Çalıştırma (proje kökünden, venv açıkken):
    python 03_gaussian_kendin_yaz/gaussian_from_scratch.py
"""
import time
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "data" / "test.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

KSIZE = 9
SIGMA = 1.5
WORK_WIDTH = 240
SHOW_WINDOWS = True


def show(title, image):
    """Resmi pencerede gösterir, tuşa basılınca kapatır. ESC'ye basılırsa sonrakileri atlar."""
    global SHOW_WINDOWS
    if not SHOW_WINDOWS:
        return
    cv2.imshow(title, image)
    if cv2.waitKey(0) == 27:
        SHOW_WINDOWS = False
    cv2.destroyWindow(title)


# 1) Gaussian kernel'i kendimiz üretelim
def gaussian_kernel_1d(size, sigma):
    """1 boyutlu Gaussian kernel üretir."""
    if size % 2 == 0:
        raise ValueError("size tek sayı olmalı (kernel'in tam ortası olsun)")

    x = np.arange(size) - size // 2

    weights = np.exp(-x ** 2 / (2 * sigma ** 2))  # Ham çan eğrisi değerleri
    return weights / weights.sum()                 # Toplamı 1 yap (normalize)


def gaussian_kernel_2d(size, sigma):
    """2 boyutlu Gaussian kernel üretir."""
    kernel_1d = gaussian_kernel_1d(size, sigma)
    return np.outer(kernel_1d, kernel_1d)


my_kernel_1d = gaussian_kernel_1d(KSIZE, SIGMA)
cv_kernel_1d = cv2.getGaussianKernel(KSIZE, SIGMA).ravel()  # OpenCV'nin kernel'i
print(f"Benim kernel     : {my_kernel_1d.round(4)}")
print(f"OpenCV'nin kernel: {cv_kernel_1d.round(4)}")
print(f"en büyük fark    : {np.abs(my_kernel_1d - cv_kernel_1d).max():.2e}")
print(f"2B kernel toplamı: {gaussian_kernel_2d(KSIZE, SIGMA).sum():.6f}  (1 olmalı)\n")


# 3) Kenar (padding) sorunu
PAD_MODE = "reflect"


# 2a) Konvolüsyon: iç içe for döngüsü (en yavaş, en anlaşılır)
def convolve_loops(image, kernel, pad_mode=PAD_MODE):
    """Her pikseli tek tek gezer, kernel ile çarpıp toplar."""
    height, width = image.shape
    ksize = kernel.shape[0]
    radius = ksize // 2

    padded = np.pad(image, radius, mode=pad_mode)  # Kenarlara r piksellik çerçeve ekle
    output = np.zeros((height, width))

    for y in range(height):          # Her satır
        for x in range(width):       # Her sütun
            total = 0.0
            for i in range(ksize):   # Kernel'in satırları
                for j in range(ksize):  # Kernel'in sütunları
                    total += padded[y + i, x + j] * kernel[i, j]
            output[y, x] = total
    return output


# 2b) Konvolüsyon: NumPy ile vektörel (piksel döngüsü yok)
def convolve_vectorized(image, kernel, pad_mode=PAD_MODE):
    """Piksel piksel gezmek yerine, resmin TAMAMINI kernel'in her elemanı için bir kez kaydırırız."""
    height, width = image.shape
    ksize = kernel.shape[0]
    radius = ksize // 2

    padded = np.pad(image, radius, mode=pad_mode)
    output = np.zeros((height, width))

    for i in range(ksize):
        for j in range(ksize):
            output += kernel[i, j] * padded[i:i + height, j:j + width]
    return output


# 2c) Konvolüsyon: ayrılabilir (önce yatay, sonra dikey)
def convolve_separable(image, kernel_1d, pad_mode=PAD_MODE):
    """2B Gaussian = iki 1B Gaussian'ın çarpımı olduğu için, 2B konvolüsyon yerine iki 1B konvolüsyon yaparız:"""
    height, width = image.shape
    ksize = kernel_1d.shape[0]
    radius = ksize // 2

    padded = np.pad(image, radius, mode=pad_mode)

    horizontal = np.zeros((height + 2 * radius, width))
    for j in range(ksize):
        horizontal += kernel_1d[j] * padded[:, j:j + width]

    # Adım 2: dikey. Şimdi satır yönünde kaydırıyoruz.
    output = np.zeros((height, width))
    for i in range(ksize):
        output += kernel_1d[i] * horizontal[i:i + height, :]
    return output


# Test resmi: gri tonlamalı ve küçük (döngülü yöntem hızlı bitsin)
img = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)  # Gri okuyoruz: tek kanal, işlem basit
if img is None:
    raise FileNotFoundError(f"Görüntü okunamadı: {INPUT_PATH}")
scale = WORK_WIDTH / img.shape[1]
gray = cv2.resize(img, (WORK_WIDTH, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
gray_float = gray.astype(np.float64)
print(f"test resmi: {gray.shape[1]}x{gray.shape[0]}, kernel: {KSIZE}x{KSIZE}, sigma={SIGMA}\n")

kernel_2d = gaussian_kernel_2d(KSIZE, SIGMA)


# 4) Sonuç tablosu: süre ve cv2'ye göre fark
def measure(function, repeat):
    """Fonksiyonu repeat kez çalıştırır, EN KISA süreyi (ms) ve son sonucu döndürür."""
    best = float("inf")
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        elapsed_ms = (time.perf_counter() - start) * 1000
        best = min(best, elapsed_ms)
    return best, result


def to_uint8(float_image):
    """Ondalıklı sonucu 0-255 arası tam sayıya çevirir (yuvarlayıp kırpar)."""
    return np.clip(np.round(float_image), 0, 255).astype(np.uint8)


time_loops, out_loops = measure(lambda: convolve_loops(gray_float, kernel_2d), repeat=1)  # Yavaş, 1 kez yeter
time_vec, out_vec = measure(lambda: convolve_vectorized(gray_float, kernel_2d), repeat=20)
time_sep, out_sep = measure(lambda: convolve_separable(gray_float, my_kernel_1d), repeat=20)
time_cv, out_cv = measure(lambda: cv2.GaussianBlur(gray, (KSIZE, KSIZE), SIGMA), repeat=200)

def mean_diff(result, reference):
    """İki resim arasındaki ortalama mutlak piksel farkı (int16'ya çevirerek, uint8 taşmasından kaçınıyoruz)."""
    return np.abs(to_uint8(result).astype(np.int16) - reference.astype(np.int16)).mean()


print(f"{'yöntem':<28}{'süre (ms)':>12}{'cv2 ile fark':>16}")
print("-" * 56)
print(f"{'(a) iç içe for döngüsü':<28}{time_loops:>12.2f}{mean_diff(out_loops, out_cv):>16.4f}")
print(f"{'(b) NumPy vektörel':<28}{time_vec:>12.2f}{mean_diff(out_vec, out_cv):>16.4f}")
print(f"{'(c) ayrılabilir':<28}{time_sep:>12.2f}{mean_diff(out_sep, out_cv):>16.4f}")
print(f"{'(d) cv2.GaussianBlur':<28}{time_cv:>12.3f}{0.0:>16.4f}")
print(f"\ncv2, döngülü yönteme göre ~{time_loops / time_cv:,.0f} kat, vektörele göre ~{time_vec / time_cv:,.0f} kat hızlı")

print(f"\n(a) ile (b) arasındaki en büyük fark: {np.abs(out_loops - out_vec).max():.2e}")
print(f"(b) ile (c) arasındaki en büyük fark: {np.abs(out_vec - out_sep).max():.2e}")

out_cv_float = cv2.GaussianBlur(gray_float.astype(np.float32), (KSIZE, KSIZE), SIGMA)
print(f"kendi sonucumuz ile cv2 (float32) arasındaki en büyük fark: {np.abs(out_sep - out_cv_float).max():.2e}")


# 3 devam) Kenar modlarını deneyelim: OpenCV'nin varsayılanı hangisi?
print("\nKenar modu karşılaştırması (cv2 float32 varsayılanına göre en büyük fark):")
for mode in ["constant", "edge", "symmetric", "reflect"]:
    mine = convolve_separable(gray_float, my_kernel_1d, pad_mode=mode)
    print(f"  {mode:<10}: {np.abs(mine - out_cv_float).max():.2e}")


# Görsel karşılaştırma
diff_image = np.clip(np.abs(to_uint8(out_sep).astype(np.int16) - out_cv.astype(np.int16)) * 50, 0, 255).astype(np.uint8)
comparison = np.hstack([gray, to_uint8(out_sep), out_cv, diff_image])
cv2.imwrite(str(OUTPUT_DIR / "comparison.png"), comparison)
print(f"\nkaydedildi: {OUTPUT_DIR / 'comparison.png'}  (orijinal | bizim | cv2 | fark x50)")
show("orijinal | bizim (ayrilabilir) | cv2 | fark x50", comparison)
cv2.destroyAllWindows()


# Kendini test et
print(f"\n21x21 kernel, piksel başına çarpma: 2B = {21 * 21}, ayrılabilir = {21 + 21}")
