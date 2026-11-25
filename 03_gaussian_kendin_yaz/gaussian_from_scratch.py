"""
Gaussian Blur'u Kendin Yaz

Bu dosyada cv2.GaussianBlur'un yaptığı işi NumPy ile kendimiz yazıyoruz:
  1) gaussian_kernel(size, sigma): kernel'i formülden üret, cv2.getGaussianKernel ile karşılaştır
  2) Konvolüsyonu üç şekilde yaz:
       (a) iç içe for döngüsü    (en yavaş, en anlaşılır)
       (b) NumPy ile vektörel    (döngü yok, kaydırılmış dilimler)
       (c) ayrılabilir           (önce yatay, sonra dikey 1 boyutlu kernel)
  3) Kenar (padding) sorunu: resmin kenarında kernel dışarı taşınca ne olur?
  4) Sonuç tablosu: 4 yöntemin süresi ve cv2'ye göre ortalama piksel farkı

Çalıştırma (proje kökünden, venv açıkken):
    python 03_gaussian_kendin_yaz/gaussian_from_scratch.py

Konvolüsyon nedir? Kernel'i resmin her pikselinin üstüne yerleştiririz. Kernel'deki her ağırlığı,
altına denk gelen pikselin değeriyle çarpar ve hepsini toplarız. Çıkan sayı o pikselin yeni değeridir.
"""
import time
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "data" / "test.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------- Ayarlar (istediğin gibi değiştirebilirsin) ----------------
KSIZE = 9        # Kernel boyutu (tek sayı). Döngülü yöntem yavaş olduğu için 9 ile başlıyoruz.
SIGMA = 1.5      # Çanın genişliği. ksize ≈ 6*sigma + 1 kuralına yakın: 6*1.5+1 = 10 -> 9
WORK_WIDTH = 240  # Döngülü yöntem çok yavaş olduğu için resmi küçük tutuyoruz.
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


# ============================================================
# 1) Gaussian kernel'i kendimiz üretelim
# ============================================================
def gaussian_kernel_1d(size, sigma):
    """
    1 boyutlu Gaussian kernel üretir.
    Formül: ağırlık(x) = exp(-x² / 2σ²), sonra toplamı 1 olacak şekilde normalize et.
    """
    if size % 2 == 0:  # % kalanı verir. Çift sayıların 2'ye bölümünden kalan 0'dır.
        raise ValueError("size tek sayı olmalı (kernel'in tam ortası olsun)")

    # x konumları: merkez 0, yanlara doğru -1, -2, ... ve +1, +2, ...
    # np.arange(size) 0,1,2,...,size-1 verir. size // 2 çıkarınca merkez 0 olur.
    # Örn: size=5 -> [0,1,2,3,4] - 2 = [-2,-1,0,1,2]
    x = np.arange(size) - size // 2

    weights = np.exp(-x ** 2 / (2 * sigma ** 2))  # Ham çan eğrisi değerleri
    return weights / weights.sum()                 # Toplamı 1 yap (normalize)


def gaussian_kernel_2d(size, sigma):
    """
    2 boyutlu Gaussian kernel üretir.
    2B Gaussian, iki 1B Gaussian'ın çarpımıdır: g(x, y) = g(x) * g(y).
    np.outer (dış çarpım): a ve b iki liste ise sonuç tablosunun [i, j] elemanı a[i] * b[j].
    """
    kernel_1d = gaussian_kernel_1d(size, sigma)
    return np.outer(kernel_1d, kernel_1d)


my_kernel_1d = gaussian_kernel_1d(KSIZE, SIGMA)
cv_kernel_1d = cv2.getGaussianKernel(KSIZE, SIGMA).ravel()  # OpenCV'nin kernel'i
print(f"Benim kernel     : {my_kernel_1d.round(4)}")
print(f"OpenCV'nin kernel: {cv_kernel_1d.round(4)}")
# np.abs(...).max(): iki kernel arasındaki en büyük fark. ~1e-16 çıkarsa aynı demektir
# (bilgisayarlar ondalık sayıları tam saklayamadığı için sıfır yerine çok küçük bir sayı çıkar).
print(f"en büyük fark    : {np.abs(my_kernel_1d - cv_kernel_1d).max():.2e}")
print(f"2B kernel toplamı: {gaussian_kernel_2d(KSIZE, SIGMA).sum():.6f}  (1 olmalı)\n")


# ============================================================
# 3) Kenar (padding) sorunu
# ============================================================
# Kernel resmin en kenarındaki piksele geldiğinde bir kısmı resmin DIŞINA taşar.
# Orada piksel yok. Çözüm: resmin etrafına yapay bir çerçeve (padding) eklemek.
# np.pad(resim, r, mode=...) bunu yapar. r = kernel yarıçapı = size // 2. Mod seçenekleri:
#   "constant"  -> dışarıyı sıfırla doldur (kenarlar koyulaşır)
#   "edge"      -> en kenardaki pikseli tekrar et   (OpenCV: BORDER_REPLICATE)
#   "symmetric" -> kenarı da dahil ederek ayna gibi yansıt (OpenCV: BORDER_REFLECT)
#   "reflect"   -> kenar pikseli HARİÇ ayna gibi yansıt   (OpenCV: BORDER_REFLECT_101)
# OpenCV'nin VARSAYILAN kenar modu BORDER_REFLECT_101'dir, yani NumPy'daki "reflect".
# Aşağıda bunu deneyle kanıtlayacağız.
PAD_MODE = "reflect"


# ============================================================
# 2a) Konvolüsyon: iç içe for döngüsü (en yavaş, en anlaşılır)
# ============================================================
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
                    # padded[y + i, x + j]: padded resimde (y, x) pikselini merkez alan
                    # pencerenin i'inci satır, j'inci sütunundaki piksel
                    total += padded[y + i, x + j] * kernel[i, j]
            output[y, x] = total
    return output
# Not: Gerçek "konvolüsyon" kernel'i ters çevirmeyi de içerir. Burada çevirmedik (buna "korelasyon"
# denir). Gaussian kernel simetrik olduğu için ters çevirince aynı kalır, sonuç değişmez.


# ============================================================
# 2b) Konvolüsyon: NumPy ile vektörel (piksel döngüsü yok)
# ============================================================
def convolve_vectorized(image, kernel, pad_mode=PAD_MODE):
    """
    Piksel piksel gezmek yerine, resmin TAMAMINI kernel'in her elemanı için bir kez kaydırırız.
    kernel[i, j] ağırlığı, resmin her pikselinin (i, j) konumundaki komşusuyla çarpılır.
    Hepsini toplayınca sonuç aynı çıkar, ama döngü sadece ksize*ksize kez döner (piksel sayısı kez değil).
    """
    height, width = image.shape
    ksize = kernel.shape[0]
    radius = ksize // 2

    padded = np.pad(image, radius, mode=pad_mode)
    output = np.zeros((height, width))

    for i in range(ksize):
        for j in range(ksize):
            # padded[i:i+height, j:j+width]: padded resmin (i, j) kadar kaydırılmış, orijinal boyutunda bir kesiti.
            # Bu kesitin her pikseli, orijinaldeki pikselin (i, j) konumundaki komşusudur.
            # Tek satırda tüm resmi birden çarpıp ekliyoruz (NumPy bunu C hızında yapar).
            output += kernel[i, j] * padded[i:i + height, j:j + width]
    return output


# ============================================================
# 2c) Konvolüsyon: ayrılabilir (önce yatay, sonra dikey)
# ============================================================
def convolve_separable(image, kernel_1d, pad_mode=PAD_MODE):
    """
    2B Gaussian = iki 1B Gaussian'ın çarpımı olduğu için, 2B konvolüsyon yerine iki 1B konvolüsyon yaparız:
    önce her satırı yatay kernel ile, sonra her sütunu dikey kernel ile.
    Sonuç birebir aynıdır ama çarpma sayısı çok azalır:
        2B: piksel başına ksize * ksize çarpma    (9x9 -> 81)
        1B+1B: piksel başına ksize + ksize çarpma (9+9 -> 18)
    """
    height, width = image.shape
    ksize = kernel_1d.shape[0]
    radius = ksize // 2

    padded = np.pad(image, radius, mode=pad_mode)

    # Adım 1: yatay. Sadece sütun yönünde kaydırıyoruz. (Satırlar padding'li kalır: height + 2*radius)
    horizontal = np.zeros((height + 2 * radius, width))
    for j in range(ksize):
        horizontal += kernel_1d[j] * padded[:, j:j + width]

    # Adım 2: dikey. Şimdi satır yönünde kaydırıyoruz.
    output = np.zeros((height, width))
    for i in range(ksize):
        output += kernel_1d[i] * horizontal[i:i + height, :]
    return output


# ------------------------------------------------------------
# Test resmi: gri tonlamalı ve küçük (döngülü yöntem hızlı bitsin)
# ------------------------------------------------------------
img = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)  # Gri okuyoruz: tek kanal, işlem basit
if img is None:
    raise FileNotFoundError(f"Görüntü okunamadı: {INPUT_PATH}")
scale = WORK_WIDTH / img.shape[1]
gray = cv2.resize(img, (WORK_WIDTH, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
gray_float = gray.astype(np.float64)  # Çarpma-toplama sırasında taşma olmasın diye ondalıklı tipe çeviriyoruz
print(f"test resmi: {gray.shape[1]}x{gray.shape[0]}, kernel: {KSIZE}x{KSIZE}, sigma={SIGMA}\n")

kernel_2d = gaussian_kernel_2d(KSIZE, SIGMA)


# ============================================================
# 4) Sonuç tablosu: süre ve cv2'ye göre fark
# ============================================================
def measure(function, repeat):
    """Fonksiyonu repeat kez çalıştırır, EN KISA süreyi (ms) ve son sonucu döndürür."""
    best = float("inf")  # inf = sonsuz. Herhangi bir süre bundan küçük olacak.
    result = None
    for _ in range(repeat):  # "_" kullanmayacağımız bir değişkenin adı
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

# cv2'nin uint8 sonucuyla kıyaslama: OpenCV 8 bit resimlerde hızlı ama yaklaşık (sabit nokta) hesap yapar,
# bu yüzden en fazla 1 seviyelik yuvarlama farkı beklenir.
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
# OpenCV neden bu kadar hızlı?
#  - C++ ile yazılmış: Python'un satır satır yorumlama yükü yok.
#  - SIMD: işlemci tek komutla birden çok pikseli aynı anda işleyebiliyor.
#  - Ayrılabilirlik: 2B yerine iki 1B filtre kullanıyor.
#  - 8 bit tamsayı hesabı: ondalık sayıdan daha hızlı.

# Üç yöntemin kendi aralarında birebir aynı sonucu verdiğini de kontrol edelim
print(f"\n(a) ile (b) arasındaki en büyük fark: {np.abs(out_loops - out_vec).max():.2e}")
print(f"(b) ile (c) arasındaki en büyük fark: {np.abs(out_vec - out_sep).max():.2e}")

# Matematiğin doğruluğunu kanıtlamak için cv2'yi ondalıklı (float32) resimle de çalıştır.
# Bu durumda OpenCV yuvarlama yapmaz, farkın pratikte sıfır çıkması gerekir.
out_cv_float = cv2.GaussianBlur(gray_float.astype(np.float32), (KSIZE, KSIZE), SIGMA)
print(f"kendi sonucumuz ile cv2 (float32) arasındaki en büyük fark: {np.abs(out_sep - out_cv_float).max():.2e}")


# ============================================================
# 3 devam) Kenar modlarını deneyelim: OpenCV'nin varsayılanı hangisi?
# ============================================================
print("\nKenar modu karşılaştırması (cv2 float32 varsayılanına göre en büyük fark):")
for mode in ["constant", "edge", "symmetric", "reflect"]:
    mine = convolve_separable(gray_float, my_kernel_1d, pad_mode=mode)
    print(f"  {mode:<10}: {np.abs(mine - out_cv_float).max():.2e}")
# "reflect" ~1e-5 (pratikte sıfır) çıkar, diğerleri 10-100 civarı çıkar. Yani OpenCV varsayılanı = BORDER_REFLECT_101.


# ============================================================
# Görsel karşılaştırma
# ============================================================
# Farkı gözle görmek için 50 ile çarpıyoruz (aksi halde 0-1 seviyelik fark siyah görünür).
diff_image = np.clip(np.abs(to_uint8(out_sep).astype(np.int16) - out_cv.astype(np.int16)) * 50, 0, 255).astype(np.uint8)
comparison = np.hstack([gray, to_uint8(out_sep), out_cv, diff_image])
cv2.imwrite(str(OUTPUT_DIR / "comparison.png"), comparison)
print(f"\nkaydedildi: {OUTPUT_DIR / 'comparison.png'}  (orijinal | bizim | cv2 | fark x50)")
show("orijinal | bizim (ayrilabilir) | cv2 | fark x50", comparison)
cv2.destroyAllWindows()


# ============================================================
# Kendini test et
# ============================================================
# 21x21 kernel ile piksel başına kaç çarpma yapılır?
#   2B konvolüsyon: 21 * 21 = 441
#   Ayrılabilir   : 21 + 21 = 42
#   Oran          : 441 / 42 = 10.5 kat daha az çarpma. Kernel büyüdükçe fark artar (ksize/2 kat).
print(f"\n21x21 kernel, piksel başına çarpma: 2B = {21 * 21}, ayrılabilir = {21 + 21}")
