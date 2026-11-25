"""
OpenCV ile Gaussian Blur: cv2.GaussianBlur() nasıl çalışır, parametreleri ne yapar?

Bu dosya şunları gösterir:
  1) Kernel nedir? (blur'un "tarifi") ve Gaussian kernel'in değerleri
  2) cv2.GaussianBlur parametreleri: ksize ve sigma
  3) ksize ve sigma kombinasyonlarını 3x3'lük ızgarada yan yana karşılaştırma
  4) Sadece sigma vermek (ksize=(0, 0)) ne yapar?
  5) Blur'un gerçekte ne işe yaradığı: gürültü azaltma

Çalıştırma (proje kökünden, venv açıkken):
    python 02_gaussian_blur/gaussian_blur.py

Her pencerede bir tuşa basınca sonrakine geçer, ESC ile tümünü atlar.

Temel fikir: Blur (bulanıklaştırma), her pikseli kendisi ve komşularının ORTALAMASI ile
değiştirmektir. Komşular farklı değerlerdeyse ortalama onları birbirine yaklaştırır, resim yumuşar.
"Gaussian" blur'da ortalama alırken yakın komşulara fazla, uzak komşulara az ağırlık verilir.
"""
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "data" / "test.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SHOW_WINDOWS = True  # ESC'ye basınca False olur ve kalan pencereler atlanır


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


# ------------------------------------------------------------
# Resmi oku ve çalışma boyutuna getir
# ------------------------------------------------------------
img = cv2.imread(str(INPUT_PATH))
if img is None:  # imread dosyayı bulamazsa hata vermez, None döndürür
    raise FileNotFoundError(f"Görüntü okunamadı: {INPUT_PATH}")

# Izgarada 9 resim yan yana olacağı için resmi küçük tutuyoruz (genişlik 480 piksel).
# Blur'un etkisini gözle görmek için detaylı bir bölge seçmek de iyi olur; burada tüm resmi alıyoruz.
img = fit_to_screen(img, 480)
print(f"çalışma boyutu: {img.shape[1]}x{img.shape[0]}")


# ============================================================
# 1) Kernel nedir?
# ============================================================
# Kernel, "her pikseli hangi komşuya hangi ağırlıkla katacağız?" sorusunun cevabını tutan
# küçük bir sayı tablosudur. Blur yapmak için bu tabloyu resmin her pikselinin üstüne
# gezdiririz: pikselin ve komşularının değerlerini tablodaki ağırlıklarla çarpıp toplarız.
#
# cv2.getGaussianKernel(ksize, sigma) bize 1 boyutlu Gaussian kernel'i verir.
# ksize = kernel'in uzunluğu, sigma = "çan eğrisinin genişliği" (aşağıda açıklıyoruz)
kernel_1d = cv2.getGaussianKernel(5, 1.0)  # (5 satır, 1 sütun) bir tablo döner
print("\n5 uzunluğunda Gaussian kernel (sigma=1):")
print(kernel_1d.ravel().round(4))  # .ravel(): tabloyu düz listeye çevirir, .round(4): 4 basamağa yuvarla

# İki boyutlu kernel, bu 1 boyutlu kernel'in kendisiyle çarpımıdır (np.outer = dış çarpım).
kernel_2d = np.outer(kernel_1d, kernel_1d)
print("\n5x5 Gaussian kernel (ortası en büyük, kenarlara doğru küçülüyor):")
print(kernel_2d.round(4))

# Kernel'deki tüm sayıların toplamı 1'dir. Neden önemli? Ağırlıklı ortalama aldığımız için
# toplam 1 olmalı. Toplam 1'den büyük olsaydı resim parlaklaşır, küçük olsaydı kararırdı.
print(f"kernel toplamı: {kernel_2d.sum():.4f}")


# ============================================================
# 2) cv2.GaussianBlur: temel kullanım
# ============================================================
# cv2.GaussianBlur(resim, ksize, sigmaX) üç temel parametre alır:
#
#   ksize  = (genişlik, yükseklik) kernel boyutu. TEK sayı olmalı: 3, 5, 7, 9, ...
#            Neden tek? Kernel'in tam ortasında bir "merkez piksel" olması gerekir.
#            Çift boyutlu kernel'in tam ortası yoktur (iki pikselin arasına düşer).
#            Büyük ksize = daha çok komşuya bakılır = daha fazla bulanıklık.
#
#   sigmaX = çan eğrisinin genişliği (standart sapma). Komşuların ağırlıkları nasıl dağılsın?
#            Küçük sigma: ağırlık merkeze toplanır, sadece en yakın komşular etkili, hafif blur.
#            Büyük sigma: ağırlık daha eşit dağılır, uzak komşular da etkili, güçlü blur.
#
# Örnek: 9x9 kernel, sigma=2
blurred = cv2.GaussianBlur(img, (9, 9), 2)
show("1 - Orijinal | GaussianBlur (9x9, sigma=2)", np.hstack([img, blurred]))


# ============================================================
# 3) ksize ve sigma kombinasyonlarını yan yana karşılaştırma (3x3 ızgara)
# ============================================================
# Soru: İkisinden hangisi bulanıklığı daha çok belirliyor, ksize mi sigma mı?
ksizes = [3, 9, 21]
sigmas = [1, 3, 8]

rows = []  # Izgaranın her satırını (3 resim yan yana) buraya ekleyeceğiz
for k in ksizes:
    row_images = []
    for s in sigmas:
        result = cv2.GaussianBlur(img, (k, k), s)
        row_images.append(add_label(result, f"ksize={k} sigma={s}"))
    rows.append(np.hstack(row_images))  # Bu satırdaki 3 resmi yatay birleştir
grid = np.vstack(rows)                  # 3 satırı alt alta birleştir (vstack = dikey birleştir)
show("2 - Izgara: satir = ksize (3, 9, 21), sutun = sigma (1, 3, 8)", grid)
cv2.imwrite(str(OUTPUT_DIR / "grid.jpg"), grid)

# Izgarada gözlemlenmesi gerekenler:
#  - ksize=3 iken sigma 1'den 8'e çıksa da fark çok azdır. Çünkü 3x3'lük kernel sadece
#    1 piksel komşuya bakabilir, sigma ne kadar büyük olursa olsun etkisi sınırlıdır.
#  - sigma=1 iken ksize 9'dan 21'e çıksa da fark çok azdır. Çünkü sigma=1 ile uzak komşuların
#    ağırlığı zaten sıfıra yakındır, büyük kernel o ağırlıkları kullanamaz.
#  - Güçlü blur için İKİSİ BİRLİKTE büyük olmalıdır: ksize=21 ve sigma=8.
# Yani her ikisi de etkili ama biri diğerini sınırlar; "kernel dışı ağırlık" kesilir.
# Pratik kural: ksize yaklaşık 6 * sigma + 1 olursa ikisi birbirine uyumlu olur.


# ============================================================
# 4) Sadece sigma vermek: ksize=(0, 0)
# ============================================================
# ksize=(0, 0) verirsek OpenCV kernel boyutunu sigma'dan KENDİSİ hesaplar
# (yaklaşık 6 * sigma + 1, tek sayıya yuvarlanır). Dokümandaki kural budur.
# Yani "ne kadar bulanık olsun" sorusuna tek bir sayıyla (sigma) cevap vermiş oluruz.
auto = cv2.GaussianBlur(img, (0, 0), 3)
manual = cv2.GaussianBlur(img, (19, 19), 3)  # 6 * 3 + 1 = 19: aynı sonucu bekliyoruz

# İki sonuç aynı mı? Aradaki farkın en büyüğüne bakalım.
# DİKKAT: uint8 sayılarda çıkarma yaparsak taşma olur (3 - 5 = 254). O yüzden önce
# int16'ya çeviriyoruz (negatif sayıları tutabilsin), sonra mutlak değer alıyoruz.
diff = np.abs(auto.astype(np.int16) - manual.astype(np.int16))
print(f"\nksize=(0,0) ile (19,19) arasındaki en büyük piksel farkı: {diff.max()}")
show("3 - ksize=(0,0) sigma=3 | ksize=(19,19) sigma=3", np.hstack([auto, manual]))

# Tersi de mümkün: sigma=0 verip sadece ksize verirsek sigma, ksize'dan hesaplanır:
#   sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
only_ksize = cv2.GaussianBlur(img, (9, 9), 0)
print("ksize=(9,9), sigma=0 -> sigma otomatik hesaplanır:", round(0.3 * ((9 - 1) * 0.5 - 1) + 0.8, 2))


# ============================================================
# 5) sigmaX ve sigmaY: yönlü blur
# ============================================================
# Şu ana kadar tek sigma verdik, OpenCV onu hem yatay (X) hem dikey (Y) yönde kullandı.
# İstersek ikisini ayrı verebiliriz: sigmaX yatay, sigmaY dikey yönde bulanıklığı belirler.
horizontal = cv2.GaussianBlur(img, (0, 0), sigmaX=8, sigmaY=0.1)  # Sadece yatayda bulanık
vertical = cv2.GaussianBlur(img, (0, 0), sigmaX=0.1, sigmaY=8)    # Sadece dikeyde bulanık
show("4 - Yatay blur | Dikey blur (hareket bulanıklığı gibi)",
     np.hstack([add_label(horizontal, "sigmaX=8 sigmaY~0"), add_label(vertical, "sigmaX~0 sigmaY=8")]))


# ============================================================
# 6) Blur ne işe yarar? Gürültü azaltma
# ============================================================
# Gürültü: resimdeki rastgele, bozucu pikseller (düşük ışıkta çekilen fotoğraflardaki "tane tane" görüntü).
# Gaussian blur komşuları ortaladığı için rastgele sapmaları yumuşatır.
# Önce yapay gürültü ekleyelim: her piksele rastgele bir değer katacağız.
rng = np.random.default_rng(42)  # 42: tohum (seed). Her çalıştırmada aynı rastgele sayıları üretir
noise = rng.normal(0, 25, img.shape)  # Ortalaması 0, standart sapması 25 olan rastgele sayılar
# Taşma tuzağı: img uint8, gürültü ondalıklı. Önce ondalıklıya çevir, topla, 0-255'e kırp, sonra uint8'e dön.
noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

denoised = cv2.GaussianBlur(noisy, (5, 5), 1.5)
show("5 - Gurultulu | Gaussian blur ile temizlenmis",
     np.hstack([add_label(noisy, "gurultulu"), add_label(denoised, "temizlenmis")]))

# Blur'un bedeli: gürültüyle birlikte kenarlar ve ince detaylar da yumuşar.
# Bu yüzden blur'u gürültüyü gidermeye yetecek kadar, fazla değil kullanırız.
# (Kenar bulma gibi sonraki adımlardan önce ön işlem olarak da sıkça kullanılır.)


# ============================================================
# 7) Kenar modu: resmin kenarında ne olur?
# ============================================================
# Kernel resmin en kenarındaki piksele gelince, kernel'in bir kısmı resmin DIŞINA taşar.
# Olmayan piksellerin yerine ne konacağını borderType belirler. OpenCV'nin varsayılanı
# BORDER_REFLECT_101'dir (kenardaki pikseli eksen alıp ayna gibi yansıtır).
default_border = cv2.GaussianBlur(img, (21, 21), 8)  # Varsayılan: BORDER_REFLECT_101
constant_border = cv2.GaussianBlur(img, (21, 21), 8, borderType=cv2.BORDER_CONSTANT)  # Dışarıyı siyah (0) say
# BORDER_CONSTANT ile kenarlar koyulaşır, çünkü kernel dışarıdaki siyah piksellerle ortalama alır.
show("6 - Varsayilan kenar | BORDER_CONSTANT (kenarlar koyulasir)",
     np.hstack([add_label(default_border, "varsayilan"), add_label(constant_border, "CONSTANT")]))

cv2.imwrite(str(OUTPUT_DIR / "blurred.jpg"), blurred)
cv2.imwrite(str(OUTPUT_DIR / "denoised.jpg"), denoised)
print(f"\nkaydedildi: {OUTPUT_DIR}")
cv2.destroyAllWindows()
