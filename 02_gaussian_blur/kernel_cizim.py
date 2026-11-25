"""
Çan eğrisi ile kernel arasındaki ilişkiyi çizerek görmek

Bu dosya iki grafik çizer:
  Grafik 1: sigma SABİT, ksize değişiyor  -> ksize çanın ne kadarını kernel'e aldığımızı belirler
  Grafik 2: ksize SABİT, sigma değişiyor  -> sigma çanın genişliğini belirler

Grafiklerde:
  - Mavi çizgi  = çan eğrisi (Gaussian fonksiyonu, sürekli bir eğri)
  - Turuncu çubuklar = kernel (eğriden alınan, toplamı 1 yapılmış birkaç nokta)

Çalıştırma (proje kökünden, venv açıkken):
    python 02_gaussian_blur/kernel_cizim.py
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt  # Grafik çizme kütüphanesi. Kodda "plt" diye yazılır.
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def bell_curve(x, sigma):
    """
    Çan eğrisinin (Gaussian) yüksekliğini hesaplar.
    exp(-x² / 2σ²) formülünü, altındaki alan 1 olacak şekilde ölçekliyoruz
    (başına 1 / (sigma * karekök(2π)) ekleyerek). Böyle yapınca kernel noktalarıyla
    aynı ölçekte çizilebilir.
    """
    return np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def draw_kernel(ax, ksize, sigma, x_limit):
    """Tek bir grafik kutusuna (ax): çan eğrisini ve kernel çubuklarını çizer."""
    # --- Çan eğrisi: x'i -x_limit ile +x_limit arasında 400 noktaya bölüp her birinin yüksekliğini çiz
    x_curve = np.linspace(-x_limit, x_limit, 400)
    ax.plot(x_curve, bell_curve(x_curve, sigma), color="tab:blue", label="çan eğrisi")

    # --- Kernel: OpenCV'nin kendi kernel'i. Tek sütunlu tablo döner, .ravel() ile düz listeye çeviriyoruz.
    kernel = cv2.getGaussianKernel(ksize, sigma).ravel()

    # Kernel'in x konumları: merkez 0, yanlara doğru -1, -2, ... ve +1, +2, ...
    radius = ksize // 2  # "//" tam sayı bölmesi. Örn: 5 // 2 = 2
    x_kernel = np.arange(-radius, radius + 1)  # ksize=5 için [-2, -1, 0, 1, 2]

    # Kernel çubuklarını çiz (bar = çubuk grafik)
    ax.bar(x_kernel, kernel, width=0.6, color="tab:orange", alpha=0.8, label="kernel")

    # Kernel'in kesildiği yerleri gösteren kesikli dikey çizgiler
    ax.axvline(-radius - 0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(radius + 0.5, color="gray", linestyle="--", linewidth=0.8)

    ax.set_title(f"ksize={ksize}, sigma={sigma}\nmerkez={kernel[radius]:.3f}  uç={kernel[0]:.3f}")
    ax.set_xlim(-x_limit, x_limit)
    ax.set_xlabel("merkezden uzaklık (x)")


# ============================================================
# Grafik 1: sigma sabit (2), ksize değişiyor
# ============================================================
# Beklenti:
#   ksize=3  -> çanın sadece tepesi alınır, çubuklar birbirine yakın, kernel çanın ÜSTÜNDE kalır
#               (çünkü az eleman olunca toplamı 1 yapmak için hepsi büyütülür)
#   ksize=13 -> çanın neredeyse tamamı alınır, çubuklar çan eğrisinin üzerine oturur
SIGMA = 2.0
ksizes = [3, 7, 13]

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4), sharey=True)  # sharey: üç grafik aynı y ölçeğini kullansın
for ax, ksize in zip(axes1, ksizes):  # zip: iki listeyi ikişer ikişer gezer
    draw_kernel(ax, ksize, SIGMA, x_limit=8)
axes1[0].set_ylabel("ağırlık")
axes1[0].legend()
fig1.suptitle(f"sigma sabit ({SIGMA}), ksize büyüdükçe çanın daha fazlası kernel'e giriyor")
fig1.tight_layout()
fig1.savefig(OUTPUT_DIR / "kernel_ksize_etkisi.png", dpi=110)


# ============================================================
# Grafik 2: ksize sabit (9), sigma değişiyor
# ============================================================
# Beklenti:
#   sigma küçük -> çan dar ve sivri, merkez çok baskın -> az blur
#   sigma büyük -> çan geniş ve yassı, ağırlıklar eşitlenir -> çok blur
#   (sigma=4 iken çan kernel'in dışına taşar, ksize=9 çanı kırpmış olur)
KSIZE = 9
sigmas = [0.5, 1, 2, 4]

fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for ax, sigma in zip(axes2, sigmas):
    draw_kernel(ax, KSIZE, sigma, x_limit=8)
axes2[0].set_ylabel("ağırlık")
axes2[0].legend()
fig2.suptitle(f"ksize sabit ({KSIZE}), sigma büyüdükçe çan genişliyor ve ağırlıklar eşitleniyor")
fig2.tight_layout()
fig2.savefig(OUTPUT_DIR / "kernel_sigma_etkisi.png", dpi=110)

print(f"kaydedildi: {OUTPUT_DIR}")
plt.show()  # Grafikleri pencerede göster
