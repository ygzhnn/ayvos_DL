"""
Çan eğrisi ile kernel arasındaki ilişkiyi çizerek görmek

Çalıştırma (proje kökünden, venv açıkken):
    python 02_gaussian_blur/kernel_cizim.py
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def bell_curve(x, sigma):
    """Çan eğrisinin (Gaussian) yüksekliğini hesaplar."""
    return np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def draw_kernel(ax, ksize, sigma, x_limit):
    """Tek bir grafik kutusuna (ax): çan eğrisini ve kernel çubuklarını çizer."""
    x_curve = np.linspace(-x_limit, x_limit, 400)
    ax.plot(x_curve, bell_curve(x_curve, sigma), color="tab:blue", label="çan eğrisi")

    kernel = cv2.getGaussianKernel(ksize, sigma).ravel()

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


# Grafik 1: sigma sabit (2), ksize değişiyor
SIGMA = 2.0
ksizes = [3, 7, 13]

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, ksize in zip(axes1, ksizes):  # zip: iki listeyi ikişer ikişer gezer
    draw_kernel(ax, ksize, SIGMA, x_limit=8)
axes1[0].set_ylabel("ağırlık")
axes1[0].legend()
fig1.suptitle(f"sigma sabit ({SIGMA}), ksize büyüdükçe çanın daha fazlası kernel'e giriyor")
fig1.tight_layout()
fig1.savefig(OUTPUT_DIR / "kernel_ksize_etkisi.png", dpi=110)


# Grafik 2: ksize sabit (9), sigma değişiyor
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
