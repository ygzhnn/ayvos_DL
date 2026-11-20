import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r"..\data\test.jpg"
img = cv2.imread(path)

if img is None:
    raise FileNotFoundError(f"Görüntü okunamadı: {path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV BGR okur, Matplotlib RGB bekler

print(f"shape: {img.shape}")   # (yükseklik, genişlik, kanal)
print(f"dtype: {img.dtype}")
print(f"min: {img.min()}")
print(f"max: {img.max()}")

# ---------------- ADIM 2: Sadece NumPy ile işlemler ----------------
h, w = img.shape[:2]

# 2a) Kırpma: önce y (satır), sonra x (sütun)
kirpik = img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
print(f"kırpılmış shape: {kirpik.shape}")

# 2b) Yatay çevirme: sütunları ters sırayla al
cevrik = img[:, ::-1]

# 2c) Parlaklık artırma
# YANLIŞ: uint8 255'i geçince başa sarar (250 + 10 = 4)
yanlis = img + 60
print(f"yanlış parlak -> min: {yanlis.min()}, max: {yanlis.max()}")

# DOĞRU: geniş tipe çevir, topla, 0-255'e sıkıştır, uint8'e geri dön
parlak = np.clip(img.astype(np.int16) + 60, 0, 255).astype(np.uint8)
print(f"doğru parlak  -> min: {parlak.min()}, max: {parlak.max()}")

# Küçük örnekle taşmayı gör
ornek = np.array([250], dtype=np.uint8)
print(f"uint8 250 + 10 = {ornek + 10}")
print(f"clip ile        = {np.clip(ornek.astype(np.int16) + 10, 0, 255).astype(np.uint8)}")

# ---------------- ADIM 3: 2x2 subplot ----------------
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].imshow(img)
ax[0, 0].set_title("Orijinal")

ax[0, 1].imshow(kirpik)
ax[0, 1].set_title("Kırpılmış")

ax[1, 0].imshow(cevrik)
ax[1, 0].set_title("Yatay çevrilmiş")

# Histogram: her kanal için 0-255 arası her değerden kaç piksel var
for i, renk in enumerate(["red", "green", "blue"]):
    sayim = np.bincount(img[:, :, i].ravel(), minlength=256)
    ax[1, 1].plot(sayim, color=renk, label=renk[0].upper())
ax[1, 1].set_title("Renk histogramı")
ax[1, 1].set_xlim(0, 255)
ax[1, 1].legend()

for a in [ax[0, 0], ax[0, 1], ax[1, 0]]:
    a.axis("off")  # görüntülerde eksen sayıları gereksiz

plt.tight_layout()
plt.show()