"""
Geleneksel Görüntü İşleme Yöntemleri: Thresholding, Canny kenar bulma, Morfolojik işlemler

Çalıştırma (proje kökünden, venv açıkken):
    python 04_geleneksel_yontemler/geleneksel_yontemler.py
"""
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "data" / "test.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SHOW_WINDOWS = True


def show(title, image):
    """Resmi pencerede gösterir, tuşa basılınca kapatır. ESC'ye basılırsa sonrakileri atlar."""
    global SHOW_WINDOWS
    if not SHOW_WINDOWS:
        return
    cv2.imshow(title, image)
    if cv2.waitKey(0) == 27:  # 27 = ESC
        SHOW_WINDOWS = False
    cv2.destroyWindow(title)


def add_label(image, text):
    """Resmin sol üstüne okunaklı bir etiket yazar. Gri (tek kanallı) resimde de çalışır."""
    labeled = image.copy()
    if labeled.ndim == 2:
        labeled = cv2.cvtColor(labeled, cv2.COLOR_GRAY2BGR)  # Renkli etiket için 3 kanala çevir
    cv2.rectangle(labeled, (0, 0), (330, 28), (0, 0, 0), -1)
    cv2.putText(labeled, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return labeled


def to_bgr(image):
    """Gri resmi 3 kanala çevirir (renkli resimle yan yana koyabilmek için)."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


# 1) THRESHOLDING (eşikleme)
doc = np.full((360, 640), 230, dtype=np.uint8)
lines = ["Gaussian blur", "Otsu threshold", "Canny edges"]
for n, text in enumerate(lines):
    cv2.putText(doc, text, (30, 90 + n * 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 40, 5, cv2.LINE_AA)

shadow = np.linspace(1.0, 0.3, doc.shape[1])
doc = (doc * shadow[np.newaxis, :]).astype(np.uint8)

# Hafif gürültü ekle (gerçek fotoğraflarda her zaman vardır)
rng = np.random.default_rng(1)
noise = rng.normal(0, 6, doc.shape)
doc = np.clip(doc.astype(np.float64) + noise, 0, 255).astype(np.uint8)
cv2.imwrite(str(OUTPUT_DIR / "doc_shadow.png"), doc)

_, fixed = cv2.threshold(doc, 127, 255, cv2.THRESH_BINARY)

otsu_value, otsu = cv2.threshold(doc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu'nun seçtiği eşik: {otsu_value:.0f}")

adaptive_mean = cv2.adaptiveThreshold(doc, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10)
adaptive_gauss = cv2.adaptiveThreshold(doc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)

row1 = np.hstack([add_label(doc, "orijinal (golgeli)"), add_label(fixed, "sabit esik 127")])
row2 = np.hstack([add_label(otsu, f"Otsu ({otsu_value:.0f})"), add_label(adaptive_gauss, "adaptive Gaussian")])
threshold_grid = np.vstack([row1, row2])
cv2.imwrite(str(OUTPUT_DIR / "threshold_compare.png"), threshold_grid)
show("1 - Thresholding karsilastirmasi", threshold_grid)

# Eşik türleri: aynı resme farklı kurallar uygulamak
_, t_binary = cv2.threshold(doc, 127, 255, cv2.THRESH_BINARY)       # >127 beyaz, değilse siyah
_, t_inv = cv2.threshold(doc, 127, 255, cv2.THRESH_BINARY_INV)      # tersi (yazı beyaz olur)
_, t_trunc = cv2.threshold(doc, 127, 255, cv2.THRESH_TRUNC)         # 127'den büyükler 127'ye kırpılır
_, t_tozero = cv2.threshold(doc, 127, 255, cv2.THRESH_TOZERO)
types_row = np.hstack([add_label(t_binary, "BINARY"), add_label(t_inv, "BINARY_INV")])
types_row2 = np.hstack([add_label(t_trunc, "TRUNC"), add_label(t_tozero, "TOZERO")])
show("1b - Esik turleri", np.vstack([types_row, types_row2]))


# 2) CANNY KENAR BULMA
img = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Görüntü okunamadı: {INPUT_PATH}")
scale = 640 / img.shape[1]
gray = cv2.resize(img, (640, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)

blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

edges_low = cv2.Canny(blurred, 20, 60)
edges_mid = cv2.Canny(blurred, 60, 150)    # Orta
edges_high = cv2.Canny(blurred, 120, 250)

canny_grid = np.hstack([add_label(edges_low, "Canny 20/60"),
                        add_label(edges_mid, "Canny 60/150"),
                        add_label(edges_high, "Canny 120/250")])
cv2.imwrite(str(OUTPUT_DIR / "canny_thresholds.png"), canny_grid)
show("2 - Canny: esiklerin etkisi", canny_grid)

rng = np.random.default_rng(7)
noisy = np.clip(gray.astype(np.float64) + rng.normal(0, 20, gray.shape), 0, 255).astype(np.uint8)
edges_no_blur = cv2.Canny(noisy, 60, 150)                              # blur YOK: gürültü kenar sanılır
edges_with_blur = cv2.Canny(cv2.GaussianBlur(noisy, (5, 5), 1.4), 60, 150)  # blur VAR
blur_row = np.hstack([add_label(noisy, "gurultulu resim"),
                      add_label(edges_no_blur, "Canny, blur yok"),
                      add_label(edges_with_blur, "Canny, blur var")])
cv2.imwrite(str(OUTPUT_DIR / "canny_blur_effect.png"), blur_row)
show("2b - Canny'de blur'un etkisi", blur_row)
print(f"gürültülü resimde kenar piksel sayısı: blur yok = {int((edges_no_blur > 0).sum())}, "
      f"blur var = {int((edges_with_blur > 0).sum())}")


# 3) MORFOLOJİK İŞLEMLER (erode, dilate, opening, closing)

text_mask = cv2.adaptiveThreshold(doc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 10)

rng = np.random.default_rng(3)
dirty = text_mask.copy()
salt = rng.random(dirty.shape) < 0.01                # %1 piksel: rastgele seçilen yerler
dirty[salt & (text_mask == 0)] = 255                 # sadece arka plandakileri beyaz yap
pepper = rng.random(dirty.shape) < 0.04
dirty[pepper & (text_mask == 255)] = 0               # sadece yazıdakileri siyah yap

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

eroded = cv2.erode(dirty, kernel, iterations=1)      # iterations: işlemi kaç kez tekrarla
dilated = cv2.dilate(dirty, kernel, iterations=1)
opened = cv2.morphologyEx(dirty, cv2.MORPH_OPEN, kernel)    # küçük beyaz lekeler gitti mi?
closed = cv2.morphologyEx(dirty, cv2.MORPH_CLOSE, kernel)   # yazıdaki siyah delikler kapandı mı?
cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

morph_row1 = np.hstack([add_label(dirty, "bozuk maske"), add_label(eroded, "erode"), add_label(dilated, "dilate")])
morph_row2 = np.hstack([add_label(opened, "opening"), add_label(closed, "closing"), add_label(cleaned, "opening + closing")])
morph_grid = np.vstack([morph_row1, morph_row2])
cv2.imwrite(str(OUTPUT_DIR / "morphology.png"), morph_grid)
show("3 - Morfolojik islemler", morph_grid)

for name, mask in [("bozuk", dirty), ("opening", opened), ("closing", closed), ("opening+closing", cleaned)]:
    n_components = cv2.connectedComponents(mask)[0] - 1
    print(f"{name:<16}: beyaz parça sayısı = {n_components}")

gradient = cv2.morphologyEx(cleaned, cv2.MORPH_GRADIENT, kernel)
show("3b - Morphological gradient (sekil kenarlari)", add_label(gradient, "dilate - erode"))


# 4) SOBEL ve LAPLACIAN (gradyan): Canny'nin 2. adımını görünür yapmak
grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # (1, 0): x yönünde türev al
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # (0, 1): y yönünde türev al
magnitude = cv2.magnitude(grad_x, grad_y)
sobel_row = np.hstack([add_label(cv2.convertScaleAbs(grad_x), "Sobel x (dikey kenar)"),
                       add_label(cv2.convertScaleAbs(grad_y), "Sobel y (yatay kenar)"),
                       add_label(cv2.convertScaleAbs(magnitude), "buyukluk (sqrt(gx2+gy2))")])

laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
sobel_row2 = np.hstack([add_label(gray, "orijinal"),
                        add_label(cv2.convertScaleAbs(laplacian), "Laplacian"),
                        add_label(edges_mid, "Canny (karsilastirma)")])
cv2.imwrite(str(OUTPUT_DIR / "sobel_laplacian.png"), np.vstack([sobel_row, sobel_row2]))
show("4 - Sobel / Laplacian", np.vstack([sobel_row, sobel_row2]))


# 5) HİSTOGRAM EŞİTLEME ve CLAHE (kontrast artırma)
low_contrast = (gray.astype(np.float64) * 0.3 + 100).astype(np.uint8)  # değerleri ~100-176 arasına sıkıştır
equalized = cv2.equalizeHist(low_contrast)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # 8x8'lik kare ızgara
clahe_result = clahe.apply(low_contrast)

hist_row = np.hstack([add_label(low_contrast, "dusuk kontrast"),
                      add_label(equalized, "equalizeHist"),
                      add_label(clahe_result, "CLAHE")])
cv2.imwrite(str(OUTPUT_DIR / "histogram_clahe.png"), hist_row)
show("5 - Histogram esitleme / CLAHE", hist_row)
for name, image in [("düşük kontrast", low_contrast), ("equalizeHist", equalized), ("CLAHE", clahe_result)]:
    print(f"{name:<15}: en koyu = {image.min():>3}, en açık = {image.max():>3}, standart sapma (kontrast) = {image.std():.1f}")


# 6) KONTUR BULMA (findContours): maskedeki şekilleri tek tek yakalamak
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_view = cv2.cvtColor(doc, cv2.COLOR_GRAY2BGR)
kept = 0
for contour in contours:
    area = cv2.contourArea(contour)  # konturun içindeki piksel alanı
    if area < 40:
        continue
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(contour_view, (x, y), (x + w, y + h), (0, 200, 0), 1)
    kept += 1
cv2.imwrite(str(OUTPUT_DIR / "contours.png"), contour_view)
show("6 - Kontur ve bounding box", contour_view)
print(f"\nbulunan kontur: {len(contours)}, alanı >= 40 olan: {kept}")


# 7) HOUGH DÖNÜŞÜMÜ: çizgi ve daire bulma
shapes = np.full((360, 640), 255, dtype=np.uint8)   # beyaz zemin
cv2.line(shapes, (40, 60), (300, 60), 0, 3)          # yatay çizgi
cv2.line(shapes, (60, 300), (260, 120), 0, 3)        # eğik çizgi
cv2.rectangle(shapes, (350, 40), (600, 160), 0, 3)   # dikdörtgen (4 çizgi)
cv2.circle(shapes, (470, 270), 60, 0, 3)             # daire
cv2.circle(shapes, (330, 280), 25, 0, 3)             # küçük daire

shape_edges = cv2.Canny(shapes, 50, 150)

lines = cv2.HoughLinesP(shape_edges, 1, np.pi / 180, threshold=60, minLineLength=60, maxLineGap=10)
hough_view = cv2.cvtColor(shapes, cv2.COLOR_GRAY2BGR)
lines = np.empty((0, 4), dtype=np.int32) if lines is None else lines.reshape(-1, 4)  # hiç çizgi bulunmazsa None döner
n_lines = len(lines)
for x1, y1, x2, y2 in lines:
    cv2.line(hough_view, (x1, y1), (x2, y2), (0, 0, 255), 2)

circles = cv2.HoughCircles(cv2.GaussianBlur(shapes, (9, 9), 2), cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                           param1=100, param2=30, minRadius=15, maxRadius=80)
n_circles = 0 if circles is None else circles.shape[1]
if circles is not None:
    for cx, cy, radius in np.round(circles[0]).astype(int):
        cv2.circle(hough_view, (cx, cy), radius, (255, 0, 0), 2)
        cv2.circle(hough_view, (cx, cy), 3, (255, 0, 0), -1)  # merkez noktası

hough_row = np.hstack([add_label(shapes, "sekiller"), add_label(shape_edges, "Canny kenarlari"),
                       add_label(hough_view, "Hough: cizgi (kirmizi) daire (mavi)")])
cv2.imwrite(str(OUTPUT_DIR / "hough.png"), hough_row)
show("7 - Hough cizgi ve daire", hough_row)
print(f"Hough: {n_lines} çizgi parçası, {n_circles} daire bulundu (gerçekte: 6 kalın çizgi, 2 daire)")

cv2.destroyAllWindows()
print(f"\nkaydedildi: {OUTPUT_DIR}")
