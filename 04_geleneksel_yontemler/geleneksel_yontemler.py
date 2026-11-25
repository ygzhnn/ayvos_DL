"""
Geleneksel Görüntü İşleme Yöntemleri: Thresholding, Canny kenar bulma, Morfolojik işlemler

  1) Thresholding (eşikleme) : resmi siyah-beyaza çevirme. Sabit eşik, Otsu ve adaptive eşik
  2) Canny                   : kenar bulma. İki eşik değeri ve blur'un önemi
  3) Morfolojik işlemler     : erode, dilate, opening, closing. İkili (siyah-beyaz) resmi temizleme

Çalıştırma (proje kökünden, venv açıkken):
    python 04_geleneksel_yontemler/geleneksel_yontemler.py

Her pencerede bir tuşa basınca sonrakine geçer, ESC ile tümünü atlar.

Bu yöntemlerin ortak noktası: öğrenme (eğitim) yok. Her biri sabit bir kural/formülle çalışır.
Derin öğrenme çağında bile maske temizleme, ön işlem ve son işlem için her gün kullanılırlar.
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
    if labeled.ndim == 2:  # ndim: boyut sayısı. 2 ise gri resim, 3 ise renkli
        labeled = cv2.cvtColor(labeled, cv2.COLOR_GRAY2BGR)  # Renkli etiket için 3 kanala çevir
    cv2.rectangle(labeled, (0, 0), (330, 28), (0, 0, 0), -1)
    cv2.putText(labeled, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return labeled


def to_bgr(image):
    """Gri resmi 3 kanala çevirir (renkli resimle yan yana koyabilmek için)."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


# ============================================================
# 1) THRESHOLDING (eşikleme)
# ============================================================
# Fikir: gri bir resimde her pikselin değeri 0 (siyah) ile 255 (beyaz) arasındadır.
# Bir eşik değeri seçeriz. Eşikten büyük pikseller beyaz, küçük pikseller siyah olur.
# Sonuç sadece iki değerli (0 ve 255) bir "ikili (binary) resim"dir. Metni arka plandan ayırmak,
# nesneyi bulmak gibi işlerin ilk adımıdır.
#
# Deney için GÖLGELİ BİR BELGE oluşturuyoruz (elimizde gerçek bir fotoğraf yok):
# beyaz kâğıt, siyah yazı, ve soldan sağa doğru koyulaşan bir gölge.
doc = np.full((360, 640), 230, dtype=np.uint8)  # np.full: her yeri 230 (açık gri kâğıt) olan 360x640 tablo
lines = ["Gaussian blur", "Otsu threshold", "Canny edges"]
for n, text in enumerate(lines):  # enumerate: hem sırayı (n) hem elemanı (text) verir
    cv2.putText(doc, text, (30, 90 + n * 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 40, 5, cv2.LINE_AA)

# Gölge: sütunlara göre 1.0'dan 0.3'e düşen bir çarpan. Sağ taraf kararır.
shadow = np.linspace(1.0, 0.3, doc.shape[1])           # 640 elemanlı, 1.0'dan 0.3'e doğru düz azalan liste
doc = (doc * shadow[np.newaxis, :]).astype(np.uint8)   # np.newaxis: listeyi satır gibi yayar, her satıra uygular

# Hafif gürültü ekle (gerçek fotoğraflarda her zaman vardır)
rng = np.random.default_rng(1)
noise = rng.normal(0, 6, doc.shape)
doc = np.clip(doc.astype(np.float64) + noise, 0, 255).astype(np.uint8)
cv2.imwrite(str(OUTPUT_DIR / "doc_shadow.png"), doc)

# --- a) Sabit eşik ---
# cv2.threshold(resim, eşik, maksimum_değer, tür) iki şey döndürür: (kullanılan_eşik, sonuç_resmi)
# THRESH_BINARY: piksel > eşik ise maksimum_değer (255), değilse 0.
_, fixed = cv2.threshold(doc, 127, 255, cv2.THRESH_BINARY)

# --- b) Otsu ---
# Eşiği biz seçmek yerine algoritma seçer. Resmin histogramına (hangi parlaklıktan kaç piksel var)
# bakar ve resmi iki gruba en iyi ayıran eşiği bulur. Eşik olarak 0 veririz, otsu bayrağını ekleriz.
# Bayrakları toplamak için "|" (bit düzeyinde veya) kullanılır.
otsu_value, otsu = cv2.threshold(doc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu'nun seçtiği eşik: {otsu_value:.0f}")

# --- c) Adaptive (uyarlamalı) eşik ---
# Tek bir eşik yerine HER PİKSEL İÇİN ayrı eşik hesaplanır: pikselin etrafındaki küçük bölgenin
# ortalaması alınır, o ortalamadan biraz altı eşik olur. Gölge gibi düzgün olmayan aydınlatmada işe yarar.
#   blockSize = 25: etrafına bakılacak bölgenin boyutu (tek sayı olmalı)
#   C = 10: hesaplanan ortalamadan çıkarılan sabit. Büyük C = daha az piksel beyaz kalır
adaptive_mean = cv2.adaptiveThreshold(doc, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10)
adaptive_gauss = cv2.adaptiveThreshold(doc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
# GAUSSIAN_C: ortalama alırken yakın komşulara fazla ağırlık verir (az önceki Gaussian kernel mantığı)

row1 = np.hstack([add_label(doc, "orijinal (golgeli)"), add_label(fixed, "sabit esik 127")])
row2 = np.hstack([add_label(otsu, f"Otsu ({otsu_value:.0f})"), add_label(adaptive_gauss, "adaptive Gaussian")])
threshold_grid = np.vstack([row1, row2])
cv2.imwrite(str(OUTPUT_DIR / "threshold_compare.png"), threshold_grid)
show("1 - Thresholding karsilastirmasi", threshold_grid)
# Gözlem: Sağ taraf gölgede olduğu için sabit eşikte kâğıdın kendisi (230*0.3 = 69 < 127) siyah olur,
# yazı kaybolur. Otsu tek bir eşik seçtiği için gölgeyi tam çözemez. Adaptive her bölgeye ayrı baktığı
# için gölgeye rağmen yazıyı çıkarır. Adaptive sonucundaki arka plan noktacıkları resme eklediğimiz
# gürültüden gelir. Eşiklemeden önce hafif blur (cv2.GaussianBlur) ya da daha büyük C ile azalır.
# Aşağıdaki morfolojik işlemler de bu noktacıkları temizlemek için kullanılacak.

# Eşik türleri: aynı resme farklı kurallar uygulamak
_, t_binary = cv2.threshold(doc, 127, 255, cv2.THRESH_BINARY)       # >127 beyaz, değilse siyah
_, t_inv = cv2.threshold(doc, 127, 255, cv2.THRESH_BINARY_INV)      # tersi (yazı beyaz olur)
_, t_trunc = cv2.threshold(doc, 127, 255, cv2.THRESH_TRUNC)         # 127'den büyükler 127'ye kırpılır
_, t_tozero = cv2.threshold(doc, 127, 255, cv2.THRESH_TOZERO)       # 127'den küçükler 0 olur, büyükler aynı kalır
types_row = np.hstack([add_label(t_binary, "BINARY"), add_label(t_inv, "BINARY_INV")])
types_row2 = np.hstack([add_label(t_trunc, "TRUNC"), add_label(t_tozero, "TOZERO")])
show("1b - Esik turleri", np.vstack([types_row, types_row2]))


# ============================================================
# 2) CANNY KENAR BULMA
# ============================================================
# Kenar: resimde parlaklığın birden değiştiği yerler (bir nesnenin sınırı gibi).
# Canny, bu ani değişimleri bulan en yaygın klasik algoritmadır. Adımları:
#   1. Gaussian blur ile gürültüyü azalt (gürültü sahte kenarlar üretmesin diye)
#   2. Her pikselde parlaklığın değişim şiddetini ve yönünü hesapla (gradyan)
#   3. Kalın kenarları ince çizgilere indir
#   4. İki eşikle (düşük, yüksek) hangi kenarların kalacağına karar ver:
#        gradyan > yüksek eşik   -> kesin kenar
#        gradyan < düşük eşik    -> kenar değil, at
#        ikisinin arasında       -> sadece kesin bir kenara bağlıysa tut
# cv2.Canny adım 1'i (blur) kendisi YAPMAZ, biz yapmalıyız.
img = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Görüntü okunamadı: {INPUT_PATH}")
scale = 640 / img.shape[1]
gray = cv2.resize(img, (640, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)

blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

# cv2.Canny(resim, düşük_eşik, yüksek_eşik). Genelde yüksek = 2 veya 3 kat düşük seçilir.
edges_low = cv2.Canny(blurred, 20, 60)     # Düşük eşikler: çok kenar, ince detaylar da gelir
edges_mid = cv2.Canny(blurred, 60, 150)    # Orta
edges_high = cv2.Canny(blurred, 120, 250)  # Yüksek eşikler: sadece belirgin kenarlar kalır

canny_grid = np.hstack([add_label(edges_low, "Canny 20/60"),
                        add_label(edges_mid, "Canny 60/150"),
                        add_label(edges_high, "Canny 120/250")])
cv2.imwrite(str(OUTPUT_DIR / "canny_thresholds.png"), canny_grid)
show("2 - Canny: esiklerin etkisi", canny_grid)

# Blur neden gerekli? Gürültülü resimde blur olmadan Canny sahte kenarlar bulur.
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


# ============================================================
# 3) MORFOLOJİK İŞLEMLER (erode, dilate, opening, closing)
# ============================================================
# Morfoloji, ikili (siyah-beyaz) resimlerde beyaz şekillerin biçimini değiştirir.
# Küçük bir "yapı elemanı" (kernel) beyaz şekillerin üstünde gezdirilir.
#   ERODE  (aşındırma) : pikselin altındaki kernel bölgesinin HEPSİ beyazsa piksel beyaz kalır,
#                        biri bile siyahsa piksel siyah olur. Beyaz şekiller küçülür, küçük beyaz lekeler yok olur.
#   DILATE (genişletme): kernel bölgesinde BİR TANE bile beyaz varsa piksel beyaz olur.
#                        Beyaz şekiller büyür, küçük siyah delikler dolar.
#   OPENING = önce erode, sonra dilate. Küçük beyaz gürültüyü siler, şekillerin boyutunu korur.
#   CLOSING = önce dilate, sonra erode. Beyaz şekillerdeki küçük siyah delikleri kapatır.
# Genelde işlemin beyaz kısım üzerinde çalıştığını varsayarız (beyaz = ilgilendiğimiz nesne).

# Test resmi: adaptive threshold ile belgeden yazıyı çıkaralım (yazı BEYAZ, arka plan SİYAH olsun: INV)
text_mask = cv2.adaptiveThreshold(doc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 10)

# Üstüne bozulma ekle: arka planda beyaz lekeler (tuz) ve yazının içinde siyah delikler (biber)
rng = np.random.default_rng(3)
dirty = text_mask.copy()
salt = rng.random(dirty.shape) < 0.01                # %1 piksel: rastgele seçilen yerler
dirty[salt & (text_mask == 0)] = 255                 # sadece arka plandakileri beyaz yap
pepper = rng.random(dirty.shape) < 0.04
dirty[pepper & (text_mask == 255)] = 0               # sadece yazıdakileri siyah yap

# Yapı elemanı: 3x3'lük kare. Büyütürsen etki artar.
# cv2.getStructuringElement ile kare (MORPH_RECT), çarpı (MORPH_CROSS) veya elips (MORPH_ELLIPSE) üretilebilir.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

eroded = cv2.erode(dirty, kernel, iterations=1)      # iterations: işlemi kaç kez tekrarla
dilated = cv2.dilate(dirty, kernel, iterations=1)
opened = cv2.morphologyEx(dirty, cv2.MORPH_OPEN, kernel)    # küçük beyaz lekeler gitti mi?
closed = cv2.morphologyEx(dirty, cv2.MORPH_CLOSE, kernel)   # yazıdaki siyah delikler kapandı mı?
# Temizleme için ikisini art arda: önce opening (lekeleri sil), sonra closing (delikleri kapat)
cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

morph_row1 = np.hstack([add_label(dirty, "bozuk maske"), add_label(eroded, "erode"), add_label(dilated, "dilate")])
morph_row2 = np.hstack([add_label(opened, "opening"), add_label(closed, "closing"), add_label(cleaned, "opening + closing")])
morph_grid = np.vstack([morph_row1, morph_row2])
cv2.imwrite(str(OUTPUT_DIR / "morphology.png"), morph_grid)
show("3 - Morfolojik islemler", morph_grid)

# Beyaz leke (gürültü) sayısını ölçelim: bağlantılı bileşen sayısı = ayrı beyaz parça sayısı
for name, mask in [("bozuk", dirty), ("opening", opened), ("closing", closed), ("opening+closing", cleaned)]:
    n_components = cv2.connectedComponents(mask)[0] - 1  # [0] bileşen sayısı, arka plan da sayıldığı için 1 çıkarıyoruz
    print(f"{name:<16}: beyaz parça sayısı = {n_components}")

# Morphological gradient: dilate - erode. Beyaz şekillerin sadece kenar çizgisini verir.
gradient = cv2.morphologyEx(cleaned, cv2.MORPH_GRADIENT, kernel)
show("3b - Morphological gradient (sekil kenarlari)", add_label(gradient, "dilate - erode"))

cv2.destroyAllWindows()
print(f"\nkaydedildi: {OUTPUT_DIR}")
