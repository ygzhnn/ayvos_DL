"""
Keypoint ve Özellik Çıkarımı: SIFT, ORB ve eşleştirme (matching)

  1) Keypoint tespiti : resimdeki "ayırt edici" noktaları bulma (SIFT ve ORB)
  2) Descriptor       : her noktayı bir sayı listesiyle (parmak izi) tarif etme
  3) Matching         : iki resimdeki aynı noktaları birbirine eşleştirme
  4) Uygulama         : eşleşmelerden resmin nereye taşındığını (homography) bulma

Çalıştırma (proje kökünden, venv açıkken):
    python 05_keypoint_ozellik/keypoint_ozellik.py

Her pencerede bir tuşa basınca sonrakine geçer, ESC ile tümünü atlar.

Temel fikir: Düz bir duvarın üzerinde bir nokta seçsen, o noktayı başka bir fotoğrafta
bulamazsın (her yer birbirine benziyor). Ama bir köşe, bir göz, bir yazı harfi gibi
ayırt edici yerleri bulabilirsin. Keypoint = böyle yerler. Descriptor = o yerin etrafının
sayısal özeti. İki resimde descriptor'ları benzeyen noktalar, aynı yer demektir.
"""
import time
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "data" / "test.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SHOW_WINDOWS = True
WORK_WIDTH = 900  # Çalışacağımız genişlik. 5120 piksellik resimde SIFT çok yavaş ve gereksiz detaylı olur.


def show(title, image):
    """Resmi pencerede gösterir, tuşa basılınca kapatır. ESC'ye basılırsa sonrakileri atlar."""
    global SHOW_WINDOWS
    if not SHOW_WINDOWS:
        return
    # Geniş resimler ekrana sığmayabilir, 1400 pikselden genişse küçültüp gösteriyoruz.
    if image.shape[1] > 1400:
        scale = 1400 / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(title, image)
    if cv2.waitKey(0) == 27:  # 27 = ESC
        SHOW_WINDOWS = False
    cv2.destroyWindow(title)


# ============================================================
# Hazırlık: orijinal resim ve ona benzeyen ikinci bir resim
# ============================================================
original = cv2.imread(str(INPUT_PATH))
if original is None:
    raise FileNotFoundError(f"Resim bulunamadı: {INPUT_PATH}")

scale = WORK_WIDTH / original.shape[1]
img1 = cv2.resize(original, (WORK_WIDTH, int(original.shape[0] * scale)), interpolation=cv2.INTER_AREA)
height, width = img1.shape[:2]
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # Keypoint algoritmaları gri resimle çalışır

# İkinci resmi aynı resmin DÖNDÜRÜLMÜŞ, KÜÇÜLTÜLMÜŞ ve KARARTILMIŞ hali olarak üretiyoruz.
# Neden? Eşleştirmenin doğru olup olmadığını kontrol edebilmek için. Resmi biz döndürdüğümüz
# için hangi noktanın nereye gittiğini matematiksel olarak biliyoruz (M matrisi).
# Gerçek hayatta bu, aynı nesneyi farklı açıdan/mesafeden/ışıkta çekmeye benzer.
#
# getRotationMatrix2D(merkez, açı, ölçek): 2x3 dönüşüm matrisi verir (warpAffine'de kullanmıştık).
M = cv2.getRotationMatrix2D((width / 2, height / 2), 30, 0.7)  # 30 derece döndür, 0.7 kat küçült
img2 = cv2.warpAffine(img1, M, (width, height), borderValue=(40, 40, 40))
img2 = cv2.convertScaleAbs(img2, alpha=0.7, beta=0)  # alpha: parlaklık çarpanı. 0.7 -> %30 daha karanlık
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imwrite(str(OUTPUT_DIR / "img1_ve_img2.jpg"), np.hstack([img1, img2]))
print(f"resim boyutu: {width}x{height}")


# ============================================================
# 1) KEYPOINT TESPİTİ: SIFT ve ORB
# ============================================================
# SIFT (Scale-Invariant Feature Transform): Resmi farklı boyutlarda (ölçeklerde) tarar ve
#   her ölçekte "blob" (leke) ve köşe benzeri noktaları arar. Bu yüzden resim yakınlaşsa/uzaklaşsa,
#   dönse, ışığı değişse de aynı noktaları bulabilir. Çok güçlü ama yavaştır.
# ORB (Oriented FAST and Rotated BRIEF): Hızlı bir köşe bulucu (FAST) ile ikili (0/1) bir
#   tarif yöntemini (BRIEF) birleştirir. SIFT'ten çok daha hızlı ve ücretsizdir, ama biraz daha az dayanıklıdır.
sift = cv2.SIFT_create(nfeatures=1000)  # nfeatures: en iyi en fazla kaç nokta tutulsun
orb = cv2.ORB_create(nfeatures=1000)

# detectAndCompute(gri_resim, maske) iki şey döndürür:
#   keypoints   -> bulunan noktaların listesi (her biri bir KeyPoint nesnesi)
#   descriptors -> her noktanın "parmak izi". Bir NumPy tablosu, satır sayısı = nokta sayısı
# maske None: resmin tamamına bak demektir.
# Isınma turu: Bir algoritmanın İLK çağrısı, kendi iç hazırlıkları yüzünden olduğundan yavaş çıkar.
# Süreyi adil ölçmek için önce ikisini bir kez boşuna çalıştırıyoruz, sonra ölçüyoruz.
sift.detectAndCompute(gray1, None)
orb.detectAndCompute(gray1, None)

start = time.perf_counter()
kp1_sift, des1_sift = sift.detectAndCompute(gray1, None)
sift_time = (time.perf_counter() - start) * 1000

start = time.perf_counter()
kp1_orb, des1_orb = orb.detectAndCompute(gray1, None)
orb_time = (time.perf_counter() - start) * 1000

print("\n--- Keypoint tespiti (orijinal resimde) ---")
print(f"SIFT: {len(kp1_sift)} nokta, {sift_time:.1f} ms | descriptor tablosu: {des1_sift.shape}, tipi {des1_sift.dtype}")
print(f"ORB : {len(kp1_orb)} nokta, {orb_time:.1f} ms | descriptor tablosu: {des1_orb.shape}, tipi {des1_orb.dtype}")

# Bir KeyPoint nesnesinin içinde ne var? İlk noktaya bakalım.
kp = kp1_sift[0]
print("\nSIFT'in ilk noktası:")
print(f"  pt       = ({kp.pt[0]:.1f}, {kp.pt[1]:.1f})  -> noktanın (x, y) konumu")
print(f"  size     = {kp.size:.1f}  -> noktanın ne kadar büyük bir bölgeyi temsil ettiği (ölçek)")
print(f"  angle    = {kp.angle:.1f}  -> baskın yön (derece). Dönmeye dayanıklılık buradan gelir")
print(f"  response = {kp.response:.4f}  -> noktanın ne kadar güçlü/belirgin olduğu")
print(f"  descriptor'ın ilk 8 değeri: {des1_sift[0][:8]}")
# SIFT descriptor'ı 128 sayıdır: noktanın etrafı 4x4 = 16 küçük kareye bölünür,
# her karede 8 yönün gradyan (parlaklık değişimi) gücü ölçülür. 16 x 8 = 128.
# ORB descriptor'ı 32 bayt = 256 bittir: etraftaki seçilmiş piksel çiftlerinden
# "hangisi daha parlak" sorusunun 256 evet/hayır cevabı.

# drawKeypoints: noktaları resmin üstüne çizer.
# DRAW_RICH_KEYPOINTS bayrağı: her noktayı çember olarak çizer; çemberin yarıçapı = ölçek,
# içindeki çizgi = yön. Bayrak olmazsa sadece küçük bir nokta çizilir.
draw_sift = cv2.drawKeypoints(img1, kp1_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
draw_orb = cv2.drawKeypoints(img1, kp1_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(str(OUTPUT_DIR / "keypoints_sift.jpg"), draw_sift)
cv2.imwrite(str(OUTPUT_DIR / "keypoints_orb.jpg"), draw_orb)
show("1) SIFT keypoint'leri", draw_sift)
show("1) ORB keypoint'leri", draw_orb)

# ------------------------------------------------------------
# FAST (Features from Accelerated Segment Test): ORB'un içindeki köşe bulucu
# ------------------------------------------------------------
# Fikir: Bir pikselin (p) etrafında 16 pikselden oluşan bir çember düşün (yarıçap 3).
# Çemberdeki piksellerden ARDIŞIK en az 9 tanesi, p'den "threshold" kadar daha parlaksa
# ya da "threshold" kadar daha koyuysa, p bir köşedir. Örnek (threshold=20, p=100):
#   çemberde art arda 9 piksel 120 ve üzeri (daha parlak) -> p köşe
#   çemberde art arda 9 piksel 80 ve altı (daha koyu)    -> p köşe
#   ikisi de yoksa (düz bölge ya da düz kenar)           -> p köşe değil
# Sadece karşılaştırma yaptığı için çok hızlıdır (gradyan, blur, matematik yok).
#
# ÖNEMLİ: FAST sadece NOKTA bulur. Descriptor (parmak izi) üretmez, ölçek ya da yön bilmez.
# Yani tek başına eşleştirme için yetmez. ORB = FAST (nokta bulma) + yön ekleme + BRIEF (tarif etme).
#
# threshold: ne kadar fark "belirgin" sayılsın. Küçük -> çok nokta, büyük -> az nokta.
# nonmaxSuppression: yan yana bulunan nokta kümelerinden sadece en güçlüsünü tutar.
#   Kapalıysa bir köşenin etrafında 3-5 komşu nokta daha çıkar.
fast = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
fast.detect(gray1, None)  # Isınma turu (yukarıdaki gibi)
start = time.perf_counter()
kp1_fast = fast.detect(gray1, None)  # Sadece detect: descriptor yok, o yüzden tek değer döner
fast_time = (time.perf_counter() - start) * 1000
print(f"\nFAST (threshold=30): {len(kp1_fast)} nokta, {fast_time:.1f} ms (descriptor yok)")

# Threshold ve nonmaxSuppression'ın etkisi:
for thr in (10, 30, 60):
    count = len(cv2.FastFeatureDetector_create(threshold=thr, nonmaxSuppression=True).detect(gray1, None))
    print(f"  threshold={thr:>2}, nonmaxSuppression açık : {count} nokta")
count_no_nms = len(cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=False).detect(gray1, None))
print(f"  threshold=30, nonmaxSuppression KAPALI: {count_no_nms} nokta")

# FAST noktalarının boyutu hep aynıdır (7), çünkü ölçek bilgisi yok. Bu yüzden çemberler eşit boyda çıkar.
draw_fast = cv2.drawKeypoints(img1, kp1_fast, None, color=(0, 255, 0),
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(str(OUTPUT_DIR / "keypoints_fast.jpg"), draw_fast)
show("1) FAST keypoint'leri (sadece nokta)", draw_fast)

# FAST'e descriptor'ı sonradan ORB ile ekleyebiliriz: noktayı FAST bulur, tarifi ORB (BRIEF) yazar.
# orb.compute(resim, noktalar) -> (noktalar, descriptor'lar). Bazı kenara yakın noktalar
# (31x31 yama sığmadığı için) elenir, o yüzden nokta sayısı biraz azalabilir.
kp1_fast_orb, des1_fast_orb = orb.compute(gray1, kp1_fast)
print(f"FAST noktalarına ORB descriptor'ı eklendi: {len(kp1_fast_orb)} nokta, descriptor tablosu {des1_fast_orb.shape}")

# İkinci resimde de aynısını yapıyoruz.
kp2_sift, des2_sift = sift.detectAndCompute(gray2, None)
kp2_orb, des2_orb = orb.detectAndCompute(gray2, None)
print(f"\nİkinci resimde: SIFT {len(kp2_sift)} nokta, ORB {len(kp2_orb)} nokta")


# ============================================================
# 2) EŞLEŞTİRME (MATCHING)
# ============================================================
# Amaç: resim 1'deki her noktanın resim 2'deki karşılığını bulmak.
# Yöntem: descriptor'ları (sayı listelerini) karşılaştır. İki liste ne kadar yakınsa o kadar benzer.
# BFMatcher (Brute Force): kaba kuvvet. Resim 1'deki her descriptor'ı resim 2'deki
#   TÜM descriptor'larla tek tek karşılaştırır ve en yakınları seçer.
#
# Yakınlık nasıl ölçülür? Descriptor türüne göre:
#   SIFT (ondalıklı sayılar) -> NORM_L2: iki listenin Öklid uzaklığı (sayıların farkının karelerinin toplamının karekökü)
#   ORB  (bitler)            -> NORM_HAMMING: kaç bit farklı? (Örn: 10110 ile 10011 -> 2 bit farklı)
#
# knnMatch(des1, des2, k=2): her nokta için en yakın k=2 adayı verir (1. en iyi ve 2. en iyi).
#
# LOWE RATIO TEST: Bir eşleşmenin güvenilir olması için 1. aday, 2. adaydan BELİRGİN şekilde
# daha yakın olmalı. Eğer 1. ve 2. aday neredeyse aynı uzaklıktaysa, nokta "belirsiz"dir
# (örneğin tekrar eden bir desen) ve eşleşmeyi çöpe atarız.
#   best.distance < 0.75 * second.distance  ->  eşleşmeyi tut
def match_descriptors(des1, des2, norm_type, ratio=0.75):
    matcher = cv2.BFMatcher(norm_type)
    knn_pairs = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn_pairs:
        # Bazen bir noktanın 2 adayı bulunamaz (pair'in uzunluğu 2'den az olur), onları atlıyoruz.
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            good.append(best)
    return good


sift_matches = match_descriptors(des1_sift, des2_sift, cv2.NORM_L2)
orb_matches = match_descriptors(des1_orb, des2_orb, cv2.NORM_HAMMING)
print(f"\nLowe testinden geçen eşleşme: SIFT {len(sift_matches)}, ORB {len(orb_matches)}")

# Bir DMatch nesnesinin içinde:
#   queryIdx -> resim 1'deki noktanın sırası (kp1[queryIdx])
#   trainIdx -> resim 2'deki karşılığının sırası (kp2[trainIdx])
#   distance -> iki descriptor arası uzaklık (küçük = benzer)


# ============================================================
# 3) EŞLEŞMELER DOĞRU MU? (gerçek cevabı biliyoruz)
# ============================================================
# img2'yi M matrisiyle ürettik. Yani resim 1'deki (x, y) noktası resim 2'de M @ [x, y, 1] noktasına gitti.
# Her eşleşme için: "resim 1'deki noktayı M ile taşırsak, eşleştirilen resim 2 noktasına yakın düşüyor mu?"
# 3 pikselden azsa eşleşme DOĞRU, fazlaysa YANLIŞ sayıyoruz.
def check_matches(matches, kp1, kp2, matrix, tolerance=3.0):
    correct = []
    for m in matches:
        x1, y1 = kp1[m.queryIdx].pt
        expected = matrix @ np.array([x1, y1, 1.0])  # "@" matris çarpımı. Sonuç: beklenen (x, y)
        x2, y2 = kp2[m.trainIdx].pt
        error = np.hypot(expected[0] - x2, expected[1] - y2)  # hypot: iki nokta arası mesafe
        correct.append(error < tolerance)
    return np.array(correct)


sift_ok = check_matches(sift_matches, kp1_sift, kp2_sift, M)
orb_ok = check_matches(orb_matches, kp1_orb, kp2_orb, M)
print("\n--- Eşleşme doğruluğu (30 derece dönme + 0.7 ölçek + karartma) ---")
print(f"SIFT: {sift_ok.sum()} / {len(sift_matches)} doğru ({100 * sift_ok.mean():.0f}%)")
print(f"ORB : {orb_ok.sum()} / {len(orb_matches)} doğru ({100 * orb_ok.mean():.0f}%)")

# drawMatches: iki resmi yan yana koyup eşleşen noktaları çizgiyle birleştirir.
# Ekranda dağınık kalmasın diye en iyi (en küçük uzaklıklı) 60 eşleşmeyi çiziyoruz.
sift_best = sorted(sift_matches, key=lambda m: m.distance)[:60]
orb_best = sorted(orb_matches, key=lambda m: m.distance)[:60]
draw_sift_match = cv2.drawMatches(img1, kp1_sift, img2, kp2_sift, sift_best, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
draw_orb_match = cv2.drawMatches(img1, kp1_orb, img2, kp2_orb, orb_best, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(str(OUTPUT_DIR / "matches_sift.jpg"), draw_sift_match)
cv2.imwrite(str(OUTPUT_DIR / "matches_orb.jpg"), draw_orb_match)
show("2) SIFT eşleşmeleri", draw_sift_match)
show("2) ORB eşleşmeleri", draw_orb_match)

# Lowe testi olmasaydı ne olurdu? Her noktaya sadece en yakın adayı eşleştirip (k=1) bakalım.
plain_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
plain_matches = plain_matcher.match(des1_sift, des2_sift)
plain_ok = check_matches(plain_matches, kp1_sift, kp2_sift, M)
print(f"\nSIFT, Lowe testi OLMADAN: {plain_ok.sum()} / {len(plain_matches)} doğru ({100 * plain_ok.mean():.0f}%)")
print("  -> test, doğru eşleşme oranını artırır (kötü eşleşmeleri eler, az sayıda doğruyu da kaybedebilir)")


# ============================================================
# 4) TÜM DEDEKTÖRLERİN KARŞILAŞTIRMASI: SIFT, SURF, FAST, BRISK, ORB, MSER, Harris, MinEigen
# ============================================================
# Hepsinin ortak amacı nokta/bölge bulmak ama "ayırt edici" sayma biçimleri farklı:
#   SIFT    : farklı ölçeklerde leke (blob) arar. Ölçek ve dönmeye dayanıklı. Yavaş.
#   SURF    : SIFT'in hızlandırılmış hali. PATENTLİ, OpenCV'nin ana paketinde YOK (contrib gerekir).
#   FAST    : 16 pikselli çemberle köşe bulur. Çok hızlı, ölçek/yön bilmez.
#   BRISK   : FAST'i ölçek piramidiyle birleştirir, ikili descriptor üretir. (OpenCV 5'te contrib'e taşındı)
#   ORB     : FAST + yön + BRIEF descriptor. Hızlı, serbest.
#   MSER    : Maximally Stable Extremal Regions. Nokta değil BÖLGE bulur: eşiği değiştirdikçe
#             şekli neredeyse hiç değişmeyen (kararlı) parlak/koyu bölgeler. Yazı, tabela, blob bulmada iyi.
#   Harris  : Köşe tespiti. Pencereyi her yöne kaydırınca parlaklık hem x hem y'de çok değişiyorsa köşedir.
#   MinEigen: Harris'e benzer (Shi-Tomasi). Aynı hesabın iki özdeğerinden KÜÇÜĞÜNE bakar. Küçüğü bile
#             büyükse hem x hem y'de güçlü değişim var demektir, yani köşe. Takip (tracking) için çok kullanılır.
#
# Harris ve MinEigen nokta listesi değil, her piksele bir "köşelik puanı" veren bir HARİTA döndürür.
# Haritadan nokta çıkarmak için: (1) puanı yüksek olanları al, (2) komşusundan büyük olan
# yerel zirveleri tut (non-maximum suppression: dilate ile kendisine eşit olan pikseller zirvedir).
def response_map_to_keypoints(response, relative_threshold=0.01, window=7, size=7):
    cutoff = relative_threshold * response.max()  # en güçlü puanın %1'i
    local_max = cv2.dilate(response, np.ones((window, window), np.uint8))  # her pikselin komşuluktaki en büyük değeri
    ys, xs = np.where((response >= cutoff) & (response == local_max))  # hem güçlü hem yerel zirve olanlar
    return [cv2.KeyPoint(float(x), float(y), size) for x, y in zip(xs, ys)]


def detect_harris(gray):
    # blockSize=2: her pikselin 2x2 komşuluğuna bak, ksize=3: Sobel boyutu, k=0.04: Harris sabiti (0.04-0.06 yaygın)
    return response_map_to_keypoints(cv2.cornerHarris(np.float32(gray), 2, 3, 0.04))


def detect_min_eigen(gray):
    # blockSize=3, ksize=3. Dönen harita: her pikselde küçük özdeğer.
    return response_map_to_keypoints(cv2.cornerMinEigenVal(gray, 3, 3), relative_threshold=0.05)


def detect_mser(gray):
    return cv2.MSER_create().detect(gray)  # MSER her bölge için merkez ve boyut veren bir nokta döndürür


def detect_with(factory):
    """cv2'deki 'create' fonksiyonundan bir dedektör üretip noktaları döndüren yardımcı."""
    return lambda gray: factory().detect(gray, None)


# Her satır: (ad, dedektör fonksiyonu). Sürümde olmayan algoritmayı atlamak için getattr kullanıyoruz:
# getattr(cv2, "BRISK_create", None) -> varsa fonksiyonu, yoksa None verir (hata vermez).
detectors = [("SIFT", detect_with(cv2.SIFT_create)),
             ("SURF", None),   # aşağıda xfeatures2d'de aranacak
             ("FAST", detect_with(cv2.FastFeatureDetector_create)),
             ("BRISK", None),
             ("ORB", detect_with(cv2.ORB_create)),
             ("MSER", detect_mser),
             ("Harris", detect_harris),
             ("MinEigen", detect_min_eigen)]

contrib = getattr(cv2, "xfeatures2d", None)  # opencv-contrib-python kuruluysa burada SURF vb. bulunur
optional_factories = {
    "SURF": getattr(contrib, "SURF_create", None),
    "BRISK": getattr(cv2, "BRISK_create", None),
}
detectors = [(name, detect_fn if detect_fn is not None else
              (detect_with(optional_factories[name]) if optional_factories.get(name) else None))
             for name, detect_fn in detectors]

# TEKRARLANABİLİRLİK (repeatability): iyi bir dedektör, resim döndürülüp küçültülüp karartıldığında
# AYNI yerlerde nokta bulmalı. Ölçüm: resim 1'deki her noktayı M ile resim 2'ye taşı, 3 piksel
# yakınında resim 2'de bir nokta bulunmuş mu bak. Bulunanların oranı = tekrarlanabilirlik.
def repeatability(kp1, kp2, matrix, tolerance=3.0):
    if len(kp1) == 0 or len(kp2) == 0:
        return 0.0
    pts1 = np.array([k.pt for k in kp1])
    pts2 = np.array([k.pt for k in kp2])
    moved = (matrix @ np.hstack([pts1, np.ones((len(pts1), 1))]).T).T  # resim 1 noktalarını resim 2'ye taşı
    # Sadece resim 2'nin içine düşenleri say (dışarı taşan noktaları bulmak zaten imkansız)
    inside = (moved[:, 0] >= 0) & (moved[:, 0] < width) & (moved[:, 1] >= 0) & (moved[:, 1] < height)
    moved = moved[inside]
    if len(moved) == 0:
        return 0.0
    # Her taşınmış nokta için en yakın resim 2 noktasına uzaklık (broadcast: (N,1,2) - (1,K,2))
    distances = np.linalg.norm(moved[:, None, :] - pts2[None, :, :], axis=2).min(axis=1)
    return (distances < tolerance).mean()


print("\n--- Dedektör karşılaştırması ---")
print(f"{'ad':<9}{'nokta':>7}{'süre (ms)':>11}{'tekrarlanabilirlik':>21}")
tiles = []
for name, detect_fn in detectors:
    if detect_fn is None:
        print(f"{name:<9}{'bu OpenCV kurulumunda yok (contrib paketi gerekir)':>50}")
        continue
    detect_fn(gray1)  # ısınma
    start = time.perf_counter()
    kps1 = detect_fn(gray1)
    elapsed = (time.perf_counter() - start) * 1000
    kps2 = detect_fn(gray2)
    rep = repeatability(kps1, kps2, M)
    print(f"{name:<9}{len(kps1):>7}{elapsed:>11.1f}{100 * rep:>20.0f}%")
    tile = cv2.drawKeypoints(img1, kps1, None, color=(0, 255, 0))  # bayraksız: küçük nokta olarak çiz
    cv2.rectangle(tile, (0, 0), (260, 26), (0, 0, 0), -1)
    cv2.putText(tile, f"{name}: {len(kps1)} nokta", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(tile)

# Karşılaştırma ızgarası: 2'şer sütunlu
if len(tiles) % 2:
    tiles.append(np.zeros_like(tiles[0]))  # tek sayıdaysa boş kareyle tamamla
rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
detector_grid = np.vstack(rows)
cv2.imwrite(str(OUTPUT_DIR / "detectors_compare.jpg"), detector_grid)
show("4) Dedektör karşılaştırması", detector_grid)


# ============================================================
# 5) UYGULAMA: eşleşmelerden dönüşümü bulmak (homography + RANSAC)
# ============================================================
# Elimizde "resim 1'deki şu nokta = resim 2'deki şu nokta" çiftleri var. Bunlardan iki resim arasındaki
# dönüşümü (3x3 homography matrisi H) hesaplayabiliriz. Kullanım alanları: panorama dikme,
# bir nesneyi sahnede bulma, kamera hareketini takip etme, AR (artırılmış gerçeklik).
#
# Sorun: eşleşmelerin bir kısmı hâlâ yanlış (outlier). Yanlışlar hesabı bozar.
# RANSAC: rastgele birkaç eşleşme seç, H hesapla, kaç eşleşme bu H ile uyuşuyor say. Bunu çok kez
# tekrarla, en çok uyuşanı seç. Uyuşmayanlar (yanlış eşleşmeler) otomatik elenir.
src_pts = np.float32([kp1_sift[m.queryIdx].pt for m in sift_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2_sift[m.trainIdx].pt for m in sift_matches]).reshape(-1, 1, 2)

# findHomography(kaynak, hedef, yöntem, tolerans_piksel):
#   H    -> 3x3 dönüşüm matrisi
#   mask -> her eşleşme için 1 (inlier: H'ye uyuyor) veya 0 (outlier: elendi)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
inlier_mask = mask.ravel().astype(bool)
print(f"\nRANSAC: {inlier_mask.sum()} / {len(sift_matches)} eşleşme H ile uyuştu (inlier)")

# Resim 1'in 4 köşesini H ile resim 2'ye taşıyıp çerçeve çiziyoruz: "resim 1, resim 2'de nerede?"
corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
projected = cv2.perspectiveTransform(corners, H)  # 4 köşeyi H ile dönüştür
result = img2.copy()
cv2.polylines(result, [np.int32(projected)], True, (0, 255, 0), 3, cv2.LINE_AA)  # True: çokgeni kapat
cv2.imwrite(str(OUTPUT_DIR / "homography_outline.jpg"), result)
show("3) Resim 1'in resim 2'deki yeri (yeşil çerçeve)", result)

# Bulduğumuz H, bizim gerçek M matrisimizle uyuşuyor mu? M affine (2x3) olduğu için
# onu 3x3'e tamamlayıp (altına [0, 0, 1] ekleyerek) karşılaştırıyoruz.
M_full = np.vstack([M, [0, 0, 1]])
print("\nGerçek dönüşüm M (3x3'e tamamlanmış):")
print(np.round(M_full, 3))
print("RANSAC'ın bulduğu H:")
print(np.round(H / H[2, 2], 3))  # H'yi sağ alt elemana bölerek ölçekliyoruz (H bir sabitle çarpılsa aynı dönüşümü verir)

# Köşe hatası: gerçek M ile taşınan köşeler ile H ile taşınan köşeler arası ortalama fark.
true_corners = (M_full @ np.hstack([corners.reshape(-1, 2), np.ones((4, 1))]).T).T[:, :2]
corner_error = np.linalg.norm(true_corners - projected.reshape(-1, 2), axis=1).mean()
print(f"\nKöşelerde ortalama hata: {corner_error:.2f} piksel")
print(f"\nçıktılar kaydedildi: {OUTPUT_DIR}")
cv2.destroyAllWindows()
