"""
Keypoint ve Özellik Çıkarımı: SIFT, ORB ve eşleştirme (matching)

Çalıştırma (proje kökünden, venv açıkken):
    python 05_keypoint_ozellik/keypoint_ozellik.py
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
WORK_WIDTH = 900


def show(title, image):
    """Resmi pencerede gösterir, tuşa basılınca kapatır. ESC'ye basılırsa sonrakileri atlar."""
    global SHOW_WINDOWS
    if not SHOW_WINDOWS:
        return
    if image.shape[1] > 1400:
        scale = 1400 / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(title, image)
    if cv2.waitKey(0) == 27:  # 27 = ESC
        SHOW_WINDOWS = False
    cv2.destroyWindow(title)


# Hazırlık: orijinal resim ve ona benzeyen ikinci bir resim
original = cv2.imread(str(INPUT_PATH))
if original is None:
    raise FileNotFoundError(f"Resim bulunamadı: {INPUT_PATH}")

scale = WORK_WIDTH / original.shape[1]
img1 = cv2.resize(original, (WORK_WIDTH, int(original.shape[0] * scale)), interpolation=cv2.INTER_AREA)
height, width = img1.shape[:2]
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

M = cv2.getRotationMatrix2D((width / 2, height / 2), 30, 0.7)  # 30 derece döndür, 0.7 kat küçült
img2 = cv2.warpAffine(img1, M, (width, height), borderValue=(40, 40, 40))
img2 = cv2.convertScaleAbs(img2, alpha=0.7, beta=0)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imwrite(str(OUTPUT_DIR / "img1_ve_img2.jpg"), np.hstack([img1, img2]))
print(f"resim boyutu: {width}x{height}")


# 1) KEYPOINT TESPİTİ: SIFT ve ORB
sift = cv2.SIFT_create(nfeatures=1000)
orb = cv2.ORB_create(nfeatures=1000)

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

draw_sift = cv2.drawKeypoints(img1, kp1_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
draw_orb = cv2.drawKeypoints(img1, kp1_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(str(OUTPUT_DIR / "keypoints_sift.jpg"), draw_sift)
cv2.imwrite(str(OUTPUT_DIR / "keypoints_orb.jpg"), draw_orb)
show("1) SIFT keypoint'leri", draw_sift)
show("1) ORB keypoint'leri", draw_orb)

# FAST (Features from Accelerated Segment Test): ORB'un içindeki köşe bulucu
fast = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
fast.detect(gray1, None)  # Isınma turu (yukarıdaki gibi)
start = time.perf_counter()
kp1_fast = fast.detect(gray1, None)
fast_time = (time.perf_counter() - start) * 1000
print(f"\nFAST (threshold=30): {len(kp1_fast)} nokta, {fast_time:.1f} ms (descriptor yok)")

# Threshold ve nonmaxSuppression'ın etkisi:
for thr in (10, 30, 60):
    count = len(cv2.FastFeatureDetector_create(threshold=thr, nonmaxSuppression=True).detect(gray1, None))
    print(f"  threshold={thr:>2}, nonmaxSuppression açık : {count} nokta")
count_no_nms = len(cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=False).detect(gray1, None))
print(f"  threshold=30, nonmaxSuppression KAPALI: {count_no_nms} nokta")

draw_fast = cv2.drawKeypoints(img1, kp1_fast, None, color=(0, 255, 0),
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(str(OUTPUT_DIR / "keypoints_fast.jpg"), draw_fast)
show("1) FAST keypoint'leri (sadece nokta)", draw_fast)

kp1_fast_orb, des1_fast_orb = orb.compute(gray1, kp1_fast)
print(f"FAST noktalarına ORB descriptor'ı eklendi: {len(kp1_fast_orb)} nokta, descriptor tablosu {des1_fast_orb.shape}")

# İkinci resimde de aynısını yapıyoruz.
kp2_sift, des2_sift = sift.detectAndCompute(gray2, None)
kp2_orb, des2_orb = orb.detectAndCompute(gray2, None)
print(f"\nİkinci resimde: SIFT {len(kp2_sift)} nokta, ORB {len(kp2_orb)} nokta")


# 2) EŞLEŞTİRME (MATCHING)
def match_descriptors(des1, des2, norm_type, ratio=0.75):
    matcher = cv2.BFMatcher(norm_type)
    knn_pairs = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn_pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            good.append(best)
    return good


sift_matches = match_descriptors(des1_sift, des2_sift, cv2.NORM_L2)
orb_matches = match_descriptors(des1_orb, des2_orb, cv2.NORM_HAMMING)
print(f"\nLowe testinden geçen eşleşme: SIFT {len(sift_matches)}, ORB {len(orb_matches)}")


# 3) EŞLEŞMELER DOĞRU MU? (gerçek cevabı biliyoruz)
def check_matches(matches, kp1, kp2, matrix, tolerance=3.0):
    correct = []
    for m in matches:
        x1, y1 = kp1[m.queryIdx].pt
        expected = matrix @ np.array([x1, y1, 1.0])
        x2, y2 = kp2[m.trainIdx].pt
        error = np.hypot(expected[0] - x2, expected[1] - y2)  # hypot: iki nokta arası mesafe
        correct.append(error < tolerance)
    return np.array(correct)


sift_ok = check_matches(sift_matches, kp1_sift, kp2_sift, M)
orb_ok = check_matches(orb_matches, kp1_orb, kp2_orb, M)
print("\n--- Eşleşme doğruluğu (30 derece dönme + 0.7 ölçek + karartma) ---")
print(f"SIFT: {sift_ok.sum()} / {len(sift_matches)} doğru ({100 * sift_ok.mean():.0f}%)")
print(f"ORB : {orb_ok.sum()} / {len(orb_matches)} doğru ({100 * orb_ok.mean():.0f}%)")

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

plain_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
plain_matches = plain_matcher.match(des1_sift, des2_sift)
plain_ok = check_matches(plain_matches, kp1_sift, kp2_sift, M)
print(f"\nSIFT, Lowe testi OLMADAN: {plain_ok.sum()} / {len(plain_matches)} doğru ({100 * plain_ok.mean():.0f}%)")
print("  -> test, doğru eşleşme oranını artırır (kötü eşleşmeleri eler, az sayıda doğruyu da kaybedebilir)")


# 4) TÜM DEDEKTÖRLERİN KARŞILAŞTIRMASI: SIFT, SURF, FAST, BRISK, ORB, MSER, Harris, MinEigen
def response_map_to_keypoints(response, relative_threshold=0.01, window=7, size=7):
    cutoff = relative_threshold * response.max()  # en güçlü puanın %1'i
    local_max = cv2.dilate(response, np.ones((window, window), np.uint8))
    ys, xs = np.where((response >= cutoff) & (response == local_max))  # hem güçlü hem yerel zirve olanlar
    return [cv2.KeyPoint(float(x), float(y), size) for x, y in zip(xs, ys)]


def detect_harris(gray):
    return response_map_to_keypoints(cv2.cornerHarris(np.float32(gray), 2, 3, 0.04))


def detect_min_eigen(gray):
    # blockSize=3, ksize=3. Dönen harita: her pikselde küçük özdeğer.
    return response_map_to_keypoints(cv2.cornerMinEigenVal(gray, 3, 3), relative_threshold=0.05)


def detect_mser(gray):
    return cv2.MSER_create().detect(gray)


def detect_with(factory):
    """cv2'deki 'create' fonksiyonundan bir dedektör üretip noktaları döndüren yardımcı."""
    return lambda gray: factory().detect(gray, None)


detectors = [("SIFT", detect_with(cv2.SIFT_create)),
             ("SURF", None),   # aşağıda xfeatures2d'de aranacak
             ("FAST", detect_with(cv2.FastFeatureDetector_create)),
             ("BRISK", None),
             ("ORB", detect_with(cv2.ORB_create)),
             ("MSER", detect_mser),
             ("Harris", detect_harris),
             ("MinEigen", detect_min_eigen)]

contrib = getattr(cv2, "xfeatures2d", None)
optional_factories = {
    "SURF": getattr(contrib, "SURF_create", None),
    "BRISK": getattr(cv2, "BRISK_create", None),
}
detectors = [(name, detect_fn if detect_fn is not None else
              (detect_with(optional_factories[name]) if optional_factories.get(name) else None))
             for name, detect_fn in detectors]

def repeatability(kp1, kp2, matrix, tolerance=3.0):
    if len(kp1) == 0 or len(kp2) == 0:
        return 0.0
    pts1 = np.array([k.pt for k in kp1])
    pts2 = np.array([k.pt for k in kp2])
    moved = (matrix @ np.hstack([pts1, np.ones((len(pts1), 1))]).T).T  # resim 1 noktalarını resim 2'ye taşı
    inside = (moved[:, 0] >= 0) & (moved[:, 0] < width) & (moved[:, 1] >= 0) & (moved[:, 1] < height)
    moved = moved[inside]
    if len(moved) == 0:
        return 0.0
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


# 5) UYGULAMA: eşleşmelerden dönüşümü bulmak (homography + RANSAC)
src_pts = np.float32([kp1_sift[m.queryIdx].pt for m in sift_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2_sift[m.trainIdx].pt for m in sift_matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
inlier_mask = mask.ravel().astype(bool)
print(f"\nRANSAC: {inlier_mask.sum()} / {len(sift_matches)} eşleşme H ile uyuştu (inlier)")

corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
projected = cv2.perspectiveTransform(corners, H)  # 4 köşeyi H ile dönüştür
result = img2.copy()
cv2.polylines(result, [np.int32(projected)], True, (0, 255, 0), 3, cv2.LINE_AA)  # True: çokgeni kapat
cv2.imwrite(str(OUTPUT_DIR / "homography_outline.jpg"), result)
show("3) Resim 1'in resim 2'deki yeri (yeşil çerçeve)", result)

M_full = np.vstack([M, [0, 0, 1]])
print("\nGerçek dönüşüm M (3x3'e tamamlanmış):")
print(np.round(M_full, 3))
print("RANSAC'ın bulduğu H:")
print(np.round(H / H[2, 2], 3))

true_corners = (M_full @ np.hstack([corners.reshape(-1, 2), np.ones((4, 1))]).T).T[:, :2]
corner_error = np.linalg.norm(true_corners - projected.reshape(-1, 2), axis=1).mean()
print(f"\nKöşelerde ortalama hata: {corner_error:.2f} piksel")
print(f"\nçıktılar kaydedildi: {OUTPUT_DIR}")
cv2.destroyAllWindows()
