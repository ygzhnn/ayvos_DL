"""
CNN Nedir? Convolution, ReLU, pooling, fully connected, softmax: hepsi NumPy ile, sıfırdan

Çalıştırma (proje kökünden, venv açıkken):
    python 06_cnn/cnn_nedir.py
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


# Temel parçalar (hepsi birkaç satır)
def conv2d(image, kernel):
    """Convolution (padding YOK, "valid"). Gaussian'ı elle yazarken kullandığımız kaydırılmış dilim yöntemi:"""
    kernel_height, kernel_width = kernel.shape
    out_height = image.shape[0] - kernel_height + 1
    out_width = image.shape[1] - kernel_width + 1
    output = np.zeros((out_height, out_width))
    for i in range(kernel_height):
        for j in range(kernel_width):
            output += kernel[i, j] * image[i:i + out_height, j:j + out_width]
    return output


def relu(x):
    """ReLU: negatifse 0, pozitifse aynen bırak. max(0, x)."""
    return np.maximum(0, x)


def max_pool(x, size=2):
    """Max pooling: resmi size x size'lık, birbirine DEĞMEYEN karelere böl, her karenin EN BÜYÜĞÜNÜ al."""
    height, width = x.shape
    return x.reshape(height // size, size, width // size, size).max(axis=(1, 3))


def softmax(scores):
    """Softmax: herhangi sayıları olasılığa çevirir (hepsi 0-1 arası, toplam 1)."""
    exps = np.exp(scores - scores.max())
    return exps / exps.sum()


# 1-5) Küçük bir resimde tüm adımlar, elle takip edilebilir sayılarla
image = np.array([[2, 2, 8, 8, 2, 2]] * 6, dtype=float)
print("--- Girdi resmi (6x6) ---")
print(image)

kernel = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=float)

feature_map = conv2d(image, kernel)
print("\n--- 1) Convolution çıktısı (4x4). Sol üst elle hesapla: (8-2)*3 = 18 ---")
print(feature_map)
print("  Pozitif (18): koyudan parlağa geçiş = şeridin SOL kenarı")
print("  Negatif (-18): parlaktan koyuya geçiş = şeridin SAĞ kenarı")

activated = relu(feature_map)
print("\n--- 2) ReLU sonrası (negatifler 0 oldu) ---")
print(activated)

pooled = max_pool(activated)
print("\n--- 3) Max pooling (2x2) sonrası ---")
print(pooled)

flat = pooled.flatten()
print("\n--- 4) Flatten ---")
print(flat)

weights = np.array([[0.1, -0.1, 0.1, -0.1],
                    [-0.1, 0.1, -0.1, 0.1]])    # sınıf 1: tersini sever
bias = np.array([0.0, 0.5])
scores = weights @ flat + bias
print("\n--- 5) FC puanları ---")
print(f"sınıf 0: 0.1*18 - 0.1*0 + 0.1*18 - 0.1*0 + 0.0 = {scores[0]:.2f}")
print(f"sınıf 1: -0.1*18 + 0.1*0 - 0.1*18 + 0.1*0 + 0.5 = {scores[1]:.2f}")
probs = softmax(scores)
print(f"softmax olasılıkları: sınıf 0 = {probs[0]:.3f}, sınıf 1 = {probs[1]:.3f} (toplam {probs.sum():.0f})")


# 6) Çıktı boyutu ve parametre sayısı
def conv_output_size(size, kernel_size, padding=0, stride=1):
    """Formül: (girdi + 2*padding - kernel) / stride + 1  (tam sayıya aşağı yuvarlanır)."""
    return (size + 2 * padding - kernel_size) // stride + 1


print("\n--- 6) Çıktı boyutu ---")
print(f"6x6 girdi, 3x3 kernel, padding=0, stride=1 -> {conv_output_size(6, 3)}x{conv_output_size(6, 3)}")
print(f"6x6 girdi, 3x3 kernel, padding=1, stride=1 -> {conv_output_size(6, 3, padding=1)} (padding boyutu korur)")
print(f"6x6 girdi, 3x3 kernel, padding=0, stride=2 -> {conv_output_size(6, 3, stride=2)} (stride atlayarak kaydırır, çıktı küçülür)")

in_channels, out_channels, kh, kw = 3, 8, 3, 3
conv_params = (kh * kw * in_channels + 1) * out_channels
fc_params = 224 * 224 * 3 * 1000 + 1000
print(f"\n224x224 RGB resim, {out_channels} adet 3x3 filtre: conv katmanı = {conv_params} parametre")
print(f"Aynı resmi doğrudan 1000 çıktılı FC'ye bağlasaydık: {fc_params:,} parametre")
print("  Neden bu kadar az? Aynı küçük kernel TÜM resimde kullanılır (parametre paylaşımı) ve")
print("  her çıktı sadece 3x3'lük yerel bölgeye bakar (yerel bağlantı).")


# 7) Gerçek resimde özellik haritaları
original = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)
if original is None:
    raise FileNotFoundError(f"Resim bulunamadı: {INPUT_PATH}")
scale = 640 / original.shape[1]
gray = cv2.resize(original, (640, int(original.shape[0] * scale)), interpolation=cv2.INTER_AREA)
gray = gray.astype(np.float32) / 255.0

filters = {
    "dikey kenar": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "yatay kenar": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "keskinlestir": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
}


def to_display(feature_map):
    """Özellik haritasını 0-255 gri resme çevirir (en büyük değer beyaz olacak şekilde ölçekler)."""
    maximum = feature_map.max()
    return (feature_map / maximum * 255).astype(np.uint8) if maximum > 0 else feature_map.astype(np.uint8)


def label(image, text):
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (260, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


tiles = [label((gray * 255).astype(np.uint8), "girdi")]
print("\n--- 7) Gerçek resimde ---")
print(f"girdi: {gray.shape[1]}x{gray.shape[0]}")
for name, filt in filters.items():
    conv_out = cv2.filter2D(gray, -1, filt)
    activated_map = relu(conv_out)                        # negatifleri sil
    h, w = activated_map.shape
    pooled_map = max_pool(activated_map[:h - h % 2, :w - w % 2])  # tek sayıysa son satırı at, 2x2 pooling
    tiles.append(label(to_display(pooled_map), f"{name} (ReLU+pool)"))
    print(f"{name:<13}: conv {conv_out.shape[1]}x{conv_out.shape[0]} -> pool {pooled_map.shape[1]}x{pooled_map.shape[0]}")

half_size = (tiles[1].shape[1], tiles[1].shape[0])
tiles[0] = cv2.resize(tiles[0], half_size, interpolation=cv2.INTER_AREA)
feature_grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
cv2.imwrite(str(OUTPUT_DIR / "feature_maps.png"), feature_grid)
show("7) Ozellik haritalari", feature_grid)


# 8) ÖĞRENME: kernel'i biz yazmasak, model nasıl bulur?
rng = np.random.default_rng(0)
train_image = rng.random((20, 20)) - 0.5
true_kernel = np.array([[-1, 0, 1]] * 3, dtype=float)       # gizli doğru cevap
target = conv2d(train_image, true_kernel)

learned = rng.normal(0, 0.1, (3, 3))                        # kernel rastgele küçük sayılarla başlar
learning_rate = 2.0
out_h, out_w = target.shape
print("\n--- 8) Öğrenme ---")
print("başlangıç kernel'i (rastgele):")
print(np.round(learned, 2))
for step in range(301):
    prediction = conv2d(train_image, learned)
    error = prediction - target                             # tahmin - hedef
    loss = (error ** 2).mean()                              # ortalama kare hata: küçüldükçe iyi
    gradient = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            gradient[i, j] = 2 * (error * train_image[i:i + out_h, j:j + out_w]).mean()
    learned -= learning_rate * gradient                     # loss'u azaltan yöne küçük adım at
    if step in (0, 10, 50, 300):
        print(f"adım {step:>3}: loss = {loss:.6f}")
print("öğrenilen kernel:")
print(np.round(learned, 2))
print("gerçek kernel:")
print(true_kernel)
print("-> Kernel'i biz yazmadık, model sadece örneklerden dikey kenar detektörünü kendi buldu.")
print("   Gerçek CNN bunu milyonlarca sayı ve binlerce resimle yapar, hedefi de sınıf etiketleri belirler.")

cv2.destroyAllWindows()
print(f"\nçıktılar kaydedildi: {OUTPUT_DIR}")
