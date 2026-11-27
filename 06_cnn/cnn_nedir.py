"""
CNN Nedir? Convolution, ReLU, pooling, fully connected, softmax: hepsi NumPy ile, sıfırdan

  1) Convolution       : küçük bir kernel resmin üstünde gezer, her yerde "bu desen var mı?" diye sorar
  2) ReLU (activation) : negatifleri sıfırla, pozitifleri bırak
  3) Max pooling       : resmi küçült, en güçlü tepkiyi tut
  4) Flatten + FC      : sayıları tek listeye diz, sınıf puanlarına çevir
  5) Softmax           : puanları olasılığa çevir (toplamları 1)
  6) Çıktı boyutu ve parametre sayısı
  7) Gerçek resimde özellik haritaları (feature map)
  8) ÖĞRENME: kernel'i elle yazmak yerine örneklerden kendi bulsun

Çalıştırma (proje kökünden, venv açıkken):
    python 06_cnn/cnn_nedir.py

Her pencerede bir tuşa basınca sonrakine geçer, ESC ile tümünü atlar.

Büyük fikir: Gaussian blur'da kernel'i biz seçmiştik (çan eğrisi). CNN'de aynı convolution işlemini
kullanırız ama kernel'in içindeki sayıları BİZ yazmayız, model örneklerden ÖĞRENİR.
CNN = "kernel'leri kendi kendine öğrenen convolution katmanları yığını".
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


# ============================================================
# Temel parçalar (hepsi birkaç satır)
# ============================================================
def conv2d(image, kernel):
    """
    Convolution (padding YOK, "valid"). Gaussian'ı elle yazarken kullandığımız kaydırılmış dilim yöntemi:
    kernel'in her sayısı için, resmin kaydırılmış bir kopyasını o sayıyla çarpıp topluyoruz.
    Çıktı boyutu: (H - kh + 1) x (W - kw + 1). Kenarda kernel sığmadığı için çıktı biraz küçülür.

    NOT: CNN'lerdeki "convolution" matematikteki gerçek convolution değil, kernel'i ÇEVİRMEDEN
    uygulanan cross-correlation'dır. Gaussian'da kernel simetrik olduğu için fark yoktu. CNN'de
    kernel zaten öğrenildiği için çevirmenin anlamı yok, herkes çevirmeden uygular ve adına convolution der.
    """
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
    """
    Max pooling: resmi size x size'lık, birbirine DEĞMEYEN karelere böl, her karenin EN BÜYÜĞÜNÜ al.
    Çıktı her yönde size kat küçülür.
    reshape hilesi: (H, W) tabloyu (H/2, 2, W/2, 2) şekline getirirsek her 2x2 kare ayrı bir grup olur,
    sonra 1. ve 3. eksenler (karenin içi) üzerinden max alırız.
    """
    height, width = x.shape
    return x.reshape(height // size, size, width // size, size).max(axis=(1, 3))


def softmax(scores):
    """
    Softmax: herhangi sayıları olasılığa çevirir (hepsi 0-1 arası, toplam 1).
    Formül: e^puan / (tüm e^puanların toplamı). Büyük puan daha da büyük olasılık alır.
    Başta max'ı çıkarmak sayı taşmasını önler, sonucu değiştirmez.
    """
    exps = np.exp(scores - scores.max())
    return exps / exps.sum()


# ============================================================
# 1-5) Küçük bir resimde tüm adımlar, elle takip edilebilir sayılarla
# ============================================================
# 6x6'lık resim: ortada parlak (8) dikey bir şerit, kenarlarda koyu (2). Her satır aynı.
image = np.array([[2, 2, 8, 8, 2, 2]] * 6, dtype=float)
print("--- Girdi resmi (6x6) ---")
print(image)

# Dikey kenar bulan kernel: sağdakini + soldakini - sayar.
# Bir bölgede sağ taraf soldan parlaksa pozitif, koyuysa negatif, düzse 0 çıkar.
kernel = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=float)

# 1) Convolution. İlk pencereyi elle hesaplayalım: resmin sol üst 3x3'ü [[2,2,8],[2,2,8],[2,2,8]]
#    Her satır: -1*2 + 0*2 + 1*8 = 6. Üç satır: 6 + 6 + 6 = 18. Yani çıktının sol üstü 18.
feature_map = conv2d(image, kernel)
print("\n--- 1) Convolution çıktısı (4x4). Sol üst elle hesapla: (8-2)*3 = 18 ---")
print(feature_map)
print("  Pozitif (18): koyudan parlağa geçiş = şeridin SOL kenarı")
print("  Negatif (-18): parlaktan koyuya geçiş = şeridin SAĞ kenarı")

# 2) ReLU: sağ kenarın negatif tepkisi silinir. Yani bu kernel sadece "koyu -> parlak" kenarlara tepki verir.
activated = relu(feature_map)
print("\n--- 2) ReLU sonrası (negatifler 0 oldu) ---")
print(activated)
# Neden gerekli? ReLU olmasa art arda gelen convolution'lar tek bir büyük convolution'a eşdeğer olurdu
# (doğrusal işlemlerin birleşimi yine doğrusal). ReLU doğrusal olmayan bir kıvrım ekler, böylece
# birden çok katman gerçekten daha karmaşık şeyler öğrenebilir.

# 3) Max pooling 2x2: 4x4 -> 2x2. Her 2x2 karenin en büyüğü kalır.
#    Sol üst kare [[18,18],[18,18]] -> 18. Sağ üst kare [[0,0],[0,0]] -> 0.
pooled = max_pool(activated)
print("\n--- 3) Max pooling (2x2) sonrası ---")
print(pooled)
# Faydası: boyut küçülür (hesap azalır) ve "desen tam burada" yerine "desen şu civarda" bilgisi
# kalır, yani küçük kaymalara dayanıklılık gelir.

# 4) Flatten: 2x2 tabloyu 4 sayılık düz listeye diz. FC (fully connected) katmanı düz liste ister.
flat = pooled.flatten()
print("\n--- 4) Flatten ---")
print(flat)

# FC: her sınıf için bir "ağırlık satırı". Puan = girdilerin ağırlıklarla çarpılıp toplanması + bias.
# Bu sayıları gerçekte eğitim bulur. Burada elle yazdık: sınıf 0 "şerit VAR", sınıf 1 "şerit YOK".
weights = np.array([[0.1, -0.1, 0.1, -0.1],     # sınıf 0: sol sütundaki büyük tepkileri sever
                    [-0.1, 0.1, -0.1, 0.1]])    # sınıf 1: tersini sever
bias = np.array([0.0, 0.5])
scores = weights @ flat + bias  # "@" matris çarpımı: her sınıf için 4 çarpım toplanır
print("\n--- 5) FC puanları ---")
print(f"sınıf 0: 0.1*18 - 0.1*0 + 0.1*18 - 0.1*0 + 0.0 = {scores[0]:.2f}")
print(f"sınıf 1: -0.1*18 + 0.1*0 - 0.1*18 + 0.1*0 + 0.5 = {scores[1]:.2f}")
probs = softmax(scores)
print(f"softmax olasılıkları: sınıf 0 = {probs[0]:.3f}, sınıf 1 = {probs[1]:.3f} (toplam {probs.sum():.0f})")


# ============================================================
# 6) Çıktı boyutu ve parametre sayısı
# ============================================================
def conv_output_size(size, kernel_size, padding=0, stride=1):
    """Formül: (girdi + 2*padding - kernel) / stride + 1  (tam sayıya aşağı yuvarlanır)."""
    return (size + 2 * padding - kernel_size) // stride + 1


print("\n--- 6) Çıktı boyutu ---")
print(f"6x6 girdi, 3x3 kernel, padding=0, stride=1 -> {conv_output_size(6, 3)}x{conv_output_size(6, 3)}")
print(f"6x6 girdi, 3x3 kernel, padding=1, stride=1 -> {conv_output_size(6, 3, padding=1)} (padding boyutu korur)")
print(f"6x6 girdi, 3x3 kernel, padding=0, stride=2 -> {conv_output_size(6, 3, stride=2)} (stride atlayarak kaydırır, çıktı küçülür)")

# Parametre sayısı: bir conv katmanında her filtre (kh x kw x girdi_kanalı) ağırlık + 1 bias taşır.
in_channels, out_channels, kh, kw = 3, 8, 3, 3
conv_params = (kh * kw * in_channels + 1) * out_channels
# Aynı iş için "her piksel her çıktıya bağlı" bir FC katmanı olsaydı (224x224 RGB, 1000 çıktı):
fc_params = 224 * 224 * 3 * 1000 + 1000
print(f"\n224x224 RGB resim, {out_channels} adet 3x3 filtre: conv katmanı = {conv_params} parametre")
print(f"Aynı resmi doğrudan 1000 çıktılı FC'ye bağlasaydık: {fc_params:,} parametre")
print("  Neden bu kadar az? Aynı küçük kernel TÜM resimde kullanılır (parametre paylaşımı) ve")
print("  her çıktı sadece 3x3'lük yerel bölgeye bakar (yerel bağlantı).")


# ============================================================
# 7) Gerçek resimde özellik haritaları
# ============================================================
original = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)
if original is None:
    raise FileNotFoundError(f"Resim bulunamadı: {INPUT_PATH}")
scale = 640 / original.shape[1]
gray = cv2.resize(original, (640, int(original.shape[0] * scale)), interpolation=cv2.INTER_AREA)
gray = gray.astype(np.float32) / 255.0  # 0-1 arasına çek. Sinir ağları genelde küçük sayılarla çalışır.

# Üç elle yazılmış filtre. Gerçek CNN'de bunları model öğrenir, ilk katmanda genelde buna benzer
# kenar/renk/doku detektörleri kendiliğinden çıkar.
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
    # cv2.filter2D de convolution uygular (kernel çevirmeden, yani CNN'deki gibi)
    conv_out = cv2.filter2D(gray, -1, filt)
    activated_map = relu(conv_out)                        # negatifleri sil
    h, w = activated_map.shape
    pooled_map = max_pool(activated_map[:h - h % 2, :w - w % 2])  # tek sayıysa son satırı at, 2x2 pooling
    tiles.append(label(to_display(pooled_map), f"{name} (ReLU+pool)"))
    print(f"{name:<13}: conv {conv_out.shape[1]}x{conv_out.shape[0]} -> pool {pooled_map.shape[1]}x{pooled_map.shape[0]}")

# Pooling sonrası resimler yarı boyutta. Girdiyi de aynı boyuta küçültüp yan yana koyuyoruz.
half_size = (tiles[1].shape[1], tiles[1].shape[0])
tiles[0] = cv2.resize(tiles[0], half_size, interpolation=cv2.INTER_AREA)
feature_grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
cv2.imwrite(str(OUTPUT_DIR / "feature_maps.png"), feature_grid)
show("7) Ozellik haritalari", feature_grid)


# ============================================================
# 8) ÖĞRENME: kernel'i biz yazmasak, model nasıl bulur?
# ============================================================
# Elimizde bir girdi resmi ve istediğimiz çıktı (hedef) olsun. Kernel'i rastgele başlatıp
# "çıktı hedeften ne kadar uzak?" diye ölçüyoruz (loss) ve kernel'in her sayısını loss'u
# azaltacak yönde azar azar değiştiriyoruz (gradient descent). Sinir ağı eğitimi tam olarak budur.
#
# Burada hedefi, bildiğimiz dikey kenar kernel'inin çıktısından üretiyoruz. Model kernel'i hiç görmüyor,
# sadece (girdi, hedef) çiftlerinden onu geri bulmaya çalışıyor.
rng = np.random.default_rng(0)
# Rastgele 20x20 resim. 0.5 çıkarıp değerleri -0.5 ile +0.5 arasına, yani sıfır etrafına koyuyoruz
# (normalizasyon): ortalaması sıfır olan girdide öğrenme çok daha düzgün ve hızlı ilerler.
train_image = rng.random((20, 20)) - 0.5
true_kernel = np.array([[-1, 0, 1]] * 3, dtype=float)       # gizli doğru cevap
target = conv2d(train_image, true_kernel)

learned = rng.normal(0, 0.1, (3, 3))                        # kernel rastgele küçük sayılarla başlar
# learning_rate: her adımda gradyanın ne kadarı kadar ilerleyelim. ÇOK BÜYÜK seçersek hedefi
# aşıp karşı tarafa fırlarız, her adımda daha da uzaklaşırız ve loss patlar (denedik, sayılar 10^100'e çıktı).
# Çok küçük seçersek öğrenme çok yavaş olur. Doğru değeri denemeyle bulmak eğitimin sıradan bir parçası.
learning_rate = 2.0
out_h, out_w = target.shape
print("\n--- 8) Öğrenme ---")
print("başlangıç kernel'i (rastgele):")
print(np.round(learned, 2))
for step in range(301):
    prediction = conv2d(train_image, learned)
    error = prediction - target                             # tahmin - hedef
    loss = (error ** 2).mean()                              # ortalama kare hata: küçüldükçe iyi
    # Gradyan: kernel'in [i, j] sayısını artırırsak loss ne kadar değişir?
    # O sayı, her pencerede image[i:, j:] ile çarpıldığı için gradyan = hata x o dilim, ortalaması.
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
