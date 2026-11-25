"""
OpenCV Giriş (devam): video okuma, gösterme ve kaydetme

  cv2.VideoCapture : webcam'den ya da dosyadan kare kare okuma
  cap.get          : fps, çözünürlük, kare sayısı gibi bilgiler
  cv2.waitKey(1)   : döngüde tuş kontrolü ('q' ile çıkış)
  cv2.VideoWriter  : işlenmiş kareleri video olarak kaydetme

Çalıştırma (proje kökünden, venv açıkken):
    python 01_opencv_giris/video_okuma.py                 -> webcam (0)
    python 01_opencv_giris/video_okuma.py data/video.mp4  -> dosya

Tuşlar:  q: çık   g: gri/renkli   f: ayna   s: kareyi kaydet

Temel fikir: video, arka arkaya gösterilen bir sürü resimdir (kare / frame).
Saniyede kaç kare gösterildiğine FPS (frames per second) denir.
Video işlemek demek bir döngüde her kareyi okuyup, işleyip, göstermek demektir.
"""
import sys  # Komut satırından verilen argümanları okumak için
import time  # Zaman ölçmek için (FPS hesabında)
from pathlib import Path

import cv2

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# sys.argv komut satırında yazılanların listesidir.
#   sys.argv[0] -> script'in adı
#   sys.argv[1] -> varsa, bizim verdiğimiz ilk argüman (video dosyası yolu)
# Argüman verilmediyse 0 kullanırız: 0, bilgisayardaki ilk (varsayılan) webcam demektir.
# "a if koşul else b" biçimi: koşul doğruysa a, değilse b.
source = sys.argv[1] if len(sys.argv) > 1 else 0

# VideoCapture bir video kaynağını (webcam ya da dosya) açar. Kareleri bu nesneden alacağız.
cap = cv2.VideoCapture(source)

# imread gibi VideoCapture de açamazsa hata vermez. isOpened() ile kontrol etmeliyiz.
# (Webcam başka bir programda açıksa ya da dosya yolu yanlışsa açılamaz.)
if not cap.isOpened():
    raise RuntimeError(f"Video kaynağı açılamadı: {source}")

# cap.get(...) videonun özelliklerini verir. Sayı olarak ondalıklı döner, int() ile tam sayıya çeviriyoruz.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# "x or 30": x sıfırsa (ya da boşsa) 30 kullan. Bazı webcam'ler FPS'i 0 bildirir, o yüzden yedek değer koyduk.
source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Webcam'de -1 ya da 0 çıkar, çünkü sonu yok
print(f"kaynak: {source} | {width}x{height} | {source_fps:.1f} fps | {total_frames} kare")

# VideoWriter: işlediğimiz kareleri bir video dosyasına yazar.
# Parametreler: (dosya yolu, codec, fps, (GENİŞLİK, YÜKSEKLİK))
# codec: videoyu sıkıştırma yöntemidir. "mp4v" .mp4 dosyaları için yaygın bir seçimdir.
# DİKKAT: Verdiğimiz boyut, yazacağımız karelerin boyutuyla birebir aynı olmalı.
# Farklıysa hata vermez ama bozuk ya da boş bir video üretir.
# Yazıcı 3 kanallı (renkli) kare bekler; tek kanallı gri kare yazarsak aynı sorun olur.
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUTPUT_DIR / "recording.mp4"), fourcc, source_fps, (width, height))

# Tuşlarla açıp kapatacağımız ayarlar (True = açık, False = kapalı)
gray_mode = False
mirror_mode = False

prev_time = time.perf_counter()  # Bir önceki karenin zamanı. perf_counter hassas bir kronometredir.
fps = 0.0                        # Ekrana yazacağımız FPS değeri
frame_count = 0                  # Şimdiye kadar işlenen kare sayısı

# while True: "sonsuz döngü". İçeriden break ile çıkarız.
while True:
    # cap.read() bir sonraki kareyi okur ve iki şey döndürür:
    #   ok    -> okuma başarılı mı (True/False)
    #   frame -> kare. Bir NumPy dizisi, tıpkı imread'in döndürdüğü gibi (BGR)
    ok, frame = cap.read()
    if not ok:  # Video bitti ya da kamera bağlantısı koptu
        break
    frame_count += 1  # "x += 1" demek "x = x + 1" demektir

    if mirror_mode:
        frame = cv2.flip(frame, 1)  # Yatay çevir (ayna etkisi)
    if gray_mode:
        # Griye çevirince kare tek kanallı olur. VideoWriter 3 kanal beklediği için
        # tekrar 3 kanala çeviriyoruz (görüntü gri kalır ama sayı tablosu 3 kanallı olur).
        frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # --- FPS hesabı ---
    # FPS = 1 / (iki kare arasında geçen süre). Örn: kareler arası 0.04 sn ise 1/0.04 = 25 FPS.
    now = time.perf_counter()
    instant_fps = 1.0 / max(now - prev_time, 1e-6)  # max(...): sıfıra bölme hatasını önler
    prev_time = now
    # Anlık FPS çok oynak olduğundan "yumuşatıyoruz": eski değerin %90'ı + yenisinin %10'u.
    # İlk karede (fps == 0) doğrudan anlık değeri alıyoruz.
    fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps

    # Köşeye bilgi yaz. putText'in parametreleri: resim, yazı, (x, y) konumu, yazı tipi,
    # boyut, renk (BGR), kalınlık, çizgi türü. f"..." içindeki {} değişkenin değerini yazıya koyar.
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Kare: {frame_count}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)

    writer.write(frame)  # Bu kareyi çıktı videosuna ekle
    cv2.imshow("Video (q: cik, g: gri, f: ayna, s: kaydet)", frame)

    # waitKey(1): 1 milisaniye tuş bekle ve devam et. Videoda bunu kullanırız.
    # waitKey(0) olsaydı her karede tuşa basana kadar durup video donardı.
    # "& 0xFF": bazı sistemlerde dönen sayının fazladan üst bitleri olur. Bu işlem sadece
    # son 8 biti tutarak tuş kodunu temizler. ord("q") 'q' harfinin sayısal kodunu verir.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break  # Döngüden çık
    elif key == ord("g"):
        gray_mode = not gray_mode  # "not": True ise False, False ise True yapar (aç/kapa anahtarı)
    elif key == ord("f"):
        mirror_mode = not mirror_mode
    elif key == ord("s"):
        # {frame_count:05d}: sayıyı 5 haneye sıfırla doldurur (3 -> 00003), dosyalar düzgün sıralanır
        save_path = OUTPUT_DIR / f"frame_{frame_count:05d}.jpg"
        cv2.imwrite(str(save_path), frame)
        print(f"kaydedildi: {save_path.name}")

# Kaynakları mutlaka serbest bırak. Yoksa webcam başka programlarda "meşgul" görünebilir
# ve video dosyası düzgün kapanmayıp bozuk kalabilir.
cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"{frame_count} kare işlendi, video: {OUTPUT_DIR / 'recording.mp4'}")
