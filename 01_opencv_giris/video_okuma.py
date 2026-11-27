"""
OpenCV Giriş (devam): video okuma, gösterme ve kaydetme

Çalıştırma (proje kökünden, venv açıkken):
    python 01_opencv_giris/video_okuma.py                 -> webcam (0)
    python 01_opencv_giris/video_okuma.py data/video.mp4  -> dosya
"""
import sys
import time  # Zaman ölçmek için (FPS hesabında)
from pathlib import Path

import cv2

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

source = sys.argv[1] if len(sys.argv) > 1 else 0

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    raise RuntimeError(f"Video kaynağı açılamadı: {source}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"kaynak: {source} | {width}x{height} | {source_fps:.1f} fps | {total_frames} kare")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUTPUT_DIR / "recording.mp4"), fourcc, source_fps, (width, height))

# Tuşlarla açıp kapatacağımız ayarlar (True = açık, False = kapalı)
gray_mode = False
mirror_mode = False

prev_time = time.perf_counter()
fps = 0.0                        # Ekrana yazacağımız FPS değeri
frame_count = 0                  # Şimdiye kadar işlenen kare sayısı

# while True: "sonsuz döngü". İçeriden break ile çıkarız.
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_count += 1  # "x += 1" demek "x = x + 1" demektir

    if mirror_mode:
        frame = cv2.flip(frame, 1)  # Yatay çevir (ayna etkisi)
    if gray_mode:
        frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    now = time.perf_counter()
    instant_fps = 1.0 / max(now - prev_time, 1e-6)  # max(...): sıfıra bölme hatasını önler
    prev_time = now
    fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps

    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Kare: {frame_count}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)

    writer.write(frame)  # Bu kareyi çıktı videosuna ekle
    cv2.imshow("Video (q: cik, g: gri, f: ayna, s: kaydet)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break  # Döngüden çık
    elif key == ord("g"):
        gray_mode = not gray_mode
    elif key == ord("f"):
        mirror_mode = not mirror_mode
    elif key == ord("s"):
        save_path = OUTPUT_DIR / f"frame_{frame_count:05d}.jpg"
        cv2.imwrite(str(save_path), frame)
        print(f"kaydedildi: {save_path.name}")

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"{frame_count} kare işlendi, video: {OUTPUT_DIR / 'recording.mp4'}")
