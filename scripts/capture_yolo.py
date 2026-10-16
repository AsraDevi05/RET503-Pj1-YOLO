"""
capture_yolo.py - Ambil FRAME PENUH + label YOLO otomatis (YuNet)

Beda dengan capture.py lama:
  - yang disimpan frame penuh 1920x1080 (bukan crop wajah)
  - label YOLO (kelas cx cy w h) dibuat otomatis dari kotak YuNet
  - frame yang berisi 0 atau >1 wajah DITOLAK (supaya label tidak salah)

Penggunaan (jalankan dari folder YOLO):
  python scripts/capture_yolo.py Asra terang --sesi sesi1 --target 50 --cam 1
  python scripts/capture_yolo.py Asra redup  --sesi sesi2 --jarak jauh --ekspresi senyum

Kontrol:
  SPASI -> simpan frame + label
  Q     -> keluar

Hasil:
  dataset_yolo_raw/images/<kelas>_<tanggal>_<sesi>_<cahaya>_<no>.jpg
  dataset_yolo_raw/labels/<kelas>_<tanggal>_<sesi>_<cahaya>_<no>.txt
  dataset_yolo_raw/metadata_yolo.csv
"""

import argparse
import csv
import os
from datetime import date

import cv2

# Urutan di sini = nomor kelas YOLO (0, 1, 2, ...).
# Kalau mau menambah kelas "unknown" (orang lain), tambahkan di akhir:
#   CLASSES = ["Asra", "Ester", "Herta"]
CLASSES = ["Asra", "Ester", "Herta"]

MODEL_PATH = "face_detection_yunet.onnx"
OUT_DIR = "dataset_yolo_raw"


def load_yunet(w, h):
    return cv2.FaceDetectorYN.create(
        MODEL_PATH, "", (w, h),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5,
    )


def detect_faces(detector, frame_bgr, min_size):
    """YuNet di OpenCV menerima gambar BGR (langsung dari kamera)."""
    _, faces = detector.detect(frame_bgr)
    if faces is None:
        return []
    return [f for f in faces if f[2] >= min_size and f[3] >= min_size]


def yolo_line(cls_id, box, W, H):
    """Kotak (x, y, w, h) piksel -> baris label YOLO (0-1)."""
    x, y, w, h = box
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(W), x + w), min(float(H), y + h)
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kelas", help="Nama kelas, harus ada di CLASSES")
    p.add_argument("cahaya", help="terang / redup / backlight")
    p.add_argument("--sesi", default="sesi1", help="jangan pakai tanda _")
    p.add_argument("--jarak", default="sedang")
    p.add_argument("--ekspresi", default="netral")
    p.add_argument("--catatan", default="", help="mis. masker, kacamata")
    p.add_argument("--target", type=int, default=50)
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--minface", type=int, default=80)
    args = p.parse_args()

    # cocokkan nama kelas tanpa peduli huruf besar/kecil
    lookup = {c.lower(): i for i, c in enumerate(CLASSES)}
    if args.kelas.lower() not in lookup:
        print(f"[ERROR] Kelas '{args.kelas}' tidak ada di CLASSES: {CLASSES}")
        return
    cls_id = lookup[args.kelas.lower()]
    kelas = CLASSES[cls_id]
    sesi = args.sesi.replace("_", "-")
    cahaya = args.cahaya.replace("_", "-")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model YuNet tidak ditemukan: {MODEL_PATH}")
        print("        Letakkan face_detection_yunet.onnx di folder tempat kamu menjalankan script.")
        return

    img_dir = os.path.join(OUT_DIR, "images")
    lbl_dir = os.path.join(OUT_DIR, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    existing = [f for f in os.listdir(img_dir) if f.startswith(kelas + "_")]
    counter = len(existing) + 1
    tanggal = date.today().strftime("%Y%m%d")
    meta_path = os.path.join(OUT_DIR, "metadata_yolo.csv")
    write_header = not os.path.exists(meta_path)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = load_yunet(W, H)

    print(f"\n[INFO] Kelas: {kelas} (id {cls_id}) | Sesi: {sesi} | Cahaya: {cahaya} | Target: {args.target}")
    print(f"[INFO] Resolusi kamera: {W}x{H}")
    print("[TIP]  Wajah harus TEPAT 1 di frame. SPASI = simpan, Q = keluar\n")

    saved = 0
    with open(meta_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["nama_file", "kelas", "kelas_id", "tanggal", "sesi",
                             "kondisi_cahaya", "jarak", "ekspresi", "catatan",
                             "frame_w", "frame_h", "box_w", "box_h"])

        while saved < args.target:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Gagal membaca kamera.")
                break

            faces = detect_faces(detector, frame, args.minface)
            ok = len(faces) == 1

            preview = frame.copy()
            color = (0, 255, 0) if ok else (0, 0, 255)
            for f in faces:
                x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                cv2.rectangle(preview, (x, y), (x + w, y + h), color, 3)

            msg = "SIAP - tekan SPASI" if ok else f"Wajah terdeteksi: {len(faces)} (harus 1)"
            cv2.putText(preview, msg, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(preview, f"{kelas} | {sesi} | {cahaya} | {saved}/{args.target}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.imshow("Capture YOLO - Smart Locker", cv2.resize(preview, (960, 540)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Keluar.")
                break
            if key != ord(" "):
                continue
            if not ok:
                print("  [!] Frame ditolak: wajah harus tepat 1.")
                continue

            f = faces[0]
            box = (float(f[0]), float(f[1]), float(f[2]), float(f[3]))
            stem = f"{kelas}_{tanggal}_{sesi}_{cahaya}_{counter:03d}"
            cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(os.path.join(lbl_dir, stem + ".txt"), "w") as lf:
                lf.write(yolo_line(cls_id, box, W, H) + "\n")
            writer.writerow([stem + ".jpg", kelas, cls_id, tanggal, sesi, cahaya,
                             args.jarak, args.ekspresi, args.catatan,
                             W, H, int(box[2]), int(box[3])])
            csvfile.flush()
            print(f"  [+] {stem}.jpg  (wajah {int(box[2])}x{int(box[3])} px)")
            saved += 1
            counter += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[SELESAI] {saved} frame tersimpan di '{OUT_DIR}'")


if __name__ == "__main__":
    main()