"""
split_yolo.py - Bagi dataset_yolo_raw menjadi train/val/test + data.yaml

Frame yang berurutan (jepretan beruntun) mirip satu sama lain. Kalau dibagi acak,
kembarannya bisa tersebar ke train dan val, lalu nilai val jadi terlalu bagus
(data leakage). Karena itu frame dikelompokkan dulu, dan satu kelompok selalu
masuk ke SATU bagian saja.

Penggunaan:
  python scripts/split_yolo.py --names Asra Ester Herta
  python scripts/split_yolo.py --names Asra Ester Herta --rasio 60 20 20 --blok 5
  python scripts/split_yolo.py --names Asra Ester Herta --val-sesi sesi2 --test-sesi sesi3

Hasil: dataset_v1/ (images/, labels/, data.yaml, split_report.csv)
"""

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SPLITS = ["train", "val", "test"]


def parse_name(stem):
    """kelas_tanggal_sesi_cahaya_nomor -> (kelas, sesi)"""
    parts = stem.split("_")
    kelas = parts[0]
    sesi = parts[2] if len(parts) >= 5 else "sesi?"
    return kelas, sesi


def ahash(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).flatten()


def assign_groups(groups, ratios):
    """Tempatkan tiap kelompok ke bagian yang kekurangan jatahnya paling besar."""
    total = sum(len(g) for g in groups)
    target = {s: r * total for s, r in zip(SPLITS, ratios)}
    count = {s: 0 for s in SPLITS}
    result = {}
    for g in groups:
        s = max(SPLITS, key=lambda k: target[k] - count[k])
        count[s] += len(g)
        for item in g:
            result[item] = s
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset_yolo_raw")
    ap.add_argument("--out", default="dataset_v1")
    ap.add_argument("--names", nargs="+", default=["Asra", "Ester"],
                    help="urutan = nomor kelas, harus sama dengan CLASSES di capture_yolo.py")
    ap.add_argument("--rasio", nargs=3, type=float, default=[70, 20, 10],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--blok", type=int, default=5,
                    help="jumlah frame berurutan yang dijaga satu kelompok")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-sesi", nargs="*", default=[])
    ap.add_argument("--test-sesi", nargs="*", default=[])
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    img_dir, lbl_dir = src / "images", src / "labels"
    if not img_dir.exists():
        sys.exit(f"[ERROR] Folder {img_dir} tidak ada. Jalankan capture_yolo.py dulu.")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"[ERROR] {out} sudah ada. Pakai --out dataset_v2 atau hapus dulu.")

    ids = {n.lower(): i for i, n in enumerate(args.names)}
    ratios = [r / sum(args.rasio) for r in args.rasio]
    rng = random.Random(args.seed)

    # ---- baca semua pasangan gambar + label
    by_class = defaultdict(list)
    for p in sorted(img_dir.glob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = lbl_dir / (p.stem + ".txt")
        if not lbl.exists():
            print(f"[SKIP] {p.name}: label tidak ada")
            continue
        kelas, sesi = parse_name(p.stem)
        if kelas.lower() not in ids:
            print(f"[SKIP] {p.name}: kelas '{kelas}' tidak ada di --names")
            continue
        first = lbl.read_text().split()
        if first and int(first[0]) != ids[kelas.lower()]:
            print(f"[PERINGATAN] {p.name}: nomor kelas di label ({first[0]}) "
                  f"tidak cocok dengan urutan --names ({ids[kelas.lower()]})")
        by_class[kelas].append((p, lbl, sesi))

    if not by_class:
        sys.exit("[ERROR] Tidak ada data yang bisa dibaca.")

    # ---- tentukan bagian untuk tiap gambar
    split_of = {}
    for kelas, items in by_class.items():
        if args.val_sesi or args.test_sesi:
            for it in items:
                s = ("val" if it[2] in args.val_sesi
                     else "test" if it[2] in args.test_sesi else "train")
                split_of[it[0]] = s
            continue
        groups = []
        by_sesi = defaultdict(list)
        for it in items:
            by_sesi[it[2]].append(it[0])
        for paths in by_sesi.values():
            paths.sort()
            for i in range(0, len(paths), args.blok):
                groups.append(paths[i:i + args.blok])
        rng.shuffle(groups)
        split_of.update(assign_groups(groups, ratios))

    # ---- salin file
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    rows = []
    for kelas, items in by_class.items():
        for p, lbl, sesi in items:
            s = split_of[p]
            shutil.copy2(p, out / "images" / s / p.name)
            shutil.copy2(lbl, out / "labels" / s / lbl.name)
            rows.append([p.name, kelas, sesi, s])

    # ---- data.yaml
    lines = [f"path: {out.resolve().as_posix()}",
             "train: images/train", "val: images/val", "test: images/test",
             "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(args.names)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(out / "split_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "kelas", "sesi", "split"])
        w.writerows(rows)

    # ---- laporan jumlah
    print(f"\nDataset tersimpan di: {out}")
    print(f"{'kelas':<10}" + "".join(f"{s:>8}" for s in SPLITS) + f"{'total':>8}")
    for kelas in by_class:
        c = [sum(1 for r in rows if r[1] == kelas and r[3] == s) for s in SPLITS]
        print(f"{kelas:<10}" + "".join(f"{n:>8}" for n in c) + f"{sum(c):>8}")
        if min(c) < 5:
            print(f"  [PERINGATAN] kelas {kelas}: salah satu bagian < 5 gambar, "
                  f"angka val/test akan sangat goyah.")

    # ---- cek kembar antar bagian (hash kasar)
    hashes = [(r[0], r[1], r[3], ahash(out / "images" / r[3] / r[0])) for r in rows]
    hashes = [h for h in hashes if h[3] is not None]
    mat = np.array([h[3] for h in hashes])
    cross = []
    for i in range(len(hashes)):
        d = (mat[i + 1:] != mat[i]).sum(axis=1)
        for j in np.where(d <= 4)[0]:
            b = hashes[i + 1 + j]
            if hashes[i][1] == b[1] and hashes[i][2] != b[2]:
                cross.append((hashes[i][0], b[0]))
    print(f"\nPasangan hampir kembar yang terpisah antar bagian: {len(cross)}")
    for a, b in cross[:5]:
        print(f"  {a}  <->  {b}")
    if cross:
        print("  Banyak pasangan = risiko data leakage. Ambil foto dengan variasi "
              "lebih besar atau bagi per sesi (--val-sesi / --test-sesi).")


if __name__ == "__main__":
    main()