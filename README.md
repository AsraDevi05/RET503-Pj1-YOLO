# Hasil Praktik Pj1 - Smart Locker Face ID
**NIM / Nama**: 4222401020 / Asra Devi Fanitya
**Dataset**: dataset_v1 (Asra, Ester, Herta - 3 kelas, 240 gambar, 2 sesi)
**Model**: YOLO26n | **Device**: CPU AMD Ryzen 3 7320U

---

## Tabel Eksperimen

  NIM / Nama : 4222401020 / Asra Devi Fanitya
  Dataset    : dataset_v1

  Eksperimen         Perubahan     mAP50  mAP50-95      P      R     ms  menit
  ---------------------------------------------------------------------------
  E1_pretrained      -             0.971     0.659  0.942  0.978   56.3    136
  E2_scratch         bobot acak    0.936     0.511  0.890  0.834   47.9    230
  E3_imgsz416        imgsz=416     0.968     0.689  0.908  0.953   36.2     39

  Model terpilih (mAP50-95 val tertinggi): E3_imgsz416

  Hasil di data TEST (sekali saja):
    all   P=0.948  R=0.951  mAP50=0.951  mAP50-95=0.712  24.4ms/img  ~41 FPS
    Asra   P=0.960  R=1.000  mAP50=0.995  mAP50-95=0.757
    Ester  P=0.958  R=1.000  mAP50=0.995  mAP50-95=0.862
    Herta  P=0.927  R=0.852  mAP50=0.864  mAP50-95=0.517

---

## Analisis

### E1 vs E2: Manfaat Transfer Learning
E1 (pretrained COCO) menghasilkan mAP50-95 val 0.659, sedangkan E2 (dari nol) hanya 0.511.
Selisih 0.148 poin membuktikan transfer learning membantu, terutama dengan data terbatas.
E2 juga butuh waktu lebih lama (230 menit vs 136 menit).

### E1 vs E3: Efek imgsz=416
Menurunkan imgsz dari 640 ke 416 meningkatkan mAP50-95 val (0.659 -> 0.689) sekaligus
mempercepat training (39 menit) dan inferensi (36 ms vs 56 ms per gambar).
Kemungkinan karena wajah di frame 1920x1080 sudah cukup besar untuk resolusi 416.

### Model Terpilih: E3
Dipilih berdasarkan mAP50-95 val tertinggi (0.689), sesuai kriteria slide 31 PPT.

### Hasil di Data Test
mAP50-95 test 0.712 lebih tinggi dari val (0.689), model tidak overfitting.
Kecepatan 24.4 ms per gambar (~41 FPS) melampaui target desain >= 10 FPS.
Herta memiliki mAP50-95 terendah (0.517), kemungkinan karena variasi jarak kurang.

### Perbandingan dengan Target Desain
| Target | Nilai Target | Hasil |
|---|---|---|
| mAP50-95 | ≥ 0,6 | 0,712 ✅ |
| FPS | ≥ 10 | ~41 FPS ✅ |
| Latensi | ≤ 500 ms | 24,4 ms ✅ |
| Akurasi per kelas | ≥ 90% | Asra 99,5%, Ester 99,5%, Herta 86,4% ⚠️ |
| FAR | ≤ 5% | belum diukur ⚠️ |

---

## Keterbatasan
- Data dari 2 sesi dengan kondisi serupa, angka mungkin optimistis untuk lapangan.
- FAR belum diukur karena tidak ada data wajah orang lain.
- Dataset kecil (80 gambar per kelas) membuat angka per kelas sensitif.

---

## File yang Dikumpulkan
- hasil_praktik_4222401020.txt
- hasil_praktik_4222401020.csv
- runs/detect/E1_pretrained/results.png
- runs/detect/E3_imgsz416/results.png
- README.md

Program Studi Teknologi Rekayasa Robotika · Politeknik Negeri Batam
