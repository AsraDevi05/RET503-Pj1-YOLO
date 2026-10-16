import csv
from pathlib import Path

NIM  = "4222401020"
NAMA = "Asra Devi Fanitya"   # ganti dengan nama lengkapmu

DATASET = "dataset_v1"

E1 = {"nama":"E1_pretrained","perubahan":"-","mAP50":0.971,"mAP5095":0.659,"P":0.942,"R":0.978,"ms":56.3,"menit":136}
E2 = {"nama":"E2_scratch","perubahan":"bobot acak","mAP50":0.936,"mAP5095":0.511,"P":0.890,"R":0.834,"ms":47.9,"menit":230}
E3 = {"nama":"E3_imgsz416","perubahan":"imgsz=416","mAP50":0.968,"mAP5095":0.689,"P":0.908,"R":0.953,"ms":36.2,"menit":39}

eksperimen = [E1, E2, E3]
model_terpilih = "E3_imgsz416"

test_all = {"P":0.948,"R":0.951,"mAP50":0.951,"mAP5095":0.712,"ms":24.4}
test_kelas = [
    {"kelas":"Asra", "P":0.960,"R":1.000,"mAP50":0.995,"mAP5095":0.757},
    {"kelas":"Ester","P":0.958,"R":1.000,"mAP50":0.995,"mAP5095":0.862},
    {"kelas":"Herta","P":0.927,"R":0.852,"mAP50":0.864,"mAP5095":0.517},
]

header = "  {:<18} {:<12} {:>6} {:>9} {:>6} {:>6} {:>6} {:>6}".format(
    "Eksperimen","Perubahan","mAP50","mAP50-95","P","R","ms","menit")
sep = "  " + "-"*75

rows_txt = [header, sep]
for e in eksperimen:
    rows_txt.append("  {:<18} {:<12} {:>6.3f} {:>9.3f} {:>6.3f} {:>6.3f} {:>6.1f} {:>6}".format(
        e["nama"],e["perubahan"],e["mAP50"],e["mAP5095"],e["P"],e["R"],e["ms"],e["menit"]))

rows_txt += [
    "",
    "  Model terpilih (mAP50-95 val tertinggi): " + model_terpilih,
    "",
    "  Hasil di data TEST (sekali saja):",
    "    all   P={P:.3f}  R={R:.3f}  mAP50={mAP50:.3f}  mAP50-95={mAP5095:.3f}  {ms:.1f}ms/img  ~{fps} FPS".format(
        fps=round(1000/test_all["ms"]), **test_all),
]
for k in test_kelas:
    rows_txt.append("    {kelas:<6} P={P:.3f}  R={R:.3f}  mAP50={mAP50:.3f}  mAP50-95={mAP5095:.3f}".format(**k))

txt = "  NIM / Nama : {} / {}\n  Dataset    : {}\n\n".format(NIM, NAMA, DATASET) + "\n".join(rows_txt)

# tulis .txt
Path("hasil_praktik_{}.txt".format(NIM)).write_text(txt, encoding="utf-8")
print("[+] hasil_praktik_{}.txt".format(NIM))

# tulis .csv
csv_path = "hasil_praktik_{}.csv".format(NIM)
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["NIM","Nama","Dataset","Eksperimen","Perubahan",
        "mAP50_val","mAP5095_val","P_val","R_val","ms","menit","terpilih","test_mAP50","test_mAP5095"])
    w.writeheader()
    for e in eksperimen:
        terpilih = e["nama"] == model_terpilih
        w.writerow({"NIM":NIM,"Nama":NAMA,"Dataset":DATASET,
            "Eksperimen":e["nama"],"Perubahan":e["perubahan"],
            "mAP50_val":e["mAP50"],"mAP5095_val":e["mAP5095"],
            "P_val":e["P"],"R_val":e["R"],"ms":e["ms"],"menit":e["menit"],
            "terpilih":"ya" if terpilih else "",
            "test_mAP50":test_all["mAP50"] if terpilih else "",
            "test_mAP5095":test_all["mAP5095"] if terpilih else ""})
print("[+] {}".format(csv_path))

# tulis README.md
readme = "\n".join([
    "# Hasil Praktik Pj1 - Smart Locker Face ID",
    "**NIM / Nama**: {} / {}".format(NIM, NAMA),
    "**Dataset**: {} (Asra, Ester, Herta - 3 kelas, 240 gambar, 2 sesi)".format(DATASET),
    "**Model**: YOLO26n | **Device**: CPU AMD Ryzen 3 7320U",
    "",
    "---",
    "",
    "## Tabel Eksperimen",
    "",
    txt,
    "",
    "---",
    "",
    "## Analisis",
    "",
    "### E1 vs E2: Manfaat Transfer Learning",
    "E1 (pretrained COCO) menghasilkan mAP50-95 val 0.659, sedangkan E2 (dari nol) hanya 0.511.",
    "Selisih 0.148 poin membuktikan transfer learning membantu, terutama dengan data terbatas.",
    "E2 juga butuh waktu lebih lama (230 menit vs 136 menit).",
    "",
    "### E1 vs E3: Efek imgsz=416",
    "Menurunkan imgsz dari 640 ke 416 meningkatkan mAP50-95 val (0.659 -> 0.689) sekaligus",
    "mempercepat training (39 menit) dan inferensi (36 ms vs 56 ms per gambar).",
    "Kemungkinan karena wajah di frame 1920x1080 sudah cukup besar untuk resolusi 416.",
    "",
    "### Model Terpilih: E3",
    "Dipilih berdasarkan mAP50-95 val tertinggi (0.689), sesuai kriteria slide 31 PPT.",
    "",
    "### Hasil di Data Test",
    "mAP50-95 test 0.712 lebih tinggi dari val (0.689), model tidak overfitting.",
    "Kecepatan 24.4 ms per gambar (~41 FPS) melampaui target desain >= 10 FPS.",
    "Herta memiliki mAP50-95 terendah (0.517), kemungkinan karena variasi jarak kurang.",
    "",
    "---",
    "",
    "## Keterbatasan",
    "- Data dari 2 sesi dengan kondisi serupa, angka mungkin optimistis untuk lapangan.",
    "- FAR belum diukur karena tidak ada data wajah orang lain.",
    "- Dataset kecil (80 gambar per kelas) membuat angka per kelas sensitif.",
    "",
    "---",
    "",
    "## File yang Dikumpulkan",
    "- hasil_praktik_{}.txt".format(NIM),
    "- hasil_praktik_{}.csv".format(NIM),
    "- runs/detect/E1_pretrained/results.png",
    "- runs/detect/E3_imgsz416/results.png",
    "- README.md",
])
Path("README.md").write_text(readme, encoding="utf-8")
print("[+] README.md")

print("\n=== ISI FILE TXT ===")
print(txt)