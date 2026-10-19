# proyek-deteksi-cacat-daun

UTS Pengolahan Citra Digital, Topik 4: Deteksi Cacat Permukaan Daun.

Program ini menerima foto daun, lalu menghasilkan **mask biner area cacat**, **bounding box**, dan **luas cacat (piksel)**. Seluruh pengolahan memakai teknik klasik (tanpa deep learning) dengan Python 3, NumPy, OpenCV, scikit-image, dan Matplotlib.

Anggota: Erwin Bernianto (NIM 2430105030027), Diki Adriyan (2430205030007), Ahmad Faris Al-Porisi (2430205030009), Moris Nedik (2430105030022), Muhammad Rasmi Baihaqi (2430205030008)

## Struktur repository

```
proyek-deteksi-cacat-daun/
├── data/
│   ├── images/          foto daun yang dipakai pipeline (50 citra: A_001.jpg, B_001.jpg, ...)
│   ├── paint/           salinan foto untuk diwarnai magenta saat membuat ground truth
│   ├── masks/           mask ground truth biner (PNG, nama file sama dengan foto)
│   └── metadata.csv     daftar citra: kondisi cahaya, split dev/test, nama file asli
├── src/
│   ├── pipeline.py          pipeline deteksi cacat (5 tahap)
│   ├── scratch_ops.py       Otsu, erosi, dan dilasi yang ditulis sendiri dengan NumPy
│   ├── verify_scratch.py    verifikasi scratch_ops terhadap OpenCV
│   ├── metrics.py           IoU, Dice, precision, recall
│   ├── evaluate.py          menjalankan pipeline, mencocokkan mask, merangkum metrik
│   ├── run_all.py           satu perintah untuk menjalankan dan mencetak tabel metrik
│   ├── experiments.py       perbandingan metode (hanya data dev)
│   ├── failure_analysis.py  analisis kegagalan (3 citra test dengan IoU terendah)
│   ├── debug_view.py        visualisasi tahap-tahap pipeline per citra
│   ├── bagi_kondisi.py      skrip persiapan data: membagi foto ke kondisi A/B/C
│   └── prepare_data.py      skrip persiapan data: resize, penamaan, dan split dev/test
├── results/             keluaran program (tabel, CSV, gambar)
├── requirements.txt
└── README.md
```

`bagi_kondisi.py` dan `prepare_data.py` hanya dijalankan sekali oleh penanggung jawab data dan memakai folder foto mentah di komputernya. Hasilnya sudah tersimpan di `data/`, jadi anggota lain **tidak perlu** menjalankan keduanya.

## Persyaratan dan instalasi

Butuh Python 3 dan pip. Langkah di VS Code (terminal: Terminal, New Terminal):

```bash
git clone https://github.com/erwinbernianto123/proyek-deteksi-cacat-daun.git
cd proyek-deteksi-cacat-daun

pip install -r requirements.txt
```

Jika `pip` tidak dikenali, pakai `python -m pip install -r requirements.txt` (di Windows bisa juga `py -m pip install -r requirements.txt`).

Pustaka yang dipakai: numpy, opencv-python, scikit-image, matplotlib, pandas.

## Data

`data/metadata.csv` memiliki kolom `file, kondisi, sumber, split, asli`.

| Kondisi | Arti |
|---|---|
| A | terang |
| B | redup |
| C | sangat gelap |

- **Split dev (60%)**: untuk menyetel parameter dan membandingkan metode.
- **Split test (40%)**: hanya untuk angka akhir. Jangan mengubah parameter berdasarkan hasil test.
- Pembagian ditetapkan sekali di `metadata.csv` dan tidak diubah.
- **Ground truth**: cacat diwarnai magenta murni (255, 0, 255) pada salinan foto di `data/paint/`, lalu mask biner disimpan sebagai PNG di `data/masks/` dengan nama file yang sama dengan fotonya. Citra yang belum punya mask dilewati dan dilaporkan saat evaluasi.

## Cara menjalankan

Semua perintah dijalankan dari folder utama repository.

**Satu perintah untuk menjalankan semuanya dan mencetak tabel metrik** (keseluruhan dan per kondisi cahaya):

```bash
python src/run_all.py --split test      # angka akhir untuk laporan
python src/run_all.py --split dev       # untuk menyetel parameter
```

Pilihan tambahan:

```bash
python src/run_all.py --split test --feature hue --blur median
python src/run_all.py --split dev --ops scratch     # memakai operasi buatan sendiri
python src/run_all.py --split test --no-save-masks  # tidak menyimpan mask prediksi
```

Perintah lain:

```bash
python src/verify_scratch.py                 # verifikasi Otsu/erosi/dilasi buatan sendiri vs OpenCV
python src/experiments.py                    # perbandingan metode (data dev)
python src/failure_analysis.py               # analisis kegagalan (jalankan setelah parameter dikunci)
python src/debug_view.py A_001.jpg           # lihat tahap-tahap pipeline satu citra
python src/debug_view.py --split dev --limit 5
```

## Keluaran

| File | Dibuat oleh | Isi |
|---|---|---|
| `results/metrics_<split>.csv` | run_all.py | skor per citra (IoU, Dice, precision, recall, luas, waktu) |
| `results/summary_<split>.txt` | run_all.py | tabel ringkasan keseluruhan dan per kondisi |
| `results/pred_masks/<split>/*.png` | run_all.py | mask prediksi |
| `results/verify_otsu.csv`, `results/verify_morph.csv` | verify_scratch.py | selisih piksel dan waktu: buatan sendiri vs OpenCV |
| `results/experiments_dev.csv` | experiments.py | hasil perbandingan metode |
| `results/failure/`, `results/failure_analysis.csv` | failure_analysis.py | panel gambar dan bukti 3 citra gagal |
| `results/debug/<nama>.png` | debug_view.py | visualisasi tahap pipeline |

## Pipeline

1. **Pra-proses**: resize (sisi terpanjang maksimal 1024 px) dan filter Gaussian atau median.
2. **Perbaikan kontras**: CLAHE pada kanal L (ruang warna Lab).
3. **Segmentasi**: mask daun dengan Otsu dan hole filling, lalu mask cacat dari fitur a* (atau hue) dengan ambang Otsu di dalam area daun.
4. **Morfologi**: opening, closing, dan pembuangan komponen cacat yang sangat kecil.
5. **Pengukuran**: bounding box dan luas cacat (piksel).

Fungsi utama: `segment_leaf_defect(image_bgr, params=None)` di `src/pipeline.py`. Nilai bawaan semua parameter ada di `DEFAULT_PARAMS` dan bisa diganti lewat argumen `params`.

## Metrik evaluasi

Mask prediksi dibandingkan dengan mask ground truth pada tingkat piksel (`src/metrics.py`):

- **IoU** = irisan / gabungan
- **Dice** = 2 × irisan / (jumlah luas kedua mask)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)

Aturan mask kosong: jika prediksi dan ground truth sama-sama kosong (daun sehat), semua metrik bernilai 1,0. Jika hanya salah satunya kosong, semua metrik bernilai 0,0.

Karena citra sehat bisa mengangkat rata-rata, tabel metrik dilaporkan dalam dua baris untuk tiap kelompok: **semua** citra dan hanya citra yang **ada cacat**.

## Integritas akademik

Otsu dan erosi/dilasi pada `src/scratch_ops.py` ditulis sendiri hanya dengan NumPy dan diverifikasi terhadap OpenCV (`src/verify_scratch.py`). Penggunaan AI untuk memahami konsep dan mencari bug dicantumkan pada lampiran laporan.
