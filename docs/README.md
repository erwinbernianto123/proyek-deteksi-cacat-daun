# docs/

Bahan dokumentasi untuk laporan: diagram alir dan grafik. Semua grafik dibuat dari file CSV di `results/` sehingga angkanya sama dengan tabel di laporan.

## diagram/

Diagram alir pipeline, dibuat di draw.io (buka file `.drawio` di https://app.diagrams.net atau aplikasi draw.io).

| File | Isi |
|---|---|
| `Diagram_Alir_Pipeline.drawio` / `.png` | versi rinci (dipakai di laporan, Gambar 4.1) |
| `Diagram_Alir_Pipeline_Sedang.drawio` / `.png` | versi sedang, satu kotak per tahap |
| `Diagram_Alir_Pipeline_Ringkas.drawio` / `.png` | versi ringkas |

Jika pipeline di `src/pipeline.py` berubah, diagram perlu diperbarui.

## grafik/

Jalankan dari folder utama repository (setelah `python src/run_all.py --split test` dan `python src/experiments.py`):

```bash
python docs/grafik/make_grafik_test.py        # grafik_metrik_test.png   <- results/metrics_test.csv
python docs/grafik/make_grafik_eksperimen.py  # eks_metode.png, eks_huethr.png   <- results/experiments_dev.csv
python docs/grafik/make_panel_hasil.py        # berhasil.png, gagal_*.png   <- pipeline + data/ + results/metrics_test.csv
```

| File | Dipakai di laporan |
|---|---|
| `grafik_metrik_test.png` | Gambar 7.1, metrik per kondisi pencahayaan |
| `eks_metode.png` | Gambar 6.1, perbandingan metode (data dev) |
| `eks_huethr.png` | Gambar 6.2, pengaruh ambang hue (data dev) |
| `berhasil.png` | Gambar 7.2, tiga hasil terbaik |
| `gagal_B_011.png`, `gagal_C_013.png`, `gagal_B_012.png` | Gambar 7.3 sampai 7.5, analisis kegagalan |

Grafik memakai warna abu-abu dan arsir (bukan warna) agar tetap terbaca saat dicetak hitam putih. Peta galat pada panel hasil memakai hijau (benar), merah (salah deteksi), dan biru (cacat terlewat).
