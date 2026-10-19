ts · PY
"""
experiments.py  --  Perbandingan metode. HANYA memakai data dev.
 
Memakai evaluate_split() milik ketua, jadi cara membaca citra, mencocokkan
mask GT, dan menghitung metrik SAMA dengan evaluasi akhir di data test.
 
Jalankan dari folder utama repo:   python src/experiments.py
Hasil: tabel di layar + results/experiments_dev.csv
"""
import csv
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate_split, summarize, format_table, METRIC_NAMES
 
ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "results" / "experiments_dev.csv"
 
# Tiap eksperimen mengubah parameter tertentu, sisanya tetap DEFAULT_PARAMS.
# Semua nama kunci HARUS ada di DEFAULT_PARAMS (pipeline.py menolak kunci asing).
#
# Ambang cacat di pipeline = max(Otsu + otsu_offset, batas median + min_delta_a),
# jadi default-nya gabungan. Untuk membandingkan murni:
#   Otsu saja          -> min_delta_a = -128  (batas median jadi tidak aktif)
#   median_offset saja -> otsu_offset = -255  (ambang Otsu jadi tidak aktif)
EXPERIMENTS = {
    "Pra-proses: Gaussian vs median": [
        ("gaussian", {"blur": "gaussian"}),
        ("median", {"blur": "median"}),
    ],
    "Ambang cacat: Otsu vs median_offset": [
        ("otsu", {"min_delta_a": -128}),
        ("median_offset", {"otsu_offset": -255}),
        ("gabungan(default)", {}),
    ],
    "Fitur cacat: a* vs hue": [
        ("a_star", {"feature": "a_star"}),
        ("hue", {"feature": "hue"}),
    ],
}
 
 
def get(summary, kondisi, subset):
    for e in summary:
        if e["kondisi"] == kondisi and e["subset"] == subset:
            return e
    return None
 
 
def main():
    all_rows = []
    first = True
    for exp_name, variants in EXPERIMENTS.items():
        print("=" * 70)
        print(exp_name, "(data dev)")
        results = []
        for label, override in variants:
            rows = evaluate_split("dev", params=override, verbose=first)  # "dev" dikunci
            first = False
            assert all(r["split"] == "dev" for r in rows)
            summ = summarize(rows)
            results.append((label, summ))
            print(f"\n[{label}]  parameter diubah: {override or '(tidak ada)'}")
            print(format_table(summ))
            for r in rows:
                all_rows.append({"eksperimen": exp_name, "metode": label, **r})
 
        # pemenang berdasarkan IoU rata-rata, keseluruhan
        for subset in ("semua", "ada cacat"):
            scored = [(lab, get(s, "SEMUA", subset)) for lab, s in results]
            scored = [(lab, e["iou"]) for lab, e in scored if e and e["n"] > 0]
            if scored:
                best = max(scored, key=lambda t: t[1])
                print(f"--> Unggul (IoU, subset '{subset}'): {best[0]} ({best[1]:.3f})")
        print()
 
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["eksperimen", "metode", "file", "kondisi", "split", *METRIC_NAMES,
            "area_pred", "area_gt", "has_defect", "time_s"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({**r, **{k: round(r[k], 4) for k in (*METRIC_NAMES, "time_s")}})
    print("Tersimpan:", OUT_CSV.relative_to(ROOT))
 
 
if __name__ == "__main__":
    main()
