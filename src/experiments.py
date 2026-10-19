
import csv
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate_split, summarize, format_table, METRIC_NAMES
from pipeline import DEFAULT_PARAMS
 
ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "results" / "experiments_dev.csv"
 
# Tiap eksperimen mengubah parameter tertentu, sisanya tetap DEFAULT_PARAMS.
# Semua nama kunci HARUS ada di DEFAULT_PARAMS (pipeline.py menolak kunci asing).
#
# PENTING: default fitur cacat sekarang "hue_abs" (cacat bila hue < hue_thr).
# Parameter otsu_offset, min_delta_a, min_delta_hue HANYA berlaku untuk fitur
# "a_star" / "hue". Karena itu eksperimen yang memakai parameter tersebut harus
# menuliskan feature secara eksplisit, kalau tidak semua variannya identik.
#
# Ambang cacat untuk a_star = max(Otsu + otsu_offset, batas median + min_delta_a).
# Untuk membandingkan murni:
#   Otsu saja          -> min_delta_a = -128  (batas median jadi tidak aktif)
#   median_offset saja -> otsu_offset = -255  (ambang Otsu jadi tidak aktif)
A_STAR = {"feature": "a_star"}
 
EXPERIMENTS = {
    "Pra-proses: Gaussian vs median": [
        ("gaussian", {"blur": "gaussian"}),
        ("median", {"blur": "median"}),
    ],
    "Ambang cacat: Otsu vs median_offset": [
        ("otsu", {**A_STAR, "min_delta_a": -128}),
        ("median_offset", {**A_STAR, "otsu_offset": -255}),
        ("gabungan a_star", {**A_STAR}),
    ],
    "Fitur cacat: a* vs hue vs hue_abs": [
        ("a_star", {"feature": "a_star"}),
        ("hue", {"feature": "hue"}),
        ("hue_abs(default)", {"feature": "hue_abs"}),
    ],
    # OPSIONAL (boleh dihapus): sensitivitas ambang hue_abs di data dev.
    # Kalau ada nilai yang jelas lebih baik dari 30, kabari ketua dulu
    # supaya DEFAULT_PARAMS diubah sebelum split test dijalankan.
    "Ambang hue_abs (hue_thr)": [
        ("hue_thr=25", {"feature": "hue_abs", "hue_thr": 25}),
        ("hue_thr=30(default)", {"feature": "hue_abs", "hue_thr": 30}),
        ("hue_thr=35", {"feature": "hue_abs", "hue_thr": 35}),
    ],
}
 
# Parameter yang tidak berpengaruh bila feature == "hue_abs"
ONLY_FOR_RELATIVE = {"otsu_offset", "min_delta_a", "min_delta_hue"}
 
 
def check_override(exp_name, label, override):
    """Cegah eksperimen 'diam-diam' tidak valid: varian mengubah parameter
    yang diabaikan oleh fitur yang sedang aktif."""
    feat = {**DEFAULT_PARAMS, **override}["feature"]
    useless = ONLY_FOR_RELATIVE & set(override)
    if feat == "hue_abs" and useless:
        raise ValueError(
            f"[{exp_name} / {label}] parameter {sorted(useless)} tidak berpengaruh "
            f"pada feature='hue_abs'. Tambahkan 'feature': 'a_star' atau 'hue'.")
 
 
def get(summary, kondisi, subset):
    for e in summary:
        if e["kondisi"] == kondisi and e["subset"] == subset:
            return e
    return None
 
 
def main():
    all_rows = []
    first = True
    for exp_name, variants in EXPERIMENTS.items():
        for label, override in variants:
            check_override(exp_name, label, override)
 
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
 
        # Pemenang berdasarkan IoU rata-rata. Utamakan subset "ada cacat":
        # citra sehat dengan prediksi kosong otomatis IoU = 1.0, jadi rata-rata
        # "semua" bisa menyesatkan.
        winners = {}
        for subset in ("ada cacat", "semua"):
            scored = [(lab, get(s, "SEMUA", subset)) for lab, s in results]
            scored = [(lab, e["iou"]) for lab, e in scored if e and e["n"] > 0]
            if scored:
                best = max(scored, key=lambda t: t[1])
                winners[subset] = best[0]
                print(f"--> Unggul (IoU, subset '{subset}'): {best[0]} ({best[1]:.3f})")
        if len(set(winners.values())) > 1:
            print("!!  Pemenang berbeda antar subset. Pakai 'ada cacat' sebagai "
                  "acuan utama dan jelaskan di laporan.")
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
