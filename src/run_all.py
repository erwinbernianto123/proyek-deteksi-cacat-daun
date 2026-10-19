"""
run_all.py  --  Satu perintah untuk menjalankan semuanya dan mencetak tabel metrik.

Dari root repo:
    python src/run_all.py --split test          # angka akhir (laporan)
    python src/run_all.py --split dev           # untuk menyetel parameter
    python src/run_all.py --split test --feature hue --blur median

Keluaran:
  - tabel metrik keseluruhan dan per kondisi cahaya (A/B/C) di terminal
  - results/metrics_<split>.csv      : skor per citra
  - results/summary_<split>.txt      : tabel ringkasan
  - results/pred_masks/<split>/*.png : mask prediksi

Aturan tugas: split "dev" dipakai untuk menyetel parameter, split "test"
hanya untuk angka akhir. Jangan mengubah parameter berdasarkan hasil test.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate_split, format_table, save_csv, summarize
from pipeline import DEFAULT_PARAMS

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Jalankan pipeline + evaluasi pada satu split")
    ap.add_argument("--split", choices=["dev", "test", "all"], required=True)
    ap.add_argument("--blur", choices=["gaussian", "median"], default=DEFAULT_PARAMS["blur"])
    ap.add_argument("--feature", choices=["hue_abs", "a_star", "hue"], default=DEFAULT_PARAMS["feature"])
    ap.add_argument("--ops", choices=["opencv", "scratch"], default=DEFAULT_PARAMS["ops"],
                    help="opencv = fungsi OpenCV, scratch = src/scratch_ops.py")
    ap.add_argument("--otsu-offset", type=float, default=DEFAULT_PARAMS["otsu_offset"])
    ap.add_argument("--no-save-masks", action="store_true")
    args = ap.parse_args()

    params = {"blur": args.blur, "feature": args.feature, "ops": args.ops,
              "otsu_offset": args.otsu_offset}

    if args.split == "test":
        print("CATATAN: split TEST = angka akhir. Jangan ubah parameter berdasarkan hasil ini.\n")
    print(f"Split: {args.split} | Parameter: {params}\n")

    out_masks = None if args.no_save_masks else ROOT / "results" / "pred_masks" / args.split
    rows = evaluate_split(args.split, params, root=ROOT, save_masks_dir=out_masks)
    if not rows:
        print("Tidak ada citra yang bisa dievaluasi (cek data/masks/ dan metadata.csv).")
        sys.exit(1)

    table = format_table(summarize(rows))
    print(table)

    results = ROOT / "results"
    csv_path = save_csv(rows, results / f"metrics_{args.split}.csv")
    (results / f"summary_{args.split}.txt").write_text(
        f"split={args.split} params={params}\n\n{table}\n", encoding="utf-8")
    print(f"\nTersimpan: {csv_path.relative_to(ROOT)}, results/summary_{args.split}.txt")


if __name__ == "__main__":
    main()
