 · PY
"""experiments.py - perbandingan metode. HANYA memakai data dev.
 
Jalankan dari folder utama repo:  python src/experiments.py
Hasil: tabel di layar + results/experiments_dev.csv
"""
import csv
import os
import sys
from collections import defaultdict
 
import cv2
import numpy as np
 
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
 
from pipeline import segment_leaf_defect, DEFAULT_PARAMS  # milik ketua
import metrics                                            # milikmu
 
IMG_DIR = os.path.join(ROOT, "data", "images")
MASK_DIR = os.path.join(ROOT, "data", "masks")
META = os.path.join(ROOT, "data", "metadata.csv")
OUT_CSV = os.path.join(ROOT, "results", "experiments_dev.csv")
 
# ---------------------------------------------------------------
# PERBANDINGAN. Tiap eksperimen mengubah SATU parameter saja,
# sisanya tetap DEFAULT_PARAMS. Nama kunci ("blur", "threshold",
# "feature") dan nilainya HARUS disamakan dengan DEFAULT_PARAMS
# di pipeline.py.
# ---------------------------------------------------------------
EXPERIMENTS = {
    "Pra-proses: Gaussian vs median": [
        ("gaussian", {"blur": "gaussian"}),
        ("median", {"blur": "median"}),
    ],
    "Threshold: Otsu vs median_offset": [
        ("otsu", {"threshold": "otsu"}),
        ("median_offset", {"threshold": "median_offset"}),
    ],
    "Fitur: a* vs hue": [
        ("a_star", {"feature": "a_star"}),
        ("hue", {"feature": "hue"}),
    ],
}
 
METRIC_NAMES = ["iou", "dice", "precision", "recall"]
 
 
# --- Penyesuaian ke kode temanmu (ubah di sini kalau beda) -------
def get_mask(result):
    """Ambil mask prediksi dari keluaran segment_leaf_defect."""
    if isinstance(result, dict):
        return result["mask"]
    if isinstance(result, (tuple, list)):
        return result[0]
    return result
 
 
def score(pred, gt):
    """Panggil fungsi di metrics.py. Sesuaikan nama fungsinya."""
    return {
        "iou": metrics.iou(pred, gt),
        "dice": metrics.dice(pred, gt),
        "precision": metrics.precision(pred, gt),
        "recall": metrics.recall(pred, gt),
    }
# -----------------------------------------------------------------
 
 
def load_dev_rows():
    with open(META, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "dev"]
    assert rows, "Tidak ada baris dev di metadata.csv"
    assert all(r["split"] == "dev" for r in rows)  # jaga-jaga: test dilarang
    return rows
 
 
def load_pair(fname):
    stem = os.path.splitext(fname)[0]
    img = cv2.imread(os.path.join(IMG_DIR, fname))
    gt = cv2.imread(os.path.join(MASK_DIR, stem + ".png"), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        return None, None
    return img, gt > 0
 
 
def main():
    rows = load_dev_rows()
    pairs = []
    for r in rows:
        img, gt = load_pair(r["file"])
        if img is None:
            print("LEWATI (citra/mask tidak ada):", r["file"])
            continue
        pairs.append((r, img, gt))
    print(f"Data dev dipakai: {len(pairs)} dari {len(rows)} citra\n")
 
    all_rows = []
    summary = {}
    for exp_name, variants in EXPERIMENTS.items():
        summary[exp_name] = []
        for label, override in variants:
            params = dict(DEFAULT_PARAMS)
            params.update(override)
            per_cond = defaultdict(list)
            for r, img, gt in pairs:
                pred = get_mask(segment_leaf_defect(img, params)) > 0
                if pred.shape != gt.shape:
                    gt_use = cv2.resize(gt.astype(np.uint8), pred.shape[::-1],
                                        interpolation=cv2.INTER_NEAREST) > 0
                else:
                    gt_use = gt
                s = score(pred, gt_use)
                vals = [s[m] for m in METRIC_NAMES]
                per_cond[r["kondisi"]].append(vals)
                per_cond["ALL"].append(vals)
                all_rows.append([exp_name, label, r["file"], r["kondisi"]] + vals)
            means = {c: np.mean(v, axis=0) for c, v in per_cond.items()}
            summary[exp_name].append((label, means))
 
    # --- cetak tabel ---
    for exp_name, items in summary.items():
        print("=" * 72)
        print(exp_name, "(data dev)")
        print(f"{'metode':<15}{'kondisi':<9}" + "".join(f"{m:>10}" for m in METRIC_NAMES))
        for label, means in items:
            for c in ["ALL"] + sorted(k for k in means if k != "ALL"):
                print(f"{label:<15}{c:<9}" + "".join(f"{x:>10.3f}" for x in means[c]))
        best = max(items, key=lambda it: it[1]["ALL"][0])
        print(f"--> Unggul (IoU rata-rata): {best[0]}\n")
 
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eksperimen", "metode", "file", "kondisi"] + METRIC_NAMES)
        w.writerows(all_rows)
    print("Tersimpan:", OUT_CSV)
 
 
if __name__ == "__main__":
    main()
