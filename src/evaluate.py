"""
evaluate.py  --  Evaluasi pipeline terhadap mask ground truth.

Metrik dihitung oleh metrics.py (milik Penguji / Faris); file ini hanya
menjalankan pipeline, mencocokkan mask GT, memanggil metrik, dan
merangkum hasil (keseluruhan dan per kondisi cahaya).

Dipakai oleh run_all.py, tapi bisa juga dipanggil langsung:

    from evaluate import evaluate_split, summarize, format_table
    rows = evaluate_split("dev", params={"blur": "median"})
    print(format_table(summarize(rows)))
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import calculate_all_metrics   # milik Faris
from pipeline import segment_leaf_defect

ROOT = Path(__file__).resolve().parent.parent
METRIC_NAMES = ("iou", "dice", "precision", "recall")


# --------------------------------------------------------------------------
# Membaca data
# --------------------------------------------------------------------------
def read_metadata(path=None):
    """Baca data/metadata.csv -> list of dict (kolom: file, kondisi, sumber, split, asli)."""
    path = Path(path) if path else ROOT / "data" / "metadata.csv"
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    need = {"file", "kondisi", "split"}
    if not rows or not need <= set(rows[0]):
        raise ValueError(f"metadata.csv harus punya kolom {sorted(need)}; ditemukan {list(rows[0]) if rows else 'kosong'}")
    return rows


def load_gt(masks_dir, filename):
    """Mask GT sebagai uint8 0/255 (1 kanal), atau None kalau file tidak ada."""
    stem = Path(filename).stem
    for name in (f"{stem}.png", f"{stem}_mask.png"):
        p = Path(masks_dir) / name
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)  # paksa 1 kanal
            if m is None:
                continue
            return np.where(m > 127, 255, 0).astype(np.uint8)
    return None


# --------------------------------------------------------------------------
# Evaluasi per citra dan per split
# --------------------------------------------------------------------------
def evaluate_image(img_bgr, gt, params=None):
    """Jalankan pipeline pada satu citra lalu hitung metrik. Return (dict_hasil, mask_prediksi)."""
    t0 = time.perf_counter()
    res = segment_leaf_defect(img_bgr, params)
    dt = time.perf_counter() - t0

    pred = res["mask"]
    if gt.shape != pred.shape:  # samakan ukuran (nearest agar tetap biner)
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

    m = calculate_all_metrics(pred, gt)
    m.update({
        "area_pred": res["area"],
        "area_gt": int(np.count_nonzero(gt)),
        "has_defect": bool(np.count_nonzero(gt)),
        "time_s": dt,
    })
    return m, pred


def evaluate_split(split, params=None, root=ROOT, save_masks_dir=None, verbose=True):
    """
    Evaluasi semua citra pada split ("dev" | "test" | "all").
    Return list of dict, satu per citra yang punya mask GT.
    Citra tanpa mask GT dilewati dan dilaporkan.
    """
    root = Path(root)
    meta = [r for r in read_metadata(root / "data" / "metadata.csv")
            if split == "all" or r["split"] == split]
    rows, missing = [], []
    for r in meta:
        name = r["file"]
        img = cv2.imread(str(root / "data" / "images" / name))
        gt = load_gt(root / "data" / "masks", name)
        if img is None:
            missing.append(f"{name} (citra tidak terbaca)")
            continue
        if gt is None:
            missing.append(f"{name} (mask GT tidak ada)")
            continue
        m, pred = evaluate_image(img, gt, params)
        m.update({"file": name, "kondisi": r["kondisi"], "split": r["split"]})
        rows.append(m)
        if save_masks_dir:
            out = Path(save_masks_dir)
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{Path(name).stem}.png"), pred)

    if verbose and missing:
        print(f"[PERINGATAN] {len(missing)} citra dilewati:")
        for x in missing[:10]:
            print("   -", x)
        if len(missing) > 10:
            print(f"   ... dan {len(missing) - 10} lainnya")
    return rows


# --------------------------------------------------------------------------
# Ringkasan
# --------------------------------------------------------------------------
def _mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def summarize(rows):
    """
    Ringkasan per kelompok: SEMUA, A, B, C.
    Untuk setiap kelompok ada dua baris:
      - semua citra
      - hanya citra yang punya cacat di GT (supaya citra sehat, yang otomatis
        mendapat skor 1.0 bila prediksi kosong, tidak mengangkat rata-rata)
    """
    out = []
    kondisi = sorted({r["kondisi"] for r in rows})
    groups = [("SEMUA", rows)] + [(k, [r for r in rows if r["kondisi"] == k]) for k in kondisi]
    for name, grp in groups:
        for label, sub in (("semua", grp), ("ada cacat", [r for r in grp if r["has_defect"]])):
            entry = {"kondisi": name, "subset": label, "n": len(sub)}
            for k in METRIC_NAMES:
                entry[k] = _mean(sub, k)
            entry["waktu_s"] = _mean(sub, "time_s")
            out.append(entry)
    return out


def format_table(summary):
    """Tabel teks rapi untuk dicetak di terminal / ditempel ke laporan."""
    head = f"{'Kondisi':<8}{'Subset':<11}{'n':>4}{'IoU':>8}{'Dice':>8}{'Prec':>8}{'Recall':>8}{'Waktu(s)':>10}"
    lines = [head, "-" * len(head)]
    def f(v):
        return f"{v:8.3f}" if v == v else f"{'-':>8}"  # nan -> "-"

    for e in summary:
        if e["n"] == 0:
            lines.append(f"{e['kondisi']:<8}{e['subset']:<11}{0:>4}" + f"{'-':>8}" * 4 + f"{'-':>10}")
            continue
        lines.append(f"{e['kondisi']:<8}{e['subset']:<11}{e['n']:>4}"
                     f"{f(e['iou'])}{f(e['dice'])}{f(e['precision'])}{f(e['recall'])}"
                     f"{e['waktu_s']:>10.3f}")
    return "\n".join(lines)


def save_csv(rows, path):
    """Simpan hasil per citra ke CSV (bahan tabel/grafik Moris dan analisis kegagalan Faris)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["file", "kondisi", "split", *METRIC_NAMES, "area_pred", "area_gt", "has_defect", "time_s"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{k: round(r[k], 4) for k in (*METRIC_NAMES, "time_s")}})
    return path
