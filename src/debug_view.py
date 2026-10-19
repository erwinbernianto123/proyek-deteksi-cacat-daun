"""
debug_view.py  --  Visualisasi tahap-tahap pipeline untuk satu citra.

Contoh pemakaian (dari root repo):
    python src/debug_view.py A_001.jpg
    python src/debug_view.py A_001.jpg B_003.jpg --feature hue
    python src/debug_view.py --split dev --limit 5

Hasil disimpan di results/debug/<nama>.png (tanpa membuka jendela).
Kalau ada mask ground truth di data/masks/, ikut ditampilkan.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")  # simpan ke file, tidak perlu GUI
import matplotlib.pyplot as plt

from pipeline import DEFAULT_PARAMS, segment_leaf_defect

ROOT = Path(__file__).resolve().parent.parent


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay(img_bgr, mask, color=(0, 0, 255), alpha=0.5):
    out = img_bgr.copy()
    m = mask > 0
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


def load_gt(masks_dir, filename):
    """Cari mask GT berdasarkan nama file; return array 0/255 atau None."""
    stem = Path(filename).stem
    for name in (f"{stem}.png", f"{stem}_mask.png", f"{stem}.jpg"):
        path = Path(masks_dir) / name
        if path.exists():
            m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return np.where(m > 127, 255, 0).astype(np.uint8)
    return None


def render_debug(img_bgr, result, gt=None, title=""):
    """Return figure matplotlib berisi tahap-tahap pipeline."""
    H, W = img_bgr.shape[:2]
    dbg = result["debug"]
    enhanced = cv2.resize(dbg["enhanced"], (W, H))
    score = cv2.resize(dbg["score"], (W, H))

    panels = [
        ("1. Asli", _rgb(img_bgr), None),
        ("2. Setelah blur + CLAHE", _rgb(enhanced), None),
        ("3. Mask daun", result["leaf_mask"], "gray"),
        (f"4. Skor cacat (ambang={dbg['threshold']:.0f})", score, "magma"),
        ("5. Mask cacat (setelah morfologi)", result["mask"], "gray"),
    ]
    final = overlay(img_bgr, result["mask"])
    for (x, y, w, h) in result["boxes"]:
        cv2.rectangle(final, (x, y), (x + w, y + h), (0, 255, 255), max(2, W // 300))
    panels.append((f"6. Hasil (luas={result['area']} px, {100 * result['area_ratio']:.1f}% daun)",
                   _rgb(final), None))
    if gt is not None:
        gt_r = gt if gt.shape == (H, W) else cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        panels.append(("7. Ground truth", _rgb(overlay(img_bgr, gt_r, (255, 0, 255))), None))

    n = len(panels)
    cols = 4 if n > 6 else 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, im, cmap) in zip(axes, panels):
        ax.imshow(im, cmap=cmap)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def _files_from_split(split, limit):
    import csv
    with open(ROOT / "data" / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if split == "all" or r["split"] == split]
    names = [r["file"] for r in rows]
    return names[:limit] if limit else names


def main():
    ap = argparse.ArgumentParser(description="Visualisasi tahap pipeline")
    ap.add_argument("files", nargs="*", help="nama file di data/images/ (mis. A_001.jpg)")
    ap.add_argument("--split", choices=["dev", "test", "all"], help="ambil file dari metadata.csv")
    ap.add_argument("--limit", type=int, default=0, help="batasi jumlah file untuk --split")
    ap.add_argument("--blur", choices=["gaussian", "median"], default=DEFAULT_PARAMS["blur"])
    ap.add_argument("--feature", choices=["a_star", "hue"], default=DEFAULT_PARAMS["feature"])
    ap.add_argument("--ops", choices=["opencv", "scratch"], default=DEFAULT_PARAMS["ops"])
    ap.add_argument("--out", default=str(ROOT / "results" / "debug"))
    args = ap.parse_args()

    files = list(args.files)
    if args.split:
        files += _files_from_split(args.split, args.limit)
    if not files:
        ap.error("beri nama file atau --split")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {"blur": args.blur, "feature": args.feature, "ops": args.ops}

    for name in files:
        path = ROOT / "data" / "images" / name
        img = cv2.imread(str(path))
        if img is None:
            print(f"[LEWAT] tidak bisa membaca {path}")
            continue
        res = segment_leaf_defect(img, params)
        gt = load_gt(ROOT / "data" / "masks", name)
        fig = render_debug(img, res, gt, title=f"{name}  |  {params}")
        out = out_dir / f"{Path(name).stem}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"[OK] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
