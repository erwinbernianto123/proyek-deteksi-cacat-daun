"""Panel hasil per citra uji (asli, [mask daun], ground truth, prediksi, peta galat)
-> docs/grafik/berhasil.png dan docs/grafik/gagal_<nama>.png
Memakai pipeline di src/pipeline.py dengan parameter bawaan dan data/images, data/masks.
Jalankan dari folder utama repo:  python docs/grafik/make_panel_hasil.py
Ubah daftar BERHASIL / GAGAL di bawah untuk memilih citra lain.
"""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from pipeline import segment_leaf_defect
plt.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 9})
BERHASIL = ["A_018.jpg", "B_015.jpg", "B_014.jpg"]   # IoU tertinggi pada citra bercacat
GAGAL = ["B_011.jpg", "C_013.jpg", "B_012.jpg"]      # tiga jenis kegagalan (failure_analysis.csv)
metrics = pd.read_csv(ROOT / "results" / "metrics_test.csv").set_index("file")

def load(n):
    img = cv2.imread(str(ROOT / "data" / "images" / n))
    gt = cv2.imread(str(ROOT / "data" / "masks" / (Path(n).stem + ".png")), 0) > 127
    return img[:, :, ::-1], gt, segment_leaf_defect(img)

def galat(img, gt, pr):  # hijau = benar, merah = salah deteksi, biru = cacat terlewat
    o = (img * 0.55).astype(np.uint8)
    o[gt & pr] = (0, 200, 0); o[pr & ~gt] = (220, 0, 0); o[gt & ~pr] = (30, 90, 255)
    return o

def panel(files, out, dengan_mask_daun):
    cols = ["Citra asli"] + (["Mask daun"] if dengan_mask_daun else []) + ["Ground truth", "Prediksi akhir", "Galat"]
    fig, ax = plt.subplots(len(files), len(cols), figsize=(7.0, (2.35 if dengan_mask_daun else 1.9) * len(files)), squeeze=False)
    for i, f in enumerate(files):
        img, gt, r = load(f); pr = r["mask"] > 0
        ims = [img] + ([r["leaf_mask"] > 0] if dengan_mask_daun else []) + [gt, pr, galat(img, gt, pr)]
        for j, (a, im) in enumerate(zip(ax[i], ims)):
            a.imshow(im, cmap="gray"); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values(): s.set_linewidth(.5)
            if i == 0: a.set_title(cols[j], fontsize=9)
        ax[i][0].set_ylabel(f"{f[:-4]}\nIoU {metrics.loc[f, 'iou']:.3f}", fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()

panel(BERHASIL, OUT / "berhasil.png", False)
for f in GAGAL:
    panel([f], OUT / f"gagal_{f[:-4]}.png", True)
print("tersimpan di", OUT)
