"""Grafik perbandingan metode dan ambang hue (data dev)
-> docs/grafik/eks_metode.png dan docs/grafik/eks_huethr.png
Sumber data: results/experiments_dev.csv (hasil python src/experiments.py).
Jalankan dari folder utama repo:  python docs/grafik/make_grafik_eksperimen.py
"""
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).parent
plt.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 9})
d = pd.read_csv(ROOT / "results" / "experiments_dev.csv")
S = d.groupby(["eksperimen", "metode"])[["iou", "dice", "precision", "recall", "time_s"]].mean()
D = d[d.has_defect].groupby(["eksperimen", "metode"])[["iou", "dice", "precision", "recall"]].mean()
n_all, n_def = d[d.eksperimen == d.eksperimen.iloc[0]].file.nunique(), d[(d.eksperimen == d.eksperimen.iloc[0]) & d.has_defect].file.nunique()

order = [("Fitur cacat: a* vs hue vs hue_abs", ["a_star", "hue", "hue_abs(default)"], ["a*", "hue relatif", "hue absolut"], "Fitur cacat"),
         ("Pra-proses: Gaussian vs median", ["gaussian", "median"], ["Gaussian", "Median"], "Pra-proses"),
         ("Ambang cacat: Otsu vs median_offset", ["otsu", "median_offset", "gabungan a_star"], ["Otsu", "Median+offset", "Gabungan"], "Ambang cacat (fitur a*)")]
fig, ax = plt.subplots(figsize=(6.2, 3.6)); x = 0; xt = []; xl = []; gl = []
for e, ms, lb, grp in order:
    s0 = x
    for m, l in zip(ms, lb):
        for dx, val, kw in ((-.2, S.loc[(e, m), "iou"], dict(color="white", hatch="///")), (.2, D.loc[(e, m), "iou"], dict(color="#555555"))):
            ax.bar(x + dx, val, .38, edgecolor="black", lw=.8, **kw)
            ax.text(x + dx, val + .01, f"{val:.2f}", ha="center", fontsize=7)
        xt.append(x); xl.append(l); x += 1
    gl.append(((s0 + x - 1) / 2, grp)); x += .6
ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=30, ha="right", fontsize=8)
for gx, g in gl: ax.text(gx, -.30, g, ha="center", fontsize=9, transform=ax.get_xaxis_transform())
ax.set_ylabel("IoU rata-rata (data dev)"); ax.set_ylim(0, .6); ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch="///", label=f"Semua citra (n={n_all})"),
                   Patch(facecolor="#555555", edgecolor="black", label=f"Citra bercacat (n={n_def})")], frameon=False, fontsize=8, loc="upper right")
plt.tight_layout(); plt.subplots_adjust(bottom=.3); plt.savefig(OUT / "eks_metode.png", dpi=200); plt.close()

e = "Ambang hue_abs (hue_thr)"; ms = ["hue_thr=25", "hue_thr=30(default)", "hue_thr=35"]
fig, ax = plt.subplots(figsize=(4.8, 3))
for k, (col, ls, lab) in enumerate([("iou", "-", "IoU"), ("precision", "--", "Precision"), ("recall", ":", "Recall")]):
    ax.plot([25, 30, 35], [D.loc[(e, m), col] for m in ms], ls, color="black", marker=["o", "s", "^"][k], label=lab, lw=1.2)
ax.set_xticks([25, 30, 35]); ax.set_xlabel("Ambang hue (hue_thr)"); ax.set_ylabel("Nilai rata-rata, citra bercacat (dev)")
ax.set_ylim(.3, .85); ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "eks_huethr.png", dpi=200); plt.close()
print("tersimpan di", OUT)
