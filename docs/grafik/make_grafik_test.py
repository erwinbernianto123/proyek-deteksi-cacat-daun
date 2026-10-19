"""Grafik metrik per kondisi pada citra uji -> docs/grafik/grafik_metrik_test.png
Sumber data: results/metrics_test.csv (hasil python src/run_all.py --split test).
Jalankan dari folder utama repo:  python docs/grafik/make_grafik_test.py
"""
import csv, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
rows = list(csv.DictReader(open(ROOT / "results" / "metrics_test.csv", encoding="utf-8")))
M = [("iou", "IoU"), ("dice", "Dice"), ("precision", "Precision"), ("recall", "Recall")]
groups = [("A", "A"), ("B", "B"), ("C", "C"), (None, "Keseluruhan")]
def sel(k, only):
    return [r for r in rows if (k is None or r["kondisi"] == k) and (not only or r["has_defect"] == "True")]
def agg(rs):
    return [st.mean(float(r[k]) for r in rs) for k, _ in M]

plt.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 10, "axes.linewidth": 0.8})
fills = ["#222222", "#7a7a7a", "#bdbdbd", "#ffffff"]; hatches = ["", "", "///", "..."]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
for ax, (only, title) in zip(axs, [(False, "Semua citra uji"), (True, "Hanya citra bercacat")]):
    w = 0.19
    for gi, (k, lab) in enumerate(groups):
        for mi, (val, (_, ml)) in enumerate(zip(agg(sel(k, only)), M)):
            ax.bar(gi + (mi - 1.5) * (w + 0.02), val, w, color=fills[mi], hatch=hatches[mi],
                   edgecolor="black", linewidth=0.7, label=ml if gi == 0 else None)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{lab}\n(n={len(sel(k, only))})" for k, lab in groups])
    ax.set_ylim(0, 1); ax.set_title(title, fontsize=10.5)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("Nilai metrik")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1))
out = Path(__file__).with_name("grafik_metrik_test.png")
fig.savefig(out, dpi=300); print("tersimpan:", out)
