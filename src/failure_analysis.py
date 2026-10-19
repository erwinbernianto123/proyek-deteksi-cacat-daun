ure analysis · PY
"""
failure_analysis.py  --  Analisis kegagalan: 3 citra TEST dengan IoU terendah.
 
Jalankan SETELAH parameter pipeline dikunci (jangan dipakai untuk menyetel
parameter; data test hanya untuk angka akhir dan analisis ini):
 
    python src/failure_analysis.py
 
Keluaran:
    results/failure/failure_<urutan>_<nama>.png   panel tahap pipeline + peta galat
    results/failure_analysis.csv                  bukti per citra + kolom isian manual
 
Skrip ini mengumpulkan BUKTI (angka + gambar), menduga tahap penyebab, lalu
menuliskan draf `penyebab` dan `usulan_perbaikan` dari angka-angka itu.
Draf ditulis otomatis; BACA gambar di results/failure/ dan sunting CSV bila
ada yang tidak sesuai dengan apa yang terlihat. Usulan perbaikan hanya boleh
diuji pada data dev, bukan dipakai untuk menyetel parameter di data test.
 
Catatan fitur default "hue_abs": cacat bila hue < hue_thr (default 30).
Di result["debug"], "score" berisi kanal hue (0-180) dan "threshold" = hue_thr.
"""
import csv
import sys
from pathlib import Path
 
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics
from evaluate import evaluate_split, load_gt
from pipeline import DEFAULT_PARAMS, segment_leaf_defect
 
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "failure"
OUT_CSV = ROOT / "results" / "failure_analysis.csv"
N_WORST = 3          # minimal 3 sesuai brief
NAN = float("nan")
 
 
def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
 
 
def overlay(img, mask, color, alpha=0.55):
    out = img.copy()
    m = mask > 0
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out
 
 
def error_map(img, pred, gt):
    """Hijau = benar (TP), merah = salah deteksi (FP), biru = terlewat (FN)."""
    out = img.copy()
    for mask, color in ((pred & gt, (0, 200, 0)), (pred & ~gt, (0, 0, 255)), (~pred & gt, (255, 80, 0))):
        out[mask] = (0.35 * out[mask] + 0.65 * np.array(color)).astype(np.uint8)
    return out
 
 
def guess_stage(gt_area, pred_area, gt_in_leaf, m, rec_raw):
    """Dugaan tahap penyebab. Hanya petunjuk; harus dicek manual pada gambar."""
    if gt_area > 0 and gt_in_leaf == gt_in_leaf and gt_in_leaf < 0.8:
        return "3a mask daun: sebagian cacat berada di luar mask daun"
    if gt_area > 0 and m["recall"] < 0.5:
        if rec_raw - m["recall"] > 0.2:
            return "4 morfologi: cacat sempat terdeteksi lalu terhapus"
        return "3b segmentasi cacat: cacat tidak terdeteksi (ambang/fitur)"
    if pred_area > 0 and m["precision"] < 0.5:
        if gt_area == 0:
            return "3b segmentasi cacat: citra tanpa cacat tetapi ada deteksi"
        return "3b segmentasi cacat: banyak area salah terdeteksi"
    return "selisih di tepi cacat (batas tidak presisi)"
 
 
def explain(kondisi, m, gt_area, pred_area, gt_in_leaf, rec_raw, sep, kecerahan, tahap):
    """Draf penyebab dan usulan perbaikan, disusun dari angka bukti."""
    feat = DEFAULT_PARAMS["feature"]
    hue_abs = feat == "hue_abs"
    thr = DEFAULT_PARAMS["hue_thr"]
    skala = "0-180" if hue_abs else "0-255"
    gelap = (kondisi == "C") or (kecerahan == kecerahan and kecerahan < 70)
    note_gelap = (f" Citra termasuk kondisi gelap (kecerahan rata-rata daun {kecerahan:.0f}/255), "
                  "sehingga kontras warna rendah"
                  + (" dan nilai hue tidak stabil." if hue_abs else ".")) if gelap else ""
    p, r = m["precision"], m["recall"]
 
    if tahap.startswith("3a"):
        sebab = (f"Hanya {gt_in_leaf * 100:.0f}% area cacat (ground truth) berada di dalam mask daun, "
                 "sehingga cacat di luar mask daun mustahil terdeteksi. Penyebab umum: cacat di tepi "
                 "daun ikut terpotong saat mask daun dibuat (Otsu pada saturasi lalu dikikis di tepi), "
                 "atau mask daun salah memilih objek pada foto sangat gelap. "
                 "Periksa juga apakah mask ground truth sudah benar." + note_gelap)
        if DEFAULT_PARAMS["leaf_use_hull"]:
            usul = ("leaf_use_hull sudah aktif secara default. Periksa panel 'Mask daun' pada gambar; "
                    "kalau komponen daun salah pilih, ini kelemahan segmentasi daun pada foto gelap. "
                    "Perbaikan (mis. kecilkan leaf_border_erode) hanya boleh diuji pada data dev.")
        else:
            usul = ("Aktifkan leaf_use_hull=True atau kecilkan leaf_border_erode agar cacat di tepi masuk "
                    "area pencarian; uji dulu pada data dev.")
    elif tahap.startswith("4"):
        mda = int(DEFAULT_PARAMS["min_defect_area"])
        sebab = (f"Cacat sempat terdeteksi sebelum morfologi (recall {rec_raw:.2f}) tetapi turun menjadi "
                 f"{r:.2f} setelah morfologi, jadi opening dan pembuangan komponen kecil menghapus "
                 "cacat yang kecil atau tipis." + note_gelap)
        usul = (f"Kecilkan open_ksize atau min_defect_area (mis. {mda} menjadi {max(mda // 2, 1)}); "
                "uji pada data dev.")
    elif "tidak terdeteksi" in tahap:
        if hue_abs:
            sep_txt = (f" Selisih rata-rata hue daun sehat dan area cacat hanya {sep:.0f} "
                       f"(skala {skala}), jadi banyak piksel cacat memiliki hue di atas ambang {thr}. "
                       "Kemungkinan cacat berwarna hijau-kekuningan di sekitar hue 30-35."
                       if sep == sep else "")
            sebab = (f"Cacat berada di dalam daun tetapi hue-nya tidak di bawah {thr} (recall {r:.2f})."
                     f"{sep_txt}" + note_gelap)
            usul = ("Periksa panel kanal hue. Kalau cacat memang berhue 30-35, menaikkan hue_thr akan "
                    "menangkapnya tetapi berisiko menandai daun sehat sebagai cacat. Uji hanya pada data "
                    "dev dan kabari ketua bila ada nilai yang jelas lebih baik, supaya DEFAULT_PARAMS "
                    "diubah sebelum split test dijalankan.")
        else:
            sep_txt = (f" Pemisahan skor cacat dan daun sehat hanya {sep:.0f} (skala {skala}), artinya fitur "
                       f"{feat} kurang membedakan cacat dari daun." if sep == sep else "")
            sebab = (f"Cacat berada di dalam daun tetapi tidak melewati ambang (recall {r:.2f}).{sep_txt}"
                     + note_gelap)
            usul = (f"Turunkan ambang (otsu_offset negatif atau min_delta_a lebih kecil) atau coba feature="
                    f"{'hue' if feat == 'a_star' else 'a_star'}"
                    + ("; naikkan clahe_clip untuk memperkuat kontras" if gelap else "")
                    + ". Uji pada data dev.")
    elif "tanpa cacat" in tahap:
        sebab = (f"Ground truth kosong (citra sehat), tetapi pipeline mendeteksi {pred_area} piksel. "
                 + (f"Piksel daun dengan hue di bawah {thr} terbaca sebagai cacat; kemungkinan bayangan, "
                    "tepi daun, atau variasi warna alami. "
                    if hue_abs else
                    "Kemungkinan bayangan, urat daun, atau variasi warna alami terbaca sebagai cacat "
                    "karena ambang terlalu rendah. ")
                 + "Periksa juga apakah ground truth memang kosong." + note_gelap)
        usul = ((f"Turunkan hue_thr (mis. dari {thr}) atau naikkan min_defect_area; " if hue_abs else
                 "Naikkan min_delta_a atau otsu_offset (positif) dan min_defect_area; ")
                + "uji pada data dev agar recall di citra bercacat tidak turun.")
    elif "salah terdeteksi" in tahap:
        sebab = (f"Prediksi ({pred_area} px) jauh lebih luas daripada ground truth ({gt_area} px), precision "
                 f"{p:.2f}. "
                 + (f"Area daun dengan hue di bawah {thr} (bayangan, tepi, atau daun yang memang "
                    "kekuningan tetapi tidak dianotasi) ikut terdeteksi." if hue_abs else
                    "Area non-cacat (bayangan, urat daun, atau tepi) ikut terdeteksi.")
                 + note_gelap)
        usul = ((f"Turunkan hue_thr (mis. dari {thr}) atau kecilkan close_ksize; " if hue_abs else
                 "Naikkan min_delta_a atau otsu_offset (positif), atau kecilkan close_ksize; ")
                + "uji pada data dev.")
    else:
        sebab = (f"Lokasi cacat sudah tepat (precision {p:.2f}, recall {r:.2f}) tetapi batasnya tidak presisi. "
                 "Wajar karena ground truth digambar manual dan batas cacat tidak tegas." + note_gelap)
        usul = "Sesuaikan close_ksize atau open_ksize; atau cantumkan sebagai keterbatasan anotasi manual."
    return sebab, usul
 
 
def analyse(row, img, gt, res):
    pred = res["mask"] > 0
    if gt.shape != pred.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    gtb = gt > 0
    leaf = res["leaf_mask"] > 0
    raw = res["debug"]["raw_defect"] > 0
 
    m = metrics.calculate_all_metrics(pred, gtb)
    gt_area, pred_area = int(gtb.sum()), int(pred.sum())
    gt_in_leaf = float((gtb & leaf).sum() / gt_area) if gt_area else NAN
    rec_raw = metrics.recall(raw, gtb)
 
    # bukti tahap 2 / 3b: kecerahan daun dan pemisahan skor cacat vs daun sehat.
    # hue_abs: cacat = hue RENDAH, jadi selisih dihitung (sehat - cacat) agar positif.
    # a_star/hue: cacat = skor TINGGI, selisih (cacat - sehat).
    H, W = pred.shape
    score = cv2.resize(res["debug"]["score"], (W, H)).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kecerahan = float(gray[leaf].mean()) if leaf.any() else float(gray.mean())
    a, b = score[leaf & gtb], score[leaf & ~gtb]
    if a.size and b.size:
        sep = float(b.mean() - a.mean()) if DEFAULT_PARAMS["feature"] == "hue_abs" \
            else float(a.mean() - b.mean())
    else:
        sep = NAN
 
    tahap = guess_stage(gt_area, pred_area, gt_in_leaf, m, rec_raw)
    sebab, usul = explain(row["kondisi"], m, gt_area, pred_area, gt_in_leaf, rec_raw, sep, kecerahan, tahap)
    out = {
        "file": row["file"], "kondisi": row["kondisi"],
        **{k: round(v, 4) for k, v in m.items()},
        "area_pred": pred_area, "area_gt": gt_area,
        "gt_di_dalam_daun": round(gt_in_leaf, 3) if gt_in_leaf == gt_in_leaf else "",
        "iou_sebelum_morfologi": round(metrics.iou(raw, gtb), 4),
        "kecerahan_daun": round(kecerahan, 1),
        "pemisahan_skor": round(sep, 1) if sep == sep else "",
        "tahap_dugaan": tahap,
        "penyebab": sebab, "usulan_perbaikan": usul,
    }
    return out, pred, gtb
 
 
def render(img, res, pred, gtb, title):
    H, W = img.shape[:2]
    dbg = res["debug"]
    if DEFAULT_PARAMS["feature"] == "hue_abs":
        score_title = f"4. Kanal hue (cacat bila hue < {dbg['threshold']:.0f})"
        score_kw = {"cmap": "magma", "vmin": 0, "vmax": 180}
    else:
        score_title = f"4. Skor cacat (ambang={dbg['threshold']:.0f})"
        score_kw = {"cmap": "magma"}
    panels = [
        ("1. Asli", _rgb(img), {}),
        ("2. Setelah blur + CLAHE", _rgb(cv2.resize(dbg["enhanced"], (W, H))), {}),
        ("3. Mask daun", res["leaf_mask"], {"cmap": "gray"}),
        (score_title, cv2.resize(dbg["score"], (W, H)), score_kw),
        ("5. Cacat sebelum morfologi", res["debug"]["raw_defect"], {"cmap": "gray"}),
        ("6. Prediksi akhir", res["mask"], {"cmap": "gray"}),
        ("7. Ground truth", _rgb(overlay(img, gtb.astype(np.uint8) * 255, (255, 0, 255))), {}),
        ("8. Galat: hijau=benar, merah=salah, biru=terlewat", _rgb(error_map(img, pred, gtb)), {}),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for ax, (name, im, kw) in zip(axes.ravel(), panels):
        ax.imshow(im, **kw)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
 
 
def main():
    rows = evaluate_split("test")          # SELALU test
    assert rows and all(r["split"] == "test" for r in rows), "data test kosong / salah split"
    worst = sorted(rows, key=lambda r: r["iou"])[:N_WORST]
 
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = []
    for rank, r in enumerate(worst, 1):
        name = r["file"]
        img = cv2.imread(str(ROOT / "data" / "images" / name))
        gt = load_gt(ROOT / "data" / "masks", name)
        res = segment_leaf_defect(img)     # parameter = DEFAULT_PARAMS (harus sudah final)
        rec, pred, gtb = analyse(r, img, gt, res)
        rec["urutan"] = rank
        table.append(rec)
        title = f"#{rank} {name} | kondisi {r['kondisi']} | IoU={rec['iou']:.3f} | {rec['tahap_dugaan']}"
        fig = render(img, res, pred, gtb, title)
        out = OUT_DIR / f"failure_{rank}_{Path(name).stem}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"#{rank} {name} (kondisi {r['kondisi']})  IoU={rec['iou']:.3f}  "
              f"P={rec['precision']:.3f}  R={rec['recall']:.3f}")
        print(f"    dugaan tahap : {rec['tahap_dugaan']}")
        print(f"    penyebab     : {rec['penyebab']}")
        print(f"    usulan       : {rec['usulan_perbaikan']}")
        print(f"    gambar       : {out.relative_to(ROOT)}")
 
    cols = ["urutan", "file", "kondisi", "iou", "dice", "precision", "recall",
            "area_pred", "area_gt", "gt_di_dalam_daun", "iou_sebelum_morfologi",
            "kecerahan_daun", "pemisahan_skor", "tahap_dugaan", "penyebab", "usulan_perbaikan"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    print("\nTersimpan:", OUT_CSV.relative_to(ROOT))
    print("Catatan: 'penyebab' dan 'usulan_perbaikan' adalah draf dari angka. Cocokkan dengan "
          "gambar di results/failure/ dan sunting CSV bila perlu.")
 
 
if __name__ == "__main__":
    main()
 
 
if __name__ == "__main__":
    main()
