import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scratch_ops import otsu_threshold, apply_threshold, erode, dilate  


def waktu_ms(fungsi, *args, ulang=5):
    """Median waktu eksekusi (ms) dari beberapa kali percobaan."""
    t = []
    for _ in range(ulang):
        t0 = time.perf_counter()
        hasil = fungsi(*args)
        t.append((time.perf_counter() - t0) * 1000)
    return hasil, float(np.median(t))


def selisih(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return int(d.max()), float(d.mean()), int(np.count_nonzero(d))


def pilih_citra(meta, n):
    """Ambil n citra dev yang tersebar merata di kondisi A/B/C."""
    dev = meta[meta["split"] == "dev"].sort_values("file")
    kondisi = sorted(dev["kondisi"].unique())
    pilihan, idx = [], 0
    while len(pilihan) < n:
        for k in kondisi:
            daftar = dev[dev["kondisi"] == k]["file"].tolist()
            if idx < len(daftar) and len(pilihan) < n:
                pilihan.append(daftar[idx])
        idx += 1
        if idx > len(dev):
            break
    return pilihan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="ukuran elemen penstruktur (ganjil)")
    ap.add_argument("--n", type=int, default=10, help="jumlah citra uji verifikasi (min. 10)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    meta = pd.read_csv(os.path.join(a.data, "metadata.csv"))
    files = pilih_citra(meta, a.n)
    kernel = np.ones((a.k, a.k), np.uint8)
    os.makedirs(a.out, exist_ok=True)

    rows_otsu, rows_morph = [], []
    for f in files:
        img = cv2.imread(os.path.join(a.data, "images", f))
        S = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
        kond = meta.loc[meta["file"] == f, "kondisi"].iloc[0]

        T_own, t_own = waktu_ms(otsu_threshold, S)
        (T_cv, b_cv), t_cv = waktu_ms(
            cv2.threshold, S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        b_own = apply_threshold(S, T_own)
        mx, mean, n = selisih(b_own, b_cv)
        rows_otsu.append(dict(citra=f, kondisi=kond, T_sendiri=T_own, T_cv2=int(T_cv),
                              selisih_maks=mx, selisih_rata=mean, piksel_beda=n,
                              waktu_sendiri_ms=t_own, waktu_cv2_ms=t_cv))

        for nama, own_fn, cv_fn in [("erosi", erode, cv2.erode), ("dilasi", dilate, cv2.dilate)]:
            r_own, t_own = waktu_ms(own_fn, b_cv, a.k)
            r_cv, t_cv = waktu_ms(cv_fn, b_cv, kernel)
            mx, mean, n = selisih(r_own, r_cv)
            rows_morph.append(dict(citra=f, kondisi=kond, operasi=nama, k=a.k,
                                   selisih_maks=mx, selisih_rata=mean, piksel_beda=n,
                                   waktu_sendiri_ms=t_own, waktu_cv2_ms=t_cv))

    df_o, df_m = pd.DataFrame(rows_otsu), pd.DataFrame(rows_morph)
    df_o.to_csv(os.path.join(a.out, "verify_otsu.csv"), index=False)
    df_m.to_csv(os.path.join(a.out, "verify_morph.csv"), index=False)

    pd.set_option("display.width", 160)
    print(f"\n=== Verifikasi Otsu ({len(files)} citra, kanal Saturation) ===")
    print(df_o.round(3).to_string(index=False))
    print("Rata-rata:", df_o[["selisih_maks", "piksel_beda", "waktu_sendiri_ms", "waktu_cv2_ms"]]
          .mean().round(3).to_dict())

    print(f"\n=== Verifikasi erosi/dilasi (k={a.k}) ===")
    print(df_m.round(3).to_string(index=False))
    print("\nRata-rata per operasi:")
    print(df_m.groupby("operasi")[["selisih_maks", "piksel_beda", "waktu_sendiri_ms", "waktu_cv2_ms"]]
          .mean().round(3).to_string())
    print(f"\nTersimpan: {a.out}/verify_otsu.csv, {a.out}/verify_morph.csv")


if __name__ == "__main__":
    main()