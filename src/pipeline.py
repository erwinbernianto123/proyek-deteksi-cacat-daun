"""
pipeline.py  --  Pipeline deteksi cacat permukaan daun (teknik klasik).

Tahap pipeline (min. 4 tahap sesuai aturan tugas):
  1. Pra-proses      : resize (opsional) + Gaussian / median blur
  2. Perbaikan kontras: CLAHE pada kanal L (ruang warna Lab)
  3. Segmentasi      : (a) mask daun = Otsu pada Saturation + hole filling
                       (b) mask cacat = threshold pada fitur a* (atau hue)
                           di dalam area daun
  4. Morfologi       : opening + closing + buang komponen kecil
  5. Pengukuran      : bounding box dan luas (piksel) cacat

Antarmuka utama (kontrak untuk seluruh tim):

    hasil = segment_leaf_defect(image_bgr, params=None)

    hasil["mask"]       : np.uint8 (H, W), nilai 0 / 255  (255 = cacat)
    hasil["leaf_mask"]  : np.uint8 (H, W), nilai 0 / 255  (255 = daun)
    hasil["bbox"]       : (x, y, w, h) kotak yang membungkus semua cacat, atau None
    hasil["boxes"]      : list (x, y, w, h) per komponen cacat
    hasil["area"]       : jumlah piksel cacat
    hasil["area_ratio"] : area cacat / area daun (0..1)
    hasil["debug"]      : dict citra antara (untuk debug_view.py)

Parameter bisa diganti lewat dict `params` (lihat DEFAULT_PARAMS), jadi
experiments.py (Gaussian vs median, a* vs hue, offset Otsu) tidak perlu
mengubah isi file ini:  segment_leaf_defect(img, {"blur": "median"}).
"""

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Parameter default.  Nilai ini TIDAK boleh dituning memakai data test.
# Tuning hanya pada split "dev" (lihat experiments.py).
# --------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # tahap 1: pra-proses
    "max_side": 1024,        # citra lebih besar di-resize (mask dikembalikan ke ukuran asli)
    "blur": "gaussian",      # "gaussian" | "median"
    "blur_ksize": 5,         # harus ganjil

    # tahap 2: kontras
    "use_clahe": True,
    "clahe_clip": 2.0,
    "clahe_grid": 8,

    # tahap 3a: mask daun
    "leaf_offset": 0,        # geser ambang Otsu saturation
    "leaf_min_area_frac": 0.005,  # komponen daun minimal (fraksi luas citra)
    "leaf_min_score": 1000.0,  # skor komponen minimal sebelum memakai kandidat cadangan
    "leaf_use_hull": False,  # perluas mask daun ke convex hull (menangkap cacat di tepi)
    "leaf_border_erode": 5,  # kikis tepi daun agar tepi/bayangan tidak dianggap cacat

    # tahap 3b: mask cacat
    "feature": "a_star",     # "a_star" | "hue"
    "otsu_offset": 0,        # geser ambang Otsu fitur cacat
    "min_delta_a": 8,        # a* minimal di atas median daun supaya dianggap cacat
    "min_delta_hue": 14,     # selisih hue minimal (skala OpenCV 0..180) untuk "hue"

    # tahap 4: morfologi
    "open_ksize": 3,
    "close_ksize": 5,
    "min_defect_area": 30,   # komponen cacat lebih kecil dari ini (piksel) dibuang

    # operasi dasar: "opencv" atau "scratch" (src/scratch_ops.py buatan Rasmi)
    "ops": "opencv",
}


# --------------------------------------------------------------------------
# Operasi dasar yang bisa ditukar antara OpenCV dan versi scratch
# --------------------------------------------------------------------------
def _scratch():
    try:
        import scratch_ops  # dimuat hanya jika ops="scratch"
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "ops='scratch' butuh src/scratch_ops.py (milik Rasmi) dengan fungsi "
            "otsu_threshold(gray), erode(mask, kernel), dilate(mask, kernel). "
            "Pakai --ops opencv sampai file itu ada.") from e
    return scratch_ops


def _otsu(gray_u8, ops="opencv"):
    """Return ambang Otsu (int) dari array uint8 (boleh 1D / 2D)."""
    if ops == "scratch":
        t = _scratch().otsu_threshold(gray_u8)
        if isinstance(t, tuple):      # kalau fungsi Rasmi mengembalikan (t, mask)
            t = t[0]
        return float(t)
    data = np.ascontiguousarray(gray_u8.reshape(-1, 1))
    t, _ = cv2.threshold(data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def _kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _erode(mask, k, ops="opencv"):
    if k <= 1:
        return mask
    if ops == "scratch":
        return _scratch().erode(mask, _kernel(k))
    return cv2.erode(mask, _kernel(k))


def _dilate(mask, k, ops="opencv"):
    if k <= 1:
        return mask
    if ops == "scratch":
        return _scratch().dilate(mask, _kernel(k))
    return cv2.dilate(mask, _kernel(k))


def _open(mask, k, ops):
    return _dilate(_erode(mask, k, ops), k, ops)


def _close(mask, k, ops):
    return _erode(_dilate(mask, k, ops), k, ops)


# --------------------------------------------------------------------------
# Tahap 1 & 2: pra-proses dan perbaikan kontras
# --------------------------------------------------------------------------
def preprocess(img_bgr, p):
    k = int(p["blur_ksize"]) | 1  # paksa ganjil
    if p["blur"] == "median":
        return cv2.medianBlur(img_bgr, k)
    if p["blur"] == "gaussian":
        return cv2.GaussianBlur(img_bgr, (k, k), 0)
    raise ValueError(f"blur tidak dikenal: {p['blur']}")


def enhance_contrast(img_bgr, p):
    """CLAHE hanya pada kanal L, warna (a*, b*) tidak diubah."""
    if not p["use_clahe"]:
        return img_bgr
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"],
                            tileGridSize=(p["clahe_grid"], p["clahe_grid"]))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------
# Tahap 3a: mask daun
# --------------------------------------------------------------------------
def fill_holes(mask):
    """Isi lubang di dalam objek: flood fill dari pojok, lalu invers."""
    h, w = mask.shape
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    # cari titik latar di tepi citra sebagai seed
    seed = None
    for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if mask[y, x] == 0:
            seed = (x, y)
            break
    if seed is None:  # semua pojok objek -> tidak ada yang perlu diisi dari luar
        return mask
    cv2.floodFill(flood, ff_mask, seed, 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def _best_component(mask, a_star, grad, min_area_frac, max_area_frac=0.45):
    """
    Pilih komponen terhubung yang paling mirip daun. Skor menggabungkan:
    luas, kepadatan bentuk (solidity), dekat pusat citra, tidak menempel
    tepi citra, dan porsi piksel hijau (a* < 128).
    Return (mask_komponen, skor).
    """
    h, w = mask.shape
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_i, best_score = 0, 0.0
    diag = np.hypot(w, h)
    for i in range(1, n):
        x, y, bw, bh, area = (int(stats[i, c]) for c in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
            cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
        if area < min_area_frac * h * w or area > max_area_frac * h * w:
            continue
        comp = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_area = max(cv2.contourArea(cv2.convexHull(np.vstack(cnts))), 1.0)
        solidity = min(area / hull_area, 1.0)
        dist = np.hypot(cents[i][0] - w / 2.0, cents[i][1] - h / 2.0) / diag
        touches = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        greenfrac = float(np.mean(a_star[lab == i] < 128))
        sub = comp > 0
        ring = sub & ~(cv2.erode(comp, np.ones((3, 3), np.uint8)) > 0)
        edge = float(grad[y:y + bh, x:x + bw][ring].mean()) if ring.any() else 0.0
        score = (np.sqrt(area) * solidity ** 2 * (1.0 - dist) * (edge + 1.0)
                 * (0.15 if touches else 1.0) * (greenfrac + 0.25))
        if score > best_score:
            best_i, best_score = i, score
    if best_i == 0:
        return np.zeros_like(mask), 0.0
    return np.where(lab == best_i, 255, 0).astype(np.uint8), best_score


def segment_leaf(img_bgr, p):
    """
    Mask daun. Kandidat 1: Otsu pada Saturation (daun lebih jenuh dari latar).
    Kandidat 2: Otsu pada V terbalik (daun lebih gelap dari latar), berguna
    pada kondisi C (sangat gelap) saat saturasi daun ~ saturasi latar.
    Komponen dengan skor terbaik dari kedua kandidat dipilih, lalu hole filling.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    a_star = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 1]
    ops = p["ops"]
    L = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0], (5, 5), 0)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    sat = hsv[:, :, 1]
    t_s = _otsu(sat, ops) + p["leaf_offset"]
    cand_sat = np.where(sat > t_s, 255, 0).astype(np.uint8)

    dark = 255 - hsv[:, :, 2]
    t_d = _otsu(dark, ops) + p["leaf_offset"]
    cand_dark = np.where(dark > t_d, 255, 0).astype(np.uint8)

    green = 255 - a_star  # makin hijau makin besar
    t_g = _otsu(green, ops) + p["leaf_offset"]
    cand_green = np.where(green > t_g, 255, 0).astype(np.uint8)

    cand_sat_inv = 255 - cand_sat  # daun lebih pucat dari latar (mis. latar kebiruan)

    # jarak warna dari latar: warna latar = median piksel di pinggir citra
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = sat.shape
    b = max(4, int(0.04 * min(h, w)))
    ring = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                           lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    diff = (lab - bg) * np.array([0.5, 1.0, 1.0], np.float32)  # bobot L kecil (bayangan)
    dist = np.clip(np.linalg.norm(diff, axis=2) * 6.0, 0, 255).astype(np.uint8)
    t_b = _otsu(dist, ops) + p["leaf_offset"]
    cand_bg = np.where(dist > t_b, 255, 0).astype(np.uint8)

    def pick(cands):
        best, best_score = np.zeros_like(cand_sat), 0.0
        for cand in cands:
            m = _open(cand, 5, ops)
            m = _close(m, 7, ops)
            m = fill_holes(m)
            comp, sc = _best_component(m, a_star, grad, p["leaf_min_area_frac"])
            if sc > best_score:
                best, best_score = comp, sc
        return best, best_score

    # kandidat utama dulu; jarak-warna-latar hanya cadangan bila skor utama rendah
    best, sc = pick((cand_sat, cand_sat_inv, cand_dark, cand_green))
    if sc < p["leaf_min_score"]:
        alt, sc2 = pick((cand_bg,))
        if sc2 > sc:
            best = alt
    best = fill_holes(best)
    if p["leaf_use_hull"] and best.any():
        # cacat kuning/cokelat di tepi tidak hijau -> ikut terpotong; hull menutup kembali
        cnts, _ = cv2.findContours(best, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack(cnts))
        best = np.zeros_like(best)
        cv2.fillPoly(best, [hull], 255)
    return best


# --------------------------------------------------------------------------
# Tahap 3b: mask cacat di dalam daun
# --------------------------------------------------------------------------
def defect_score(img_bgr, leaf_mask, p):
    """
    Skor cacat uint8 per piksel (makin besar makin mungkin cacat) dan
    batas bawah ambang (floor) supaya daun sehat tidak dipecah oleh Otsu.
    """
    inside = leaf_mask > 0
    if p["feature"] == "a_star":
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        a = lab[:, :, 1].astype(np.float32)           # hijau < 128 < merah
        med = float(np.median(a[inside])) if inside.any() else 128.0
        score = np.clip(a - med + 128.0, 0, 255).astype(np.uint8)
        floor = 128.0 + p["min_delta_a"]
    elif p["feature"] == "hue":
        hue = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 0].astype(np.float32)
        med = float(np.median(hue[inside])) if inside.any() else 60.0
        d = np.abs(hue - med)
        d = np.minimum(d, 180.0 - d)                  # hue melingkar
        score = np.clip(d * 2.0, 0, 255).astype(np.uint8)
        floor = 2.0 * p["min_delta_hue"]
    else:
        raise ValueError(f"feature tidak dikenal: {p['feature']}")
    return score, floor


def segment_defect(img_bgr, leaf_mask, p):
    ops = p["ops"]
    score, floor = defect_score(img_bgr, leaf_mask, p)

    # area pencarian: daun dikikis sedikit agar tepi daun tidak ikut
    search = _erode(leaf_mask, int(p["leaf_border_erode"]) * 2 + 1, ops)
    if search.sum() == 0:
        search = leaf_mask

    vals = score[search > 0]
    if vals.size == 0:
        return np.zeros_like(leaf_mask), score, 255.0
    t_otsu = _otsu(vals, ops)
    thr = max(t_otsu + p["otsu_offset"], floor)
    mask = np.where((score > thr) & (search > 0), 255, 0).astype(np.uint8)
    return mask, score, thr


# --------------------------------------------------------------------------
# Tahap 4: morfologi + buang komponen kecil
# --------------------------------------------------------------------------
def clean_mask(mask, p):
    ops = p["ops"]
    m = _open(mask, int(p["open_ksize"]), ops)
    m = _close(m, int(p["close_ksize"]), ops)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= p["min_defect_area"]:
            out[lab == i] = 255
    return out


# --------------------------------------------------------------------------
# Tahap 5: pengukuran
# --------------------------------------------------------------------------
def measure(defect_mask, leaf_mask):
    area = int(np.count_nonzero(defect_mask))
    leaf_area = int(np.count_nonzero(leaf_mask))
    boxes = []
    n, _, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h = (int(stats[i, c]) for c in
                      (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        boxes.append((x, y, w, h))
    bbox = None
    if area > 0:
        ys, xs = np.nonzero(defect_mask)
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return {
        "bbox": bbox,
        "boxes": boxes,
        "area": area,
        "leaf_area": leaf_area,
        "area_ratio": (area / leaf_area) if leaf_area else 0.0,
    }


# --------------------------------------------------------------------------
# Fungsi utama
# --------------------------------------------------------------------------
def segment_leaf_defect(image_bgr, params=None):
    """Jalankan seluruh pipeline pada satu citra BGR. Lihat docstring modul."""
    p = dict(DEFAULT_PARAMS)
    if params:
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise KeyError(f"parameter tidak dikenal: {sorted(unknown)}")
        p.update(params)

    H0, W0 = image_bgr.shape[:2]
    scale = min(1.0, p["max_side"] / max(H0, W0))
    img = image_bgr
    if scale < 1.0:
        img = cv2.resize(image_bgr, (int(W0 * scale), int(H0 * scale)),
                         interpolation=cv2.INTER_AREA)

    blurred = preprocess(img, p)                    # tahap 1
    enhanced = enhance_contrast(blurred, p)         # tahap 2
    leaf = segment_leaf(enhanced, p)                # tahap 3a
    raw, score, thr = segment_defect(enhanced, leaf, p)  # tahap 3b
    defect = clean_mask(raw, p)                     # tahap 4

    if scale < 1.0:  # kembalikan ke ukuran asli (nearest agar tetap biner)
        defect = cv2.resize(defect, (W0, H0), interpolation=cv2.INTER_NEAREST)
        leaf = cv2.resize(leaf, (W0, H0), interpolation=cv2.INTER_NEAREST)
        raw = cv2.resize(raw, (W0, H0), interpolation=cv2.INTER_NEAREST)

    info = measure(defect, leaf)                    # tahap 5
    info.update({
        "mask": defect,
        "leaf_mask": leaf,
        "debug": {
            "blurred": blurred, "enhanced": enhanced, "score": score,
            "threshold": thr, "raw_defect": raw,
        },
    })
    return info
