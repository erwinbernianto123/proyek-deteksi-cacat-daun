import numpy as np


def otsu_threshold(gray):
    hist = np.bincount(gray.ravel(), minlength=256)
    p = hist / gray.size
    i = np.arange(256)

    skor_terbaik = -1
    T_terbaik = 0
    for T in range(256):
        w0 = p[:T+1].sum()                       
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            continue                            
        mu0 = (i[:T+1] * p[:T+1]).sum() / w0     
        mu1 = (i[T+1:] * p[T+1:]).sum() / w1     
        skor = w0 * w1 * (mu0 - mu1) ** 2        
        if skor > skor_terbaik:                  
            skor_terbaik = skor
            T_terbaik = T
    return int(T_terbaik)


def apply_threshold(gray, T):
    """Mengembalikan citra biner uint8: 255 jika gray > T, selain itu 0."""
    return (gray > T).astype(np.uint8) * 255


def _elemen_penstruktur(k):
    """Ubah k menjadi elemen penstruktur boolean.
    - k angka (mis. 3)      -> persegi k x k berisi True semua
    - k array 0/1 (mis. elips dari pipeline) -> dipakai apa adanya"""
    if np.isscalar(k):
        return np.ones((int(k), int(k)), dtype=bool)
    return np.asarray(k) > 0


def erode(binary, k=3):
    se = _elemen_penstruktur(k)
    kh, kw = se.shape
    ry, rx = kh // 2, kw // 2                     
    H, W = binary.shape
    pad = np.pad(binary, ((ry, ry), (rx, rx)), mode="constant", constant_values=255)
    out = np.full((H, W), 255, dtype=binary.dtype)
    for dy in range(kh):                           
        for dx in range(kw):
            if se[dy, dx]:                        
                out = np.minimum(out, pad[dy:dy + H, dx:dx + W])
    return out


def dilate(binary, k=3):
    se = _elemen_penstruktur(k)
    kh, kw = se.shape
    ry, rx = kh // 2, kw // 2
    H, W = binary.shape
    pad = np.pad(binary, ((ry, ry), (rx, rx)), mode="constant", constant_values=0)
    out = np.zeros((H, W), dtype=binary.dtype)
    for dy in range(kh):
        for dx in range(kw):
            if se[dy, dx]:
                out = np.maximum(out, pad[dy:dy + H, dx:dx + W])
    return out