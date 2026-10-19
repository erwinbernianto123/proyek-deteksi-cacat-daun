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

    return (gray > T).astype(np.uint8) * 255


def erode(binary, k=3):
    r = k // 2
    H, W = binary.shape
    pad = np.pad(binary, r, mode="constant", constant_values=255)
    out = np.full((H, W), 255, dtype=binary.dtype)
    for dy in range(k):                 # geser jendela: k*k posisi tetangga
        for dx in range(k):
            out = np.minimum(out, pad[dy:dy + H, dx:dx + W])
    return out


def dilate(binary, k=3):
    r = k // 2
    H, W = binary.shape
    pad = np.pad(binary, r, mode="constant", constant_values=0)
    out = np.zeros((H, W), dtype=binary.dtype)
    for dy in range(k):
        for dx in range(k):
            out = np.maximum(out, pad[dy:dy + H, dx:dx + W])
    return out