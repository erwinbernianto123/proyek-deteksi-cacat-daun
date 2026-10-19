"""Operasi dari nol, hanya memakai NumPy. Ditulis oleh: <Muhammad Rasmi Baihaqi>."""
import numpy as np


def otsu_threshold(gray):
    """gray: array 2D uint8 (0-255).
    Mengembalikan ambang T (int). Piksel > T = objek (sama seperti cv2)."""
    raise NotImplementedError  # TODO: tulis sendiri


def apply_threshold(gray, T):
    """Mengembalikan citra biner uint8: 255 jika gray > T, selain itu 0."""
    raise NotImplementedError


def erode(binary, k=3):
    raise NotImplementedError  # dikerjakan setelah Otsu


def dilate(binary, k=3):
    raise NotImplementedError