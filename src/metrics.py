import numpy as np

def _to_boolean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Fungsi pembantu internal untuk mengubah mask input
    (baik bernilai 0/255 maupun boolean) menjadi array boolean.
    """
    if mask.dtype == bool:
        return mask
    return mask > 0

def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Menghitung Intersection over Union (IoU).
    """
    pred_bool = _to_boolean_mask(pred_mask)
    gt_bool = _to_boolean_mask(gt_mask)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    # Aturan mask kosong: jika kedua mask sama-sama kosong (sehat), nilai 1.0
    if union == 0:
        return 1.0

    return float(intersection / union)


def dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Menghitung Dice Coefficient.
    """
    pred_bool = _to_boolean_mask(pred_mask)
    gt_bool = _to_boolean_mask(gt_mask)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    total = pred_bool.sum() + gt_bool.sum()

    # Aturan mask kosong: jika total piksel kosong semua, nilai 1.0
    if total == 0:
        return 1.0

    return float((2.0 * intersection) / total)


def precision(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Menghitung Precision.
    """
    pred_bool = _to_boolean_mask(pred_mask)
    gt_bool = _to_boolean_mask(gt_mask)

    tp = np.logical_and(pred_bool, gt_bool).sum()
    tp_fp = pred_bool.sum()

    if tp_fp == 0:
        return 1.0 if gt_bool.sum() == 0 else 0.0

    return float(tp / tp_fp)


def recall(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Menghitung Recall.
    """
    pred_bool = _to_boolean_mask(pred_mask)
    gt_bool = _to_boolean_mask(gt_mask)

    tp = np.logical_and(pred_bool, gt_bool).sum()
    tp_fn = gt_bool.sum()

    if tp_fn == 0:
        return 1.0 if pred_bool.sum() == 0 else 0.0

    return float(tp / tp_fn)


def calculate_all_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Fungsi untuk menghitung seluruh metrik sekaligus dalam bentuk dictionary.
    """
    return {
        "iou": iou(pred_mask, gt_mask),
        "dice": dice(pred_mask, gt_mask),
        "precision": precision(pred_mask, gt_mask),
        "recall": recall(pred_mask, gt_mask)
    }
