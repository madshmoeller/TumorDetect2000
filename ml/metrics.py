"""Segmentation metrics, shared by baselines, the deep model and figures.

Everything that scores a prediction against ground truth goes through this
module, so a baseline's Dice and the deep model's Dice are computed by
identical code — no room for a scoring discrepancy to sneak the comparison in
figure F09.
"""

from __future__ import annotations

import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Both boolean, same shape. Dice = 1.0 when both are empty (agreement)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2 * inter / denom)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Recall: fraction of true positive voxels the prediction found."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if gt.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return float(np.logical_and(pred, gt).sum() / gt.sum())


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return float(np.logical_and(pred, gt).sum() / pred.sum())


def volume_mm3(mask: np.ndarray, voxel_volume: float = 1.0) -> float:
    return float(mask.astype(bool).sum() * voxel_volume)


def hausdorff95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile symmetric surface distance, in voxels (1mm iso -> mm).

    Computed via distance transforms rather than an explicit surface-point
    search: `distance_transform_edt` on the complement gives, at every voxel,
    the distance to the nearest foreground voxel. Sampling that at the *other*
    mask's surface voxels gives the one-directional distance set in O(N) after
    two transforms, instead of an O(N*M) point-to-point search.

    Undefined (returns nan) when either mask is empty — HD95 has no meaning
    against nothing, and nan is what lets figures skip these cases instead of
    silently treating them as zero error.
    """
    from scipy import ndimage

    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")

    def surface(mask):
        er = ndimage.binary_erosion(mask, border_value=0)
        return mask & ~er

    pred_s, gt_s = surface(pred), surface(gt)
    if not pred_s.any() or not gt_s.any():  # a single-voxel mask has no interior to erode away
        pred_s, gt_s = pred, gt

    dt_gt = ndimage.distance_transform_edt(~gt)
    dt_pred = ndimage.distance_transform_edt(~pred)

    d_pred_to_gt = dt_gt[pred_s]
    d_gt_to_pred = dt_pred[gt_s]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95))


def case_metrics(pred: np.ndarray, gt: np.ndarray, *, surface: bool = True) -> dict:
    """The standard bundle computed for every case, every model, every baseline."""
    out = {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "sensitivity": sensitivity(pred, gt),
        "precision": precision(pred, gt),
        "pred_volume_mm3": volume_mm3(pred),
        "true_volume_mm3": volume_mm3(gt),
    }
    if surface:
        out["hd95"] = hausdorff95(pred, gt)
    return out


# ── cohort-level summaries ───────────────────────────────────────────────────


def bootstrap_ci(values: np.ndarray, *, n: int = 10_000, seed: int = 7, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. n=60 is small; report the interval, not just the point."""
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=float)
    idx = rng.randint(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired Wilcoxon signed-rank test. Returns the p-value.

    Used to test whether the deep model beats a baseline on the *same* cases —
    paired because both scores exist for every one of the 60 cases.
    """
    from scipy import stats

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    if np.all(diff == 0):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="greater").pvalue)
