"""Baselines: FLAIR threshold and per-voxel random forest.

Rule 3: decide what "good" is relative to *before* the deep model runs. Both
baselines are scored here, on all 60 cases, on the same 5-fold split
(`config.fold_assignment`) that the deep models will later use — so the
comparison in figure F09 is apples-to-apples and not "baseline in-sample vs
model out-of-fold".

    python -m ml.baselines
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from scipy import ndimage

from . import config as C
from . import data as D
from . import metrics as M

# ── baseline 1: FLAIR threshold ─────────────────────────────────────────────
#
# No training at all — a fixed rule applied identically to every case. This is
# the "does deep learning earn its keep" control: threshold the normalised
# FLAIR channel, keep the largest connected components.


def flair_threshold_predict(images: np.ndarray, brain: np.ndarray, *, z: float = 1.25) -> np.ndarray:
    """images: (4, Z, Y, X) normalised. Returns a binary mask."""
    flair = images[0]
    raw = (flair > z) & brain
    return _keep_largest_components(raw)


def _keep_largest_components(mask: np.ndarray, max_components: int = 3) -> np.ndarray:
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1][:max_components]
    keep = np.zeros_like(mask)
    for idx in order:
        if sizes[idx] >= C.MIN_COMPONENT_VOXELS:
            keep |= labeled == (idx + 1)
    return keep


def tune_flair_threshold(train_ids: list[str], z_grid: np.ndarray) -> float:
    """Pick the z-threshold that maximises mean Dice on `train_ids`.

    Tuned per fold on the training split only — the held-out cases never
    influence the threshold, which is what makes the held-out score honest.
    """
    best_z, best_dice = float(z_grid[0]), -1.0
    for z in z_grid:
        dices = []
        for cid in train_ids:
            images, labels, brain = D.load_cached(cid, mmap=False)
            pred = flair_threshold_predict(images.astype(np.float32), brain, z=z)
            dices.append(M.dice(pred, labels > 0))
        mean_dice = float(np.mean(dices))
        if mean_dice > best_dice:
            best_z, best_dice = float(z), mean_dice
    return best_z


# ── baseline 2: per-voxel random forest ─────────────────────────────────────
#
# Stronger than it sounds: 4 intensities + multi-scale smoothed features per
# voxel, subsampled for training (a full brain is ~1.4M voxels/case; we do not
# need all of them to fit a forest). A far more honest "is the U-Net's shape
# prior actually earning anything" control than the threshold alone.

SIGMAS = (1.0, 2.0, 4.0)


def _voxel_features(images: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """images: (4, Z, Y, X) normalised. Returns (N_voxels_in_brain, n_features)."""
    feats = [images[c][brain] for c in range(images.shape[0])]
    for c in range(images.shape[0]):
        for sigma in SIGMAS:
            smoothed = ndimage.gaussian_filter(images[c], sigma=sigma)
            feats.append(smoothed[brain])
    return np.stack(feats, axis=1).astype(np.float32)


def train_random_forest(train_ids: list[str], *, voxels_per_case: int = 4000, seed: int = 0):
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(seed)
    X_parts, y_parts = [], []
    for cid in train_ids:
        images, labels, brain = D.load_cached(cid, mmap=False)
        images = images.astype(np.float32)
        feats = _voxel_features(images, brain)
        y = (labels[brain] > 0).astype(np.uint8)

        n = len(y)
        take = min(voxels_per_case, n)
        # Half the sample is tumour voxels if available, half background —
        # tumour is ~7.5% of brain, so uniform sampling would starve the
        # positive class the forest most needs to see.
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_pos = min(take // 2, len(pos_idx))
        n_neg = take - n_pos
        sel = np.concatenate(
            [
                rng.choice(pos_idx, n_pos, replace=False) if n_pos else np.array([], dtype=int),
                rng.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False),
            ]
        )
        X_parts.append(feats[sel])
        y_parts.append(y[sel])

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=14, n_jobs=-1, random_state=seed, class_weight="balanced"
    )
    clf.fit(X, y)
    return clf


def random_forest_predict(clf, images: np.ndarray, brain: np.ndarray) -> np.ndarray:
    feats = _voxel_features(images, brain)
    prob = clf.predict_proba(feats)[:, 1]
    out = np.zeros(brain.shape, dtype=np.float32)
    out[brain] = prob
    return _keep_largest_components(out > 0.5)


# ── driver ───────────────────────────────────────────────────────────────────


def run(ids: list[str], *, verbose: bool = True) -> dict:
    folds = C.fold_assignment(ids)
    z_grid = np.linspace(0.5, 2.5, 9)

    results = {"flair_threshold": {}, "random_forest": {}}
    per_fold_threshold = {}

    for f in range(C.N_FOLDS):
        train_ids = [c for c in ids if folds[c] != f]
        val_ids = [c for c in ids if folds[c] == f]
        if verbose:
            print(f"\nfold {f}: {len(train_ids)} train, {len(val_ids)} val")

        t0 = time.time()
        z = tune_flair_threshold(train_ids, z_grid)
        per_fold_threshold[f] = z
        if verbose:
            print(f"  FLAIR threshold tuned: z={z:.2f}  ({time.time() - t0:.0f}s)")

        t0 = time.time()
        clf = train_random_forest(train_ids)
        if verbose:
            print(f"  random forest trained  ({time.time() - t0:.0f}s)")

        for cid in val_ids:
            images, labels, brain = D.load_cached(cid, mmap=False)
            images = images.astype(np.float32)
            gt = labels > 0

            pred_t = flair_threshold_predict(images, brain, z=z)
            pred_rf = random_forest_predict(clf, images, brain)

            results["flair_threshold"][cid] = M.case_metrics(pred_t, gt)
            results["random_forest"][cid] = M.case_metrics(pred_rf, gt)
            if verbose:
                print(
                    f"  {cid}  threshold dice {results['flair_threshold'][cid]['dice']:.3f}   "
                    f"rf dice {results['random_forest'][cid]['dice']:.3f}"
                )

    results["fold_assignment"] = folds
    results["flair_threshold_per_fold"] = per_fold_threshold
    return results


def summarise(results: dict) -> dict:
    out = {}
    for name in ("flair_threshold", "random_forest"):
        dices = np.array([v["dice"] for v in results[name].values()])
        out[name] = {
            "mean_dice": float(dices.mean()),
            "median_dice": float(np.median(dices)),
            "min_dice": float(dices.min()),
            "max_dice": float(dices.max()),
            "n": len(dices),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="standard", choices=["standard", "tiny"])
    args = ap.parse_args(argv)

    ids = C.case_ids(args.tier)
    t0 = time.time()
    results = run(ids)
    summary = summarise(results)

    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    (C.OUTPUTS / "baselines.json").write_text(json.dumps(results, indent=2))
    (C.OUTPUTS / "baselines_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'model':<18} {'mean':>7} {'median':>7} {'min':>7} {'max':>7}   n")
    for name, s in summary.items():
        print(
            f"{name:<18} {s['mean_dice']:>7.3f} {s['median_dice']:>7.3f} "
            f"{s['min_dice']:>7.3f} {s['max_dice']:>7.3f}   {s['n']}"
        )
    print(f"\ndone in {time.time() - t0:.0f}s")
    print(f"wrote {C.OUTPUTS / 'baselines.json'} and baselines_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
