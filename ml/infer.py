"""Out-of-fold inference: reconstruct a full-volume prediction per case, score it.

For every case, the *only* model ever used to predict it is the one fold model
that held that case out during training — never any of the other 4 (see the
TTA_FLIP comment in config.py for why an all-fold ensemble would leak).

    python -m ml.infer --arch 25d
    python -m ml.infer --arch 3d
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
from scipy import ndimage

from . import config as C
from . import data as D
from . import metrics as M
from . import model as MDL
from .train import ARCH

PREDICTIONS = C.OUTPUTS / "predictions"


def load_fold_model(arch: str, fold: int, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(C.OUTPUTS / "checkpoints" / f"{arch}_fold{fold}.pt", map_location=device)
    model = ARCH[arch]["model_cls"]().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


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


# ── 2.5D: batched per-slice sliding window over the full depth ─────────────


@torch.no_grad()
def predict_prob_25d(model, images: np.ndarray, device: torch.device, *, tta: bool, batch: int = 32) -> np.ndarray:
    """images: (4, D, H, W) float. Returns a (D, H, W) probability volume."""
    depth = images.shape[1]
    ctx = C.SLICE_CONTEXT
    padded = np.pad(images, ((0, 0), (ctx, ctx), (0, 0), (0, 0)), mode="edge")  # clamp at the boundary

    prob = np.zeros((depth, *images.shape[2:]), dtype=np.float32)
    use_amp = device.type == "cuda"

    for start in range(0, depth, batch):
        end = min(start + batch, depth)
        stacks = []
        for z in range(start, end):
            # +ctx offsets the index into `padded`, which is `images` shifted by ctx.
            stacks.append(padded[:, z : z + 2 * ctx + 1].reshape(-1, *images.shape[2:]))
        x = torch.from_numpy(np.stack(stacks)).float().to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            p = torch.sigmoid(model(x))
            if tta:
                p_flip = torch.sigmoid(model(torch.flip(x, dims=[-1])))
                p = (p + torch.flip(p_flip, dims=[-1])) / 2
        prob[start:end] = p[:, 0].float().cpu().numpy()

    return prob


# ── 3D: sliding-window patches over the cropped volume ──────────────────────


def _window_starts(size: int, patch: int, stride: int) -> list[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


@torch.no_grad()
def predict_prob_3d(model, images: np.ndarray, device: torch.device, *, tta: bool) -> np.ndarray:
    """images: (4, Z, Y, X) float. Returns a (Z, Y, X) probability volume.

    Uniform-average sliding window (not Gaussian-weighted, unlike nnU-Net's
    default): simpler, and at 50% stride the accuracy difference is marginal
    next to the imbalance/normalisation issues this project actually stress-
    tests in F16 — flagged as a real simplification, not silently assumed away.
    """
    pz, py, px = C.PATCH_3D
    dz, dy, dx = images.shape[1:]
    stride = (pz // 2, py // 2, px // 2)

    prob_sum = np.zeros((dz, dy, dx), dtype=np.float32)
    count = np.zeros((dz, dy, dx), dtype=np.float32)
    use_amp = device.type == "cuda"

    for z in _window_starts(dz, pz, stride[0]):
        for y in _window_starts(dy, py, stride[1]):
            for x in _window_starts(dx, px, stride[2]):
                patch = images[:, z : z + pz, y : y + py, x : x + px]
                t = torch.from_numpy(np.ascontiguousarray(patch)).float().unsqueeze(0).to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    p = torch.sigmoid(model(t))
                    if tta:
                        p_flip = torch.sigmoid(model(torch.flip(t, dims=[-1])))
                        p = (p + torch.flip(p_flip, dims=[-1])) / 2
                prob_sum[z : z + pz, y : y + py, x : x + px] += p[0, 0].float().cpu().numpy()
                count[z : z + pz, y : y + py, x : x + px] += 1

    return prob_sum / np.maximum(count, 1)


# ── driver ───────────────────────────────────────────────────────────────────


def predict_case(arch: str, cid: str, model, device: torch.device, *, tta: bool) -> np.ndarray:
    images, _, brain = D.load_cached(cid, mmap=False)
    images = np.asarray(images).astype(np.float32)
    if arch == "25d":
        prob = predict_prob_25d(model, images, device, tta=tta)
    else:
        prob = predict_prob_3d(model, images, device, tta=tta)
    prob = prob * brain  # never predict foreground outside the skull-stripped brain
    return prob


#: Grid swept per case to pick the operating point (figure F08). Computed
#: inline from the probability volume already in memory rather than cached to
#: disk — storing every case's full probability volume (~22 MB each) would
#: cost over a GB for a number that only needs the resulting curve.
THRESHOLD_GRID = np.round(np.arange(0.3, 0.75, 0.05), 2)


def run(
    arch: str, ids: list[str], device: torch.device, *, tta: bool = True, verbose: bool = True,
    sweep_thresholds: bool = True,
) -> dict:
    folds = C.fold_assignment(ids)
    models = {}
    results = {}
    sweep = {float(t): [] for t in THRESHOLD_GRID} if sweep_thresholds else {}

    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    (PREDICTIONS / arch).mkdir(exist_ok=True)

    t0 = time.time()
    for cid in ids:
        f = folds[cid]
        if f not in models:
            models[f], _ = load_fold_model(arch, f, device)

        prob = predict_case(arch, cid, models[f], device, tta=tta)
        pred = _keep_largest_components(prob > C.DEFAULT_THRESHOLD)

        _, labels, _ = D.load_cached(cid, mmap=False)
        gt = np.asarray(labels) > 0
        m = M.case_metrics(pred, gt)
        m["fold"] = f
        results[cid] = m

        if sweep_thresholds:
            for t in THRESHOLD_GRID:
                sweep[float(t)].append(M.dice(prob > t, gt))  # no postprocessing here — see F13 for that ablation

        np.savez_compressed(PREDICTIONS / arch / f"{cid}.npz", pred=pred.astype(np.uint8))
        if verbose:
            print(f"  {cid}  fold {f}  dice {m['dice']:.3f}  hd95 {m['hd95']:.1f}")

    elapsed = time.time() - t0
    dices = np.array([v["dice"] for v in results.values()])
    lo, hi = M.bootstrap_ci(dices)
    summary = {
        "arch": arch, "tta": tta, "n": len(dices),
        "mean_dice": float(dices.mean()), "median_dice": float(np.median(dices)),
        "min_dice": float(dices.min()), "max_dice": float(dices.max()),
        "mean_dice_ci95": [lo, hi],
        "wall_clock_minutes": elapsed / 60,
        "threshold_sweep": {t: float(np.mean(v)) for t, v in sweep.items()} if sweep_thresholds else None,
    }
    return {"per_case": results, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=["25d", "3d"])
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--tier", default="standard", choices=["standard", "tiny"])
    args = ap.parse_args(argv)

    device = MDL.pick_device()
    ids = C.case_ids(args.tier)
    out = run(args.arch, ids, device, tta=not args.no_tta)

    suffix = "_notta" if args.no_tta else ""
    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    (C.OUTPUTS / f"infer_{args.arch}{suffix}.json").write_text(json.dumps(out, indent=2))

    s = out["summary"]
    print(f"\n{args.arch}{' (no TTA)' if args.no_tta else ''}: "
          f"mean {s['mean_dice']:.3f} (95% CI {s['mean_dice_ci95'][0]:.3f}-{s['mean_dice_ci95'][1]:.3f})  "
          f"median {s['median_dice']:.3f}  min {s['min_dice']:.3f}  max {s['max_dice']:.3f}  "
          f"n={s['n']}  {s['wall_clock_minutes']:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
