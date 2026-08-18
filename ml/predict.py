"""Inference on NEW, unlabelled cases — the test-set entry point.

Distinct from `ml/infer.py`, and the difference matters:

    ml/infer.py    scores the 60 development cases OUT OF FOLD. Case i is
                   predicted by fold i's model only, because the other four
                   models trained on it. Requires ground-truth labels. This is
                   how the pre-registered Dice is produced and it is the only
                   honest way to score these 60.

    ml/predict.py  predicts cases the project has NEVER seen. No labels needed,
                   and it ENSEMBLES all five fold models, which is legitimate
                   here for exactly the reason it is forbidden there: a genuinely
                   new scan was not in anybody's training set, so averaging the
                   five costs nothing and reliably helps. See config.TTA_FLIP's
                   comment — the ensemble was always described as "a real
                   artifact worth shipping for a genuinely new scan"; this is
                   that artifact.

Preprocessing is byte-identical to training because `ml.data.normalise` is
per-case and self-contained: a z-score inside each case's own brain mask, with
no dataset-level statistics. A new case therefore needs no reference to the
cohort at all.

    python -m ml.predict --input-dir DIR --output-dir DIR
    python -m ml.predict --input-dir DIR --output-dir DIR --no-ensemble --fold 2
    python -m ml.predict --self-test                # verify the path end-to-end

Outputs, per case, into --output-dir:
    <case>_mask.nii.gz    binary prediction in ORIGINAL 240x240x155 geometry
    predictions.json      per-case volume_mm3, confidence, verdict, timings
    <case>.npz            same mask in ml/infer.py's format, so
                          `python -m ml.export_web --full` can publish it
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback

import numpy as np
import torch
from scipy import ndimage

from . import config as C
from . import data as D
from . import model as MDL
from .infer import _keep_largest_components, predict_prob_25d, predict_prob_3d

#: A predicted whole-tumour volume below this is reported as a negative finding
#: rather than a tiny detection. Set to MIN_COMPONENT_VOXELS so it agrees with
#: the postprocessing that already removed anything smaller: reporting "tumour
#: found, 300 mm3" for a blob the postprocessor deleted would be incoherent.
MIN_REPORTABLE_MM3 = C.MIN_COMPONENT_VOXELS * C.VOXEL_VOLUME_MM3


def load_case_nifti(path: pathlib.Path) -> tuple[np.ndarray, "object", dict]:
    """Load one unlabelled 4-modality NIfTI. Returns (images (C,Z,Y,X), affine, info).

    Mirrors `data.load_raw`'s `.T` convention but does not require a label file
    and does not hard-fail on unexpected geometry — a test set we have not seen
    is exactly where an unchecked shape assumption would bite, so the shape is
    reported and carried rather than asserted away.
    """
    import nibabel as nib

    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    info: dict = {"source": path.name, "raw_shape": list(arr.shape)}

    if arr.ndim == 5 and arr.shape[3] == 1:  # (x,y,z,1,t), a common NIfTI quirk
        arr = arr[:, :, :, 0, :]
        info["squeezed_axis3"] = True
    if arr.ndim != 4:
        raise ValueError(f"expected a 4D (x,y,z,modality) volume, got shape {arr.shape}")
    if arr.shape[3] != C.N_CHANNELS:
        raise ValueError(f"expected {C.N_CHANNELS} modalities in the last axis, got {arr.shape[3]}")

    images = np.ascontiguousarray(arr.T, dtype=np.float32)  # -> (C, Z, Y, X)
    info["zyx_shape"] = list(images.shape[1:])
    info["nonstandard_geometry"] = tuple(arr.shape[:3]) != C.VOLUME_SHAPE
    return images, img.affine, info


def preprocess(images: np.ndarray, info: dict) -> tuple[np.ndarray, np.ndarray, bool]:
    """Brain-mask, z-score and crop exactly as training did.

    Returns (images, brain, cropped). `cropped` is False when the canonical crop
    would clip brain voxels — in that case the full volume is used instead. The
    model is fully convolutional and inference is a sliding window, so a larger
    input is safe; silently amputating a test case's brain would not be.
    """
    brain = D.brain_mask(images)
    if not brain.any():
        raise ValueError("brain mask is empty — all four modalities are zero everywhere")
    images = D.normalise(images, brain)

    zs, ze = C.CROP_Z
    ys, ye = C.CROP_Y
    xs, xe = C.CROP_X
    fits = (
        images.shape[1:] == (C.CROP_D + 0, C.VOLUME_SHAPE[1], C.VOLUME_SHAPE[0])
        or images.shape[1:] == (C.VOLUME_SHAPE[2], C.VOLUME_SHAPE[1], C.VOLUME_SHAPE[0])
    )
    clipped = int(brain.sum() - brain[zs:ze, ys:ye, xs:xe].sum()) if fits else -1

    if fits and clipped == 0:
        info["crop"] = "canonical"
        return D.crop(images), D.crop(brain), True

    info["crop"] = "none (full volume)"
    info["crop_would_clip_voxels"] = clipped
    return images, brain, False


@torch.no_grad()
def predict_prob(
    models: list[torch.nn.Module], images: np.ndarray, brain: np.ndarray,
    device: torch.device, *, arch: str, tta: bool,
) -> np.ndarray:
    """Mean sigmoid over every supplied model, each optionally TTA-flipped."""
    acc = None
    for m in models:
        fn = predict_prob_25d if arch == "25d" else predict_prob_3d
        p = fn(m, images, device, tta=tta)
        acc = p if acc is None else acc + p
    prob = acc / len(models)
    return prob * brain  # never predict tumour outside the skull-stripped brain


def summarise(pred: np.ndarray, prob: np.ndarray) -> dict:
    """Turn a mask into the fields the website's Detection contract expects."""
    voxels = int(pred.sum())
    volume = float(voxels * C.VOXEL_VOLUME_MM3)
    inside = prob[pred] if voxels else np.array([], dtype=np.float32)

    # "Confidence" here is the mean predicted probability inside the retained
    # mask. It is a legible summary of how sure the network was, NOT a
    # calibrated probability that a tumour exists — no calibration was fitted
    # and none is claimed. Reported as such so the website cannot imply more.
    confidence = float(inside.mean()) if voxels else 0.0
    _, n_comp = ndimage.label(pred)
    return {
        "voxels": voxels,
        "volume_mm3": volume,
        "confidence": confidence,
        "confidence_is_calibrated": False,
        "n_components": int(n_comp),
        "verdict": "tumour detected" if volume >= MIN_REPORTABLE_MM3 else "no tumour detected",
        "prob_max": float(prob.max()),
    }


def check_gpu_free(device: torch.device, *, need_gb: float = 4.0) -> float | None:
    """Fail early and legibly if another process is holding the card.

    Training peaks at ~23.0 GiB of 23.6 GiB on this machine, so anything else
    running leaves almost nothing. Without this check the symptom is a bare
    `CUDA error: out of memory` from somewhere deep in a conv, which is a poor
    thing to debug in a hurry with a test set waiting.
    """
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info()
    free_gb, total_gb = free / 1e9, total / 1e9
    if free_gb < need_gb:
        raise SystemExit(
            f"only {free_gb:.2f} GB of {total_gb:.1f} GB GPU memory free; need ~{need_gb:.0f} GB.\n"
            f"Something else is using the card. Check with:\n"
            f"    nvidia-smi --query-compute-apps=pid,used_memory --format=csv\n"
            f"If it is the overnight run, either wait for it or stop it:\n"
            f"    pkill -f 'ml.train'   # then re-run this command\n"
            f"To run on CPU instead (slow, minutes per case):  CUDA_VISIBLE_DEVICES='' python3 -m ml.predict ..."
        )
    return free_gb


def load_models(arch: str, device: torch.device, *, ckpt_dir: pathlib.Path,
                folds: list[int] | None) -> tuple[list[torch.nn.Module], list[dict]]:
    from .train import ARCH

    wanted = list(range(C.N_FOLDS)) if folds is None else folds
    models, meta = [], []
    for f in wanted:
        p = ckpt_dir / f"{arch}_fold{f}.pt"
        if not p.exists():
            raise SystemExit(f"missing checkpoint {p} — train first, or pass --folds/--ckpt-dir")
        ck = torch.load(p, map_location=device, weights_only=False)
        ds = ck.get("deep_supervision", False)
        # Deep supervision changes the state_dict key set (aux_heads.*), and
        # load_state_dict is strict. Build the model to match the checkpoint
        # rather than to match the current config, so a checkpoint trained
        # under the other setting still loads instead of erroring at 3am.
        m = ARCH[arch]["model_cls"](deep_supervision=ds).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)
        meta.append({"fold": f, "epoch": ck.get("epoch"), "val_dice_proxy": ck.get("val_dice"),
                     "deep_supervision": ds, "checkpoint": str(p)})
    return models, meta


def _find_label(labels_dir: pathlib.Path, cid: str) -> pathlib.Path | None:
    for pattern in (f"{cid}.nii.gz", f"{cid}.nii", f"{cid}_seg.nii.gz", f"{cid}_mask.nii.gz"):
        p = labels_dir / pattern
        if p.exists():
            return p
    hits = sorted(labels_dir.glob(f"{cid}*"))
    return hits[0] if hits else None


def _score(pred: np.ndarray, label_path: pathlib.Path, cropped: bool) -> dict:
    """Score a prediction against a supplied ground truth.

    Unlike ml/infer.py this is NOT out-of-fold and does not need to be: a real
    test set was in nobody's training fold, so ensembling all five models and
    scoring against these labels is honest. It is simply not the same quantity as
    the pre-registered out-of-fold number and must not be compared to it directly.
    """
    import nibabel as nib

    from . import metrics as M

    gt_full = np.ascontiguousarray(np.asanyarray(nib.load(str(label_path)).dataobj).T) > 0
    gt = D.crop(gt_full) if cropped else gt_full
    if gt.shape != pred.shape:
        raise ValueError(f"label shape {gt.shape} != prediction shape {pred.shape}")
    m = M.case_metrics(pred, gt)
    m["true_volume_mm3"] = float(gt.sum() * C.VOXEL_VOLUME_MM3)
    m["label_source"] = label_path.name
    return m


def run(
    inputs: list[pathlib.Path], out_dir: pathlib.Path, *, arch: str = "3d", tta: bool = True,
    folds: list[int] | None = None, ckpt_dir: pathlib.Path | None = None,
    threshold: float = C.DEFAULT_THRESHOLD, save_nifti: bool = True, verbose: bool = True,
    labels_dir: pathlib.Path | None = None,
) -> dict:
    import nibabel as nib

    ckpt_dir = ckpt_dir or (C.OUTPUTS / "checkpoints")
    device = MDL.pick_device()
    free_gb = check_gpu_free(device)
    models, meta = load_models(arch, device, ckpt_dir=ckpt_dir, folds=folds)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        kind = f"{len(models)}-fold ensemble" if len(models) > 1 else f"single fold {meta[0]['fold']}"
        print(f"device={device}  arch={arch}  {kind}  tta={tta}  threshold={threshold}")
        print(f"{len(inputs)} case(s) -> {out_dir}\n")

    results, failures = {}, {}
    t_all = time.time()
    for path in inputs:
        cid = path.name.split(".nii")[0]
        t0 = time.time()
        try:
            images, affine, info = load_case_nifti(path)
            proc, brain, cropped = preprocess(images, info)
            prob = predict_prob(models, proc, brain, device, arch=arch, tta=tta)
            pred = _keep_largest_components(prob > threshold)

            summary = summarise(pred, prob)
            summary.update(info)

            if labels_dir is not None:
                lp = _find_label(labels_dir, cid)
                if lp is None:
                    summary["scoring_skipped"] = f"no label file matching {cid} in {labels_dir}"
                else:
                    summary.update(_score(pred, lp, cropped))

            summary["seconds"] = round(time.time() - t0, 2)
            results[cid] = summary

            # Back to the original geometry so the mask overlays the input file.
            full = D.uncrop(pred, fill=False) if cropped else pred
            if save_nifti:
                nib.save(nib.Nifti1Image(full.astype(np.uint8).T, affine),
                         str(out_dir / f"{cid}_mask.nii.gz"))
            # infer.py's format, so export_web.py --full can publish it unchanged.
            np.savez_compressed(out_dir / f"{cid}.npz", pred=pred.astype(np.uint8))

            if verbose:
                extra = f"  dice {summary['dice']:.4f}" if "dice" in summary else ""
                print(f"  {cid:24s} {summary['verdict']:20s} {summary['volume_mm3']:9.0f} mm3  "
                      f"conf {summary['confidence']:.3f}{extra}  {summary['seconds']:.1f}s")
        except Exception as e:
            # One malformed case must not abort a batch that runs unattended.
            failures[cid] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
            if verbose:
                print(f"  {cid:24s} FAILED  {type(e).__name__}: {e}")

    out = {
        "arch": arch, "tta": tta, "threshold": threshold, "ensemble_folds": [m["fold"] for m in meta],
        "models": meta, "n_ok": len(results), "n_failed": len(failures),
        "wall_clock_minutes": (time.time() - t_all) / 60,
        "per_case": results, "failures": failures,
    }

    scored = [v["dice"] for v in results.values() if "dice" in v]
    if scored:
        from . import metrics as M

        d = np.array(scored)
        lo, hi = M.bootstrap_ci(d)
        out["scored_summary"] = {
            "n": len(d), "mean_dice": float(d.mean()), "median_dice": float(np.median(d)),
            "min_dice": float(d.min()), "max_dice": float(d.max()), "mean_dice_ci95": [lo, hi],
            # Spelled out in the artifact so nobody has to remember it later.
            "note": ("5-fold ENSEMBLE + TTA on unseen cases. Honest (these cases were in no "
                     "training fold) but NOT the same quantity as the pre-registered "
                     "out-of-fold single-model number of 0.8869 — do not compare directly."),
        }
        if verbose:
            s = out["scored_summary"]
            print(f"\nSCORED: mean Dice {s['mean_dice']:.4f} CI95 "
                  f"[{s['mean_dice_ci95'][0]:.4f}, {s['mean_dice_ci95'][1]:.4f}]  "
                  f"median {s['median_dice']:.4f}  worst {s['min_dice']:.4f}  (n={s['n']})")
    (out_dir / "predictions.json").write_text(json.dumps(out, indent=2))
    if verbose:
        print(f"\n{len(results)} ok, {len(failures)} failed, {out['wall_clock_minutes']:.1f} min")
        print(f"wrote {out_dir / 'predictions.json'}")
    return out


def self_test(n: int = 3) -> int:
    """Run the real test-set path over known cases and check it against ground truth.

    Deliberately goes through `load_case_nifti` and `preprocess` from the raw
    NIfTI rather than the training cache, so it exercises the same code a test
    set will hit. Because it ensembles all five folds over cases that four of
    them trained on, the Dice it prints is LEAKED and meaningless as a score —
    it is a smoke test that the plumbing works, not a measurement. The honest
    number for these cases is ml/infer.py's out-of-fold result.
    """
    from . import metrics as M

    ids = C.case_ids()[:n]
    out = C.OUTPUTS / "predict_selftest"
    print(f"SELF-TEST on {ids} (leaked Dice — plumbing check only)\n")
    r = run([C.IMAGES / f"{c}.nii.gz" for c in ids], out, verbose=True)
    if r["n_failed"]:
        print("\nFAIL: some cases errored")
        for cid, f in r["failures"].items():
            print(f"  {cid}: {f['error']}")
        return 1

    print()
    ok = True
    for cid in ids:
        pred = np.load(out / f"{cid}.npz")["pred"].astype(bool)
        _, labels, _ = D.load_cached(cid, mmap=False)
        gt = np.asarray(labels) > 0
        d = M.dice(pred, gt)
        shape_ok = pred.shape == gt.shape
        nii = out / f"{cid}_mask.nii.gz"
        print(f"  {cid}  dice(leaked) {d:.4f}  shape_match={shape_ok}  nifti={nii.exists()}")
        if not shape_ok or not nii.exists() or d < 0.5:
            ok = False
    print(f"\n{'PASS' if ok else 'FAIL'}: test-set inference path "
          f"{'works end-to-end' if ok else 'has a problem'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=pathlib.Path, help="directory of 4-modality .nii.gz files")
    ap.add_argument("--inputs", type=pathlib.Path, nargs="*", help="explicit file list instead")
    ap.add_argument("--output-dir", type=pathlib.Path, default=C.OUTPUTS / "predict")
    ap.add_argument("--labels-dir", type=pathlib.Path, default=None,
                    help="optional ground truth; if given, scores each case (honest for a real test set)")
    ap.add_argument("--arch", default="3d", choices=["25d", "3d"])
    ap.add_argument("--ckpt-dir", type=pathlib.Path, default=None)
    ap.add_argument("--folds", type=str, default=None, help="e.g. '2' or '0,1,2'; default all 5 (ensemble)")
    ap.add_argument("--no-ensemble", action="store_true", help="use fold 0 only unless --folds given")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--threshold", type=float, default=C.DEFAULT_THRESHOLD)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.input_dir:
        inputs = sorted(args.input_dir.glob("*.nii.gz")) or sorted(args.input_dir.glob("*.nii"))
    elif args.inputs:
        inputs = list(args.inputs)
    else:
        ap.error("pass --input-dir, --inputs, or --self-test")
    if not inputs:
        ap.error(f"no .nii.gz files found in {args.input_dir}")

    folds = [int(x) for x in args.folds.split(",")] if args.folds else ([0] if args.no_ensemble else None)
    r = run(inputs, args.output_dir, arch=args.arch, tta=not args.no_tta, folds=folds,
            ckpt_dir=args.ckpt_dir, threshold=args.threshold, labels_dir=args.labels_dir)
    return 1 if r["n_failed"] and not r["n_ok"] else 0


if __name__ == "__main__":
    sys.exit(main())
