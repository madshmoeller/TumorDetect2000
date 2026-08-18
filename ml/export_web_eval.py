"""Export the 60 unlabelled eval cases to the website, and tag both splits.

The training export (`ml/export_web.py`) cannot do this: every one of its
label-derived fields — the ground-truth atlas, the thumbnail tint, the
best-slice index, `trueVolumeMm3`, the sub-label breakdown — needs a mask that
does not exist for the eval set. Rather than thread "maybe there is no label"
through that module and risk the working train export, this one builds the
label-free entries and merges them into the manifest.

What replaces the missing ground truth:

    truth atlas      -> None. The viewer already guards on it, so the EXPERT
                        MASK toggle simply has nothing to draw.
    thumbnail tint   -> the model's PREDICTION, tinted cyan rather than the
                        ground truth's pink/orange, so a glance at the grid
                        never implies a human drew it.
    bestIndex        -> the predicted lesion's largest slice.
    maskAreaBySlice  -> predicted area per slice.
    dice / hd95      -> None, which is what makes the viewer hide the
                        prediction-accuracy panel. There is no ground truth to
                        score against and none is invented.

    python -m ml.export_web_eval            # export eval cases + merge manifest
    python -m ml.export_web_eval --tag-only # just add split tags to the manifest
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from PIL import Image

from . import config as C
from . import data as D
from .export_web import (ORIENTATIONS, PREDICTION_COLOR, THUMB_SIZE, THUMB_TINT_ALPHA,
                         _profile, _slice_indices, build_modality_atlas,
                         build_prediction_atlas)

EVAL_IMAGES = C.ROOT / "agentic-medical-ai-lab" / "data" / "eval" / "images"
EVAL_PRED = C.OUTPUTS / "kaggle_eval"
MANIFEST = C.WEB_CASES / "manifest.json"


def eval_case_ids() -> list[str]:
    tier = EVAL_IMAGES.parent / "tier_eval.txt"
    if tier.exists():
        return [l.strip() for l in tier.read_text().splitlines()
                if l.strip() and not l.startswith("#")]
    return sorted(p.name.split(".nii")[0] for p in EVAL_IMAGES.glob("*.nii.gz"))


def load_eval_case(case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (images, brain, pred) all in the canonical crop space.

    Preprocessing matches training exactly — `data.normalise` is a per-case
    z-score inside the brain mask, so no cohort statistics are involved.

    The canonical crop is forced here even for the one case whose brain exceeds
    it by 140 voxels (eval_004). That is a DISPLAY decision only: the submitted
    mask for that case was produced from the full volume and is unaffected. The
    atlas grid geometry is fixed by config.ORIENTATION_*, so a differently-shaped
    volume cannot be tiled into it.
    """
    from .predict import load_case_nifti

    images, _affine, _info = load_case_nifti(EVAL_IMAGES / f"{case_id}.nii.gz")
    brain = D.brain_mask(images)
    images = D.normalise(images, brain)

    pred = np.load(EVAL_PRED / f"{case_id}.npz")["pred"].astype(bool)
    target = (C.CROP_D, *C.CROP_HW)
    if pred.shape != target:                     # produced without the crop
        pred = D.crop(pred)
    return D.crop(images), D.crop(brain), pred


def build_pred_thumb(images: np.ndarray, brain: np.ndarray, pred: np.ndarray, best_z: int) -> Image.Image:
    """FLAIR at the largest predicted slice, prediction tinted CYAN.

    Cyan, not the ground truth's pink/orange: the grid must not let a predicted
    outline pass for an expert one at a glance.
    """
    disp = D.window_to_uint8(images[0].astype(np.float32), brain)[best_z]
    rgb = np.stack([disp] * 3, axis=-1).astype(np.float32)
    m = pred[best_z]
    tint = np.array(PREDICTION_COLOR, dtype=np.float32)
    rgb[m] = rgb[m] * (1 - THUMB_TINT_ALPHA) + tint * THUMB_TINT_ALPHA
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB").resize(
        (THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)


def export_eval_case(case_id: str, out_dir) -> dict:
    images, brain, pred = load_eval_case(case_id)
    case_dir = out_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    disp_vols = [D.window_to_uint8(images[c], brain) for c in range(C.N_CHANNELS)]
    channels, masks, best_index, area = {}, {}, {}, {}

    for orientation in ORIENTATIONS:
        z_indices = _slice_indices(orientation)

        paths = {}
        for c, key in enumerate(C.MODALITY_KEYS):
            atlas = build_modality_atlas(disp_vols[c], orientation, z_indices)
            p = case_dir / f"{key}_{orientation}.png"
            Image.fromarray(atlas, mode="L").save(p, optimize=True)
            paths[key] = str(p.relative_to(C.ROOT))
        channels[orientation] = paths

        atlas = build_prediction_atlas(pred, orientation, z_indices)
        pp = case_dir / f"pred_{orientation}.png"
        Image.fromarray(atlas, mode="RGBA").save(pp, optimize=True)
        # truth is None, not a missing key: the viewer reads
        # masks[orientation].truth and must find an explicit null.
        masks[orientation] = {"truth": None, "prediction": str(pp.relative_to(C.ROOT))}

        profile = _profile(pred, orientation)
        best_raw = int(profile.argmax())
        best_index[orientation] = int(np.argmin(np.abs(np.array(z_indices) - best_raw)))
        area[orientation] = [int(profile[i]) for i in z_indices]

    best_z = int(_profile(pred, "axial").argmax())
    thumb = case_dir / "thumb.png"
    build_pred_thumb(images, brain, pred, best_z).save(thumb, optimize=True)

    return {
        "id": case_id,
        "label": case_id.upper(),
        "split": "test",
        "thumb": str(thumb.relative_to(C.ROOT)),
        "channels": channels,
        "masks": masks,
        "bestIndex": best_index,
        "maskAreaBySlice": area,
        "metrics": {
            # No ground truth exists for these cases, so every truth-derived
            # field is null rather than zero — zero would read as "no tumour".
            "trueVolumeMm3": None,
            "sublabelsMm3": None,
            "positiveSlices": None,
            "predictedVolumeMm3": int(pred.sum()),
            "dice": None,
            "hd95": None,
        },
    }


def merge(entries: list[dict], *, tag_only: bool = False) -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} not found — run `python -m ml.export_web --full` first")
    m = json.loads(MANIFEST.read_text())

    # Existing entries are the labelled development cohort.
    for c in m["cases"]:
        c.setdefault("split", "train")

    if not tag_only:
        by_id = {c["id"]: c for c in m["cases"]}
        for e in entries:
            by_id[e["id"]] = e
        # train first, then test, each in id order
        m["cases"] = sorted(by_id.values(), key=lambda c: (c.get("split") != "train", c["id"]))

    counts = {}
    for c in m["cases"]:
        counts[c.get("split", "train")] = counts.get(c.get("split", "train"), 0) + 1
    m["splits"] = counts
    m["nCases"] = len(m["cases"])
    MANIFEST.write_text(json.dumps(m, indent=2))
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag-only", action="store_true",
                    help="only add split tags to existing manifest entries")
    ap.add_argument("--limit", type=int, default=None, help="export only the first N eval cases")
    args = ap.parse_args(argv)

    entries = []
    if not args.tag_only:
        ids = eval_case_ids()
        if args.limit:
            ids = ids[: args.limit]
        missing = [c for c in ids if not (EVAL_PRED / f"{c}.npz").exists()]
        if missing:
            raise SystemExit(f"no prediction for {len(missing)} case(s), e.g. {missing[:3]} — "
                             f"run `python -m ml.predict --input-dir {EVAL_IMAGES} "
                             f"--output-dir {EVAL_PRED}` first")
        C.WEB_CASES.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, cid in enumerate(ids, 1):
            print(f"\r  exporting {i}/{len(ids)}  {cid}", end="", flush=True)
            entries.append(export_eval_case(cid, C.WEB_CASES))
        print(f"\n  {len(entries)} eval cases in {time.time() - t0:.0f}s")

    m = merge(entries, tag_only=args.tag_only)
    print(f"manifest: {m['nCases']} cases  splits={m['splits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
