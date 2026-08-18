"""Export the dataset to web-viewable slice atlases + a manifest.

Act II (`--ground-truth-only`): for every case, one greyscale atlas per
modality per viewing plane, one per-sublabel tumour-mask atlas per plane, and
a small thumbnail — all built from the *ground truth* only, no model exists
yet.

Act III (`--full --arch {25d,3d}`): a second pass that additionally reads
`outputs/predictions/<arch>/<case>.npz` and `outputs/infer_<arch>.json`
(written by `ml.infer`), builds one prediction-mask atlas per orientation per
case, and merges the real dice/hd95/predicted-volume numbers into the same
manifest fields `export_case` already writes as `None` in Act II — additive,
not a rewrite, which is why `app.js`/`viewer.js` never had to change shape.

Atlas format: every ATLAS_STEP'th slice along one axis, tiled into one PNG per
modality (COLS x ROWS grid of TILE x TILE tiles). The browser loads each atlas
once and blits tiles with `drawImage`, so scrubbing through slices costs zero
network requests. Three viewing planes (axial/coronal/sagittal) each get their
own atlas set, all sharing the same square tile size — see
`config.ORIENTATION_*` for how the non-axial planes get padded to fit it.
Geometry lives in `ml/config.py` and is echoed into the manifest so the
frontend never hardcodes it.

    python -m ml.export_web --ground-truth-only
    python -m ml.export_web --full --arch 3d
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

#: The dataset ships three tumour sub-labels (oedema, non-enhancing core,
#: enhancing), not one — the model's *target* collapses them to a single
#: whole-lesion class (see RULES.md), but the ground truth the viewer shows
#: doesn't have to, and the composition bar/legend already displayed these
#: three volumes in these exact colours. This atlas used to paint a flat
#: binary tint that didn't match its own legend; it now doesn't.
#: Matches `SUBLABEL_COLORS` in figstyle.py and viewer.js — one palette,
#: three places, kept in sync by hand because there's no shared JSON source
#: for it yet.
SUBLABEL_COLOR = {
    1: (42, 120, 214),  # oedema             — #2a78d6
    2: (235, 104, 52),  # non-enhancing core — #eb6834
    3: (27, 175, 122),  # enhancing tumour   — #1baf7a
}
TRUTH_ALPHA = 190
THUMB_SIZE = 176
THUMB_TINT_ALPHA = 0.55

#: Single flat colour for the model's binary prediction mask — matches
#: figstyle.PREDICTION (#00e0f0), the cyan already used for prediction layers
#: in the legacy mock GUI, so a real prediction reads as "the same kind of
#: thing" the demo already trained the eye on.
PREDICTION_COLOR = (0, 224, 240)
PREDICTION_ALPHA = 190

ORIENTATIONS = tuple(C.ORIENTATION_AXIS)  # ("axial", "coronal", "sagittal")


# ── generic orientation slicing ─────────────────────────────────────────────


def _pad_to_tile(slice2d: np.ndarray) -> np.ndarray:
    """Zero-pad a (h, ATLAS_TILE) slice up to a square (ATLAS_TILE, ATLAS_TILE)
    tile. Axial slices are already square and pass through untouched.
    """
    t = C.ATLAS_TILE
    h, w = slice2d.shape
    assert w == t, f"unexpected slice width {w}, expected {t}"
    if h == t:
        return slice2d
    out = np.zeros((t, t), dtype=slice2d.dtype)
    top = (t - h) // 2
    out[top : top + h, :] = slice2d
    return out


def orientation_slice(volume: np.ndarray, orientation: str, index: int) -> np.ndarray:
    """volume: (Z, Y, X). Returns the 2D slice at `index` along the given
    plane's axis, zero-padded to a square ATLAS_TILE tile."""
    return _pad_to_tile(np.take(volume, index, axis=C.ORIENTATION_AXIS[orientation]))


def _slice_indices(orientation: str) -> list[int]:
    return list(range(*C.ORIENTATION_RANGE[orientation], C.ORIENTATION_STEP[orientation]))


def _profile(volume_bool: np.ndarray, orientation: str) -> np.ndarray:
    """Sum a boolean (Z, Y, X) volume over the two axes *not* in `orientation`
    — e.g. for coronal, the tumour cross-section area at every Y."""
    axis = C.ORIENTATION_AXIS[orientation]
    other = tuple(a for a in range(3) if a != axis)
    return volume_bool.sum(axis=other)


# ── atlas builders ───────────────────────────────────────────────────────────


def build_modality_atlas(disp_vol: np.ndarray, orientation: str, z_indices: list[int]) -> np.ndarray:
    """disp_vol: (Z, Y, X) uint8, already windowed over the whole case volume
    (matching `eda.fig03_example`) — per-slice windowing would make brightness
    flicker as you scrub, since a slice with little brain would get the same
    contrast stretch as one that is mostly brain.
    """
    t = C.ATLAS_TILE
    rows, cols = C.ORIENTATION_ROWS[orientation], C.ORIENTATION_COLS
    atlas = np.zeros((rows * t, cols * t), dtype=np.uint8)
    for i, idx in enumerate(z_indices):
        row, col = divmod(i, cols)
        atlas[row * t : (row + 1) * t, col * t : (col + 1) * t] = orientation_slice(disp_vol, orientation, idx)
    return atlas


def build_mask_atlas(labels: np.ndarray, orientation: str, z_indices: list[int]) -> np.ndarray:
    """RGBA atlas, one colour per tumour sub-label, alpha baked in."""
    t = C.ATLAS_TILE
    rows, cols = C.ORIENTATION_ROWS[orientation], C.ORIENTATION_COLS
    atlas = np.zeros((rows * t, cols * t, 4), dtype=np.uint8)
    for i, idx in enumerate(z_indices):
        row, col = divmod(i, cols)
        lab_tile = orientation_slice(labels, orientation, idx)
        tile = np.zeros((t, t, 4), dtype=np.uint8)
        for sub, colour in SUBLABEL_COLOR.items():
            m = lab_tile == sub
            tile[m, 0], tile[m, 1], tile[m, 2] = colour
            tile[m, 3] = TRUTH_ALPHA
        atlas[row * t : (row + 1) * t, col * t : (col + 1) * t] = tile
    return atlas


def build_prediction_atlas(pred: np.ndarray, orientation: str, z_indices: list[int]) -> np.ndarray:
    """Same tiling as `build_mask_atlas`, one flat colour, binary rather than per-sublabel —
    the model predicts whole-lesion only (RULES.md), it has no sub-label to colour by."""
    t = C.ATLAS_TILE
    rows, cols = C.ORIENTATION_ROWS[orientation], C.ORIENTATION_COLS
    atlas = np.zeros((rows * t, cols * t, 4), dtype=np.uint8)
    for i, idx in enumerate(z_indices):
        row, col = divmod(i, cols)
        tile_mask = orientation_slice(pred, orientation, idx) > 0
        tile = np.zeros((t, t, 4), dtype=np.uint8)
        tile[tile_mask, 0], tile[tile_mask, 1], tile[tile_mask, 2] = PREDICTION_COLOR
        tile[tile_mask, 3] = PREDICTION_ALPHA
        atlas[row * t : (row + 1) * t, col * t : (col + 1) * t] = tile
    return atlas


def export_prediction(case_id: str, out_dir, arch: str, case_metrics: dict) -> dict:
    """Act III addition: prediction atlases + real metrics for one case.

    Returns the fragment to merge into that case's manifest entry — never
    constructs a new entry, since `export_case`'s Act II fields (channels,
    ground-truth masks, thumb) are already correct and untouched.
    """
    pred_path = C.OUTPUTS / "predictions" / arch / f"{case_id}.npz"
    pred = np.load(pred_path)["pred"].astype(bool)  # (Z, Y, X), same crop space as labels

    case_dir = out_dir / case_id
    masks_by_orientation = {}
    for orientation in ORIENTATIONS:
        z_indices = _slice_indices(orientation)
        atlas = build_prediction_atlas(pred, orientation, z_indices)
        p = case_dir / f"pred_{orientation}.png"
        Image.fromarray(atlas, mode="RGBA").save(p, optimize=True)
        masks_by_orientation[orientation] = str(p.relative_to(C.ROOT))

    return {
        "masks_prediction": masks_by_orientation,
        "predictedVolumeMm3": int(pred.sum()),
        "dice": case_metrics["dice"],
        "hd95": case_metrics["hd95"],
    }


def build_thumb(images: np.ndarray, brain: np.ndarray, labels: np.ndarray, best_z: int) -> Image.Image:
    """FLAIR at the largest-lesion axial slice, sub-labels tinted in, for the case grid."""
    flair_vol = D.window_to_uint8(images[0].astype(np.float32), brain)
    disp = flair_vol[best_z]
    rgb = np.stack([disp] * 3, axis=-1).astype(np.float32)
    lab_slice = labels[best_z]
    for sub, colour in SUBLABEL_COLOR.items():
        m = lab_slice == sub
        tint = np.array(colour, dtype=np.float32)
        rgb[m] = rgb[m] * (1 - THUMB_TINT_ALPHA) + tint * THUMB_TINT_ALPHA
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    return img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)


def export_case(case_id: str, out_dir) -> dict:
    images, labels, brain = D.load_cached(case_id, mmap=False)
    images = np.asarray(images).astype(np.float32)
    labels = np.asarray(labels)
    brain = np.asarray(brain)
    tumour = labels > 0

    case_dir = out_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # Windowing is per-channel, over the whole volume, computed once here and
    # reused across all three orientations' atlases.
    disp_vols = [D.window_to_uint8(images[c], brain) for c in range(C.N_CHANNELS)]

    channels_by_orientation: dict[str, dict[str, str]] = {}
    masks_by_orientation: dict[str, dict[str, object]] = {}
    best_index: dict[str, int] = {}
    area_by_orientation: dict[str, list[int]] = {}

    for orientation in ORIENTATIONS:
        z_indices = _slice_indices(orientation)

        channel_paths = {}
        for c, key in enumerate(C.MODALITY_KEYS):
            atlas = build_modality_atlas(disp_vols[c], orientation, z_indices)
            p = case_dir / f"{key}_{orientation}.png"
            Image.fromarray(atlas, mode="L").save(p, optimize=True)
            channel_paths[key] = str(p.relative_to(C.ROOT))
        channels_by_orientation[orientation] = channel_paths

        mask_atlas = build_mask_atlas(labels, orientation, z_indices)
        truth_path = case_dir / f"truth_{orientation}.png"
        Image.fromarray(mask_atlas, mode="RGBA").save(truth_path, optimize=True)
        masks_by_orientation[orientation] = {"truth": str(truth_path.relative_to(C.ROOT)), "prediction": None}

        profile = _profile(tumour, orientation)
        best_raw = int(profile.argmax())
        best_index[orientation] = int(np.argmin(np.abs(np.array(z_indices) - best_raw)))
        area_by_orientation[orientation] = [int(profile[i]) for i in z_indices]

    best_z = int(_profile(tumour, "axial").argmax())
    thumb_path = case_dir / "thumb.png"
    build_thumb(images, brain, labels, best_z).save(thumb_path, optimize=True)

    counts = np.bincount(labels.ravel(), minlength=4)

    return {
        "id": case_id,
        "label": case_id.upper(),
        "thumb": str(thumb_path.relative_to(C.ROOT)),
        "channels": channels_by_orientation,
        "masks": masks_by_orientation,
        "bestIndex": best_index,
        "maskAreaBySlice": area_by_orientation,
        "metrics": {
            "trueVolumeMm3": int(tumour.sum()),
            "sublabelsMm3": {
                "oedema": int(counts[1]),
                "nonEnhancingCore": int(counts[2]),
                "enhancingTumour": int(counts[3]),
            },
            "positiveSlices": int((tumour.sum(axis=(1, 2)) > 0).sum()),
            "predictedVolumeMm3": None,
            "dice": None,
            "hd95": None,
        },
    }


def _merge_predictions(entries: list[dict], arch: str) -> dict:
    """Read outputs/infer_<arch>.json + per-case .npz predictions, merge into `entries` in place.

    Additive by construction: every field this touches (`masks[o]["prediction"]`,
    `metrics.predictedVolumeMm3/dice/hd95`) already exists as `None` from
    `export_case` — nothing here changes the manifest's shape, only its values.
    """
    infer_path = C.OUTPUTS / f"infer_{arch}.json"
    if not infer_path.exists():
        raise FileNotFoundError(f"{infer_path} not found — run `python -m ml.infer --arch {arch}` first")
    infer = json.loads(infer_path.read_text())

    by_id = {e["id"]: e for e in entries}
    for cid, case_metrics in infer["per_case"].items():
        entry = by_id.get(cid)
        if entry is None:
            continue
        frag = export_prediction(cid, C.WEB_CASES, arch, case_metrics)
        for orientation, path in frag["masks_prediction"].items():
            entry["masks"][orientation]["prediction"] = path
        entry["metrics"]["predictedVolumeMm3"] = frag["predictedVolumeMm3"]
        entry["metrics"]["dice"] = frag["dice"]
        entry["metrics"]["hd95"] = frag["hd95"]

    return infer["summary"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ground-truth-only", action="store_true")
    mode.add_argument("--full", action="store_true", help="also merge in a trained model's predictions")
    ap.add_argument("--arch", choices=["25d", "3d"], default="3d", help="which model to merge with --full")
    ap.add_argument("--tier", default="standard", choices=["standard", "tiny"])
    args = ap.parse_args(argv)

    ids = C.case_ids(args.tier)
    C.WEB_CASES.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    entries = []
    for i, cid in enumerate(ids, 1):
        print(f"\r  exporting {i}/{len(ids)}  {cid}", end="", flush=True)
        entries.append(export_case(cid, C.WEB_CASES))
    print()

    model_summary = None
    if args.full:
        print(f"  merging {args.arch} predictions...")
        model_summary = _merge_predictions(entries, args.arch)

    manifest = {
        "groundTruthOnly": not args.full,
        "model": args.arch if args.full else None,
        "modelSummary": model_summary,
        "nCases": len(entries),
        "orientations": list(ORIENTATIONS),
        "atlas": {
            orientation: {
                "cols": C.ORIENTATION_COLS,
                "rows": C.ORIENTATION_ROWS[orientation],
                "tile": C.ATLAS_TILE,
                "z0": C.ORIENTATION_RANGE[orientation][0],
                "step": C.ORIENTATION_STEP[orientation],
                "n": C.ORIENTATION_N[orientation],
            }
            for orientation in ORIENTATIONS
        },
        "modalities": dict(zip(C.MODALITY_KEYS, C.MODALITIES)),
        "sublabels": {
            "oedema": "#2a78d6",
            "nonEnhancingCore": "#eb6834",
            "enhancingTumour": "#1baf7a",
        },
        "cases": entries,
    }
    manifest_path = C.WEB_CASES / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    size = sum(p.stat().st_size for p in C.WEB_CASES.rglob("*") if p.is_file())
    print(f"wrote {len(entries)} cases to {C.WEB_CASES}  ({size / 1e6:.1f} MB)  in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
