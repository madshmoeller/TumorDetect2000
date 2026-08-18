"""Loading, normalisation, cropping and caching of the 60-case cohort.

Array convention, used everywhere downstream:

    images  (C, Z, Y, X)  float32/float16, C = 4 modalities
    labels  (Z, Y, X)     uint8, values 0..3
    brain   (Z, Y, X)     bool

The NIfTI files store x fastest and channel slowest, so `array.T` maps the
on-disk (x, y, z, t) directly onto (t, z, y, x) with no further reordering.

Run `python -m ml.data --audit` to verify the dataset facts this project relies
on, and `python -m ml.data --build` to write the preprocessing cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import config as C

# ── loading ────────────────────────────────────────────────────────────────


def load_raw(case_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (images, labels) for one case, unnormalised, uncropped.

    images: (4, 155, 240, 240) float32
    labels: (155, 240, 240) uint8
    """
    import nibabel as nib

    img = np.asanyarray(nib.load(C.IMAGES / f"{case_id}.nii.gz").dataobj)
    lab = np.asanyarray(nib.load(C.LABELS / f"{case_id}.nii.gz").dataobj)

    if img.ndim != 4 or img.shape[:3] != C.VOLUME_SHAPE or img.shape[3] != C.N_CHANNELS:
        raise ValueError(f"{case_id}: unexpected image shape {img.shape}")
    if lab.shape != C.VOLUME_SHAPE:
        raise ValueError(f"{case_id}: unexpected label shape {lab.shape}")

    return (
        np.ascontiguousarray(img.T, dtype=np.float32),
        np.ascontiguousarray(lab.T, dtype=np.uint8),
    )


def brain_mask(images: np.ndarray) -> np.ndarray:
    """Skull-stripped volumes carry exact zeros outside the head.

    Any voxel non-zero in any modality is brain. This is the mask everything is
    normalised within and evaluated within — background air is free
    true-negatives and would flatter any accuracy-like metric.
    """
    return np.any(images != 0, axis=0)


def normalise(images: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Per-case, per-channel z-score computed inside the brain mask.

    This is not optional. Brain-mean FLAIR ranges from ~130 (case_056) to ~685
    (case_001) across this cohort — a 5x spread with no clinical meaning. A
    model fed raw intensities learns which case it is looking at. Figure F02
    shows the before/after.

    Voxels outside the brain are set to 0, which is the post-normalisation mean
    and so carries no signal.
    """
    out = np.zeros_like(images, dtype=np.float32)
    for c in range(images.shape[0]):
        vals = images[c][brain]
        mu = float(vals.mean())
        sd = float(vals.std())
        if sd < 1e-6:  # a dead channel would otherwise produce inf
            raise ValueError(f"channel {c} has near-zero variance inside the brain")
        out[c][brain] = (vals - mu) / sd
    return out


def crop(arr: np.ndarray) -> np.ndarray:
    """Apply the canonical crop to the last three axes."""
    zs, ze = C.CROP_Z
    ys, ye = C.CROP_Y
    xs, xe = C.CROP_X
    return arr[..., zs:ze, ys:ye, xs:xe]


# ── cache ──────────────────────────────────────────────────────────────────


def cache_paths(case_id: str) -> dict[str, "object"]:
    return {
        "img": C.CACHE / f"{case_id}_img.npy",
        "lab": C.CACHE / f"{case_id}_lab.npy",
        "brain": C.CACHE / f"{case_id}_brain.npy",
    }


def is_cached(case_id: str) -> bool:
    return all(p.exists() for p in cache_paths(case_id).values())


def build_cache(case_id: str, *, force: bool = False) -> None:
    """Preprocess one case to disk: normalised, cropped, float16."""
    paths = cache_paths(case_id)
    if not force and is_cached(case_id):
        return

    images, labels = load_raw(case_id)
    brain = brain_mask(images)
    _assert_crop_contains(case_id, brain)

    images = normalise(images, brain)

    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.save(paths["img"], crop(images).astype(np.float16))
    np.save(paths["lab"], crop(labels))
    np.save(paths["brain"], crop(brain))


def load_cached(case_id: str, *, mmap: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (images float16, labels uint8, brain bool), cropped and normalised."""
    if not is_cached(case_id):
        build_cache(case_id)
    mode = "r" if mmap else None
    p = cache_paths(case_id)
    return (
        np.load(p["img"], mmap_mode=mode),
        np.load(p["lab"], mmap_mode=mode),
        np.load(p["brain"], mmap_mode=mode),
    )


def _assert_crop_contains(case_id: str, brain: np.ndarray) -> None:
    """The crop must not clip a single brain voxel. Fail loudly if it does."""
    zs, ze = C.CROP_Z
    ys, ye = C.CROP_Y
    xs, xe = C.CROP_X
    outside = brain.sum() - brain[zs:ze, ys:ye, xs:xe].sum()
    if outside:
        z = np.where(brain.any(axis=(1, 2)))[0]
        y = np.where(brain.any(axis=(0, 2)))[0]
        x = np.where(brain.any(axis=(0, 1)))[0]
        raise ValueError(
            f"{case_id}: crop clips {outside} brain voxels. "
            f"bbox z[{z.min()},{z.max()}] y[{y.min()},{y.max()}] x[{x.min()},{x.max()}] "
            f"vs crop z[{zs},{ze}) y[{ys},{ye}) x[{xs},{xe})"
        )


def uncrop(arr: np.ndarray, fill: float = 0) -> np.ndarray:
    """Inverse of `crop` — place a cropped volume back into full 240x240x155 space.

    Used so predictions can be compared against, and written alongside, the
    original label volumes without the crop silently changing the geometry.
    """
    zs, ze = C.CROP_Z
    ys, ye = C.CROP_Y
    xs, xe = C.CROP_X
    full = np.full(arr.shape[:-3] + C.VOLUME_SHAPE[::-1], fill, dtype=arr.dtype)
    full[..., zs:ze, ys:ye, xs:xe] = arr
    return full


# ── display windowing (shared by figures and the web export) ───────────────


def window_to_uint8(channel: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Map one normalised channel to 0..255 for display.

    The window is the 0.5th-99.5th percentile of the *brain* voxels, so a single
    hot voxel cannot crush the rest of the image to flat grey. Because z-scoring
    is a monotone affine map per case per channel, windowing the normalised data
    by percentile gives pixel-identical output to windowing the raw data — which
    is why the cache serves both the model and the browser.
    """
    vals = channel[brain]
    lo, hi = np.percentile(vals, C.WINDOW_PCT)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    out = np.clip((channel.astype(np.float32) - lo) / (hi - lo), 0, 1)
    out[~brain] = 0
    return (out * 255).astype(np.uint8)


# ── audit ──────────────────────────────────────────────────────────────────


def audit_geometry(case_ids: list[str], *, verbose: bool = True) -> dict:
    """Check every dataset fact this project depends on, on every case.

    Cheap enough (~2 min for 60 cases) to be worth running whenever the data
    might have changed.
    """
    import nibabel as nib

    shapes, dtypes, zooms = set(), set(), set()
    bboxes = []
    stats = []

    for i, cid in enumerate(case_ids, 1):
        if verbose:
            print(f"\r  auditing {i}/{len(case_ids)}  {cid}", end="", flush=True)
        im = nib.load(C.IMAGES / f"{cid}.nii.gz")
        lb = nib.load(C.LABELS / f"{cid}.nii.gz")
        shapes.add((im.shape, lb.shape))
        dtypes.add((im.get_data_dtype().str, lb.get_data_dtype().str))
        zooms.add(tuple(np.round(im.header.get_zooms()[:3], 4)))

        images, labels = load_raw(cid)
        brain = brain_mask(images)
        tumour = labels > 0

        bz = np.where(brain.any(axis=(1, 2)))[0]
        by = np.where(brain.any(axis=(0, 2)))[0]
        bx = np.where(brain.any(axis=(0, 1)))[0]
        bboxes.append((bz.min(), bz.max(), by.min(), by.max(), bx.min(), bx.max()))

        counts = np.bincount(labels.ravel(), minlength=4)
        per_slice = tumour.sum(axis=(1, 2))
        brain_vals = {}
        for c in range(C.N_CHANNELS):
            brain_vals[C.MODALITIES[c]] = float(images[c][brain].mean())

        stats.append(
            {
                "case": cid,
                "brain_voxels": int(brain.sum()),
                "tumour_voxels": int(tumour.sum()),
                "label_counts": counts.tolist(),
                "positive_slices": int((per_slice > 0).sum()),
                "tumour_z": [int(np.where(per_slice > 0)[0].min()), int(np.where(per_slice > 0)[0].max())],
                "best_slice": int(per_slice.argmax()),
                "brain_mean": brain_vals,
            }
        )

    if verbose:
        print()

    bb = np.array(bboxes)
    union = dict(
        z=[int(bb[:, 0].min()), int(bb[:, 1].max())],
        y=[int(bb[:, 2].min()), int(bb[:, 3].max())],
        x=[int(bb[:, 4].min()), int(bb[:, 5].max())],
    )

    tv = np.array([s["tumour_voxels"] for s in stats])
    bv = np.array([s["brain_voxels"] for s in stats])
    total = int(np.prod(C.VOLUME_SHAPE))

    report = {
        "n_cases": len(case_ids),
        "unique_shapes": sorted(str(s) for s in shapes),
        "unique_dtypes": sorted(str(d) for d in dtypes),
        "unique_zooms": sorted(str(z) for z in zooms),
        "brain_bbox_union": union,
        "crop": {"z": list(C.CROP_Z), "y": list(C.CROP_Y), "x": list(C.CROP_X)},
        "crop_contains_all_brain": (
            union["z"][0] >= C.CROP_Z[0]
            and union["z"][1] < C.CROP_Z[1]
            and union["y"][0] >= C.CROP_Y[0]
            and union["y"][1] < C.CROP_Y[1]
            and union["x"][0] >= C.CROP_X[0]
            and union["x"][1] < C.CROP_X[1]
        ),
        "brain_fraction_mean": float((bv / total).mean()),
        "tumour_voxels": {
            "min": int(tv.min()),
            "p25": int(np.percentile(tv, 25)),
            "median": int(np.median(tv)),
            "p75": int(np.percentile(tv, 75)),
            "max": int(tv.max()),
        },
        "tumour_fraction_of_volume_median": float(np.median(tv / total)),
        "tumour_fraction_of_brain_median": float(np.median(tv / bv)),
        "positive_slices": int(sum(s["positive_slices"] for s in stats)),
        "total_slices": len(case_ids) * C.VOLUME_SHAPE[2],
        "cases_missing_a_sublabel": {
            C.SUBLABELS[k]: [s["case"] for s in stats if s["label_counts"][k] == 0] for k in (1, 2, 3)
        },
        "per_case": stats,
    }
    return report


EXPECTED = {
    # Asserted by --audit. These are the numbers quoted in the plan and README;
    # if the loader ever changes shape or orientation, this catches it.
    "n_cases": 60,
    "tumour_voxels_median": 108705,
    "tumour_voxels_min": 7285,
    "tumour_voxels_max": 256875,
    "positive_slices": 4290,
    "total_slices": 9300,
    "brain_fraction_mean": 0.15969,
    "crop_contains_all_brain": True,
}


def _verify(report: dict) -> int:
    checks = [
        ("n_cases", report["n_cases"], EXPECTED["n_cases"], 0),
        ("tumour voxels median", report["tumour_voxels"]["median"], EXPECTED["tumour_voxels_median"], 0),
        ("tumour voxels min", report["tumour_voxels"]["min"], EXPECTED["tumour_voxels_min"], 0),
        ("tumour voxels max", report["tumour_voxels"]["max"], EXPECTED["tumour_voxels_max"], 0),
        ("positive slices", report["positive_slices"], EXPECTED["positive_slices"], 0),
        ("total slices", report["total_slices"], EXPECTED["total_slices"], 0),
        ("brain fraction", round(report["brain_fraction_mean"], 5), EXPECTED["brain_fraction_mean"], 1e-5),
    ]
    failed = 0
    for name, got, want, tol in checks:
        ok = abs(got - want) <= tol
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got} (expected {want})")
        failed += not ok
    ok = report["crop_contains_all_brain"]
    print(f"  {'PASS' if ok else 'FAIL'}  crop contains all brain voxels in all cases")
    failed += not ok
    return failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", action="store_true", help="verify dataset facts on every case")
    ap.add_argument("--build", action="store_true", help="write the preprocessing cache")
    ap.add_argument("--force", action="store_true", help="rebuild cache entries that already exist")
    ap.add_argument("--tier", default="standard", choices=["standard", "tiny"])
    args = ap.parse_args(argv)

    ids = C.case_ids(args.tier)
    print(f"cohort: {args.tier}, {len(ids)} cases")

    if args.audit:
        t0 = time.time()
        report = audit_geometry(ids)
        C.OUTPUTS.mkdir(parents=True, exist_ok=True)
        (C.OUTPUTS / "audit.json").write_text(json.dumps(report, indent=2))
        print(f"\ngeometry:  shapes={report['unique_shapes']}")
        print(f"           dtypes={report['unique_dtypes']}  zooms={report['unique_zooms']}")
        bb = report["brain_bbox_union"]
        print(f"brain bbox union: z{bb['z']} y{bb['y']} x{bb['x']}")
        print(f"crop:             z{report['crop']['z']} y{report['crop']['y']} x{report['crop']['x']}")
        missing = {k: v for k, v in report["cases_missing_a_sublabel"].items() if v}
        print(f"cases missing a sub-label: {missing or 'none'}")
        print("\nverification:")
        failed = _verify(report)
        print(f"\nwrote {C.OUTPUTS / 'audit.json'}  ({time.time() - t0:.0f}s)")
        if failed:
            print(f"\n{failed} check(s) FAILED")
            return 1

    if args.build:
        t0 = time.time()
        for i, cid in enumerate(ids, 1):
            print(f"\r  caching {i}/{len(ids)}  {cid}", end="", flush=True)
            build_cache(cid, force=args.force)
        size = sum(p.stat().st_size for p in C.CACHE.glob("*.npy"))
        print(f"\n  cache: {C.CACHE}  {size / 1e9:.2f} GB  ({time.time() - t0:.0f}s)")

    if not (args.audit or args.build):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
