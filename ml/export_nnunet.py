"""Convert this cohort into nnU-Net v2 raw format, for the reference-arm experiment.

The nnU-Net run is NOT the pre-registered deliverable and cannot be: nnU-Net's
epoch is a fixed 250 iterations regardless of dataset size, so its default
1000-epoch schedule is ~24 h per fold on this machine's 3090 Ti against a
pre-registered 170 min total training budget. It is run as an explicitly
labelled external reference on a single fold, to answer one question — is the
hand-rolled residual-encoder U-Net in `ml/model.py` leaving anything on the
table? — without spending the headline number on it.

Two things this script exists to get right, because both are easy to get
silently wrong:

1. **Channel splitting.** Our images are one 4D NIfTI per case, (240, 240, 155,
   4). nnU-Net wants one 3D file per modality, suffixed `_0000` .. `_0003`.

2. **Fold identity.** nnU-Net generates its own 5-fold split by default. Left
   alone it would score fold 0 on a *different* 12 cases than our baselines and
   our own models did, and figure F09's three-way comparison would silently
   stop comparing like with like. `splits` below writes nnU-Net's
   `splits_final.json` from `config.fold_assignment(seed=20260817)` — the same
   single source of truth every other model in this project uses.

    python -m ml.export_nnunet raw      # write nnUNet_raw/DatasetXXX/...
    python -m ml.export_nnunet splits   # write splits_final.json (AFTER preprocessing)

Order matters. `nnUNetv2_plan_and_preprocess` creates the preprocessed directory,
and nnU-Net writes its own `splits_final.json` at training time only if one is
not already there — so `splits` must run after preprocessing and before training.

Deliberately does NOT pip-install nnunetv2. Doing so in the anaconda base
environment could pull a different torch than the 2.4.0+cu121 the pre-registered
results are produced with, which would compromise the very run it is meant to be
a reference for. Install it into a dedicated venv instead.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

import nibabel as nib
import numpy as np

from . import config as C

#: 500-series to stay clear of nnU-Net's own published dataset IDs.
DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID:03d}_TumorWT"

#: Sibling of the repo rather than inside it: nnU-Net writes preprocessed copies
#: of the whole cohort and we do not want multi-GB derived data landing in the
#: project tree or in the web export.
NNUNET_ROOT = C.ROOT.parent / "nnunet_tumordetect"
RAW = NNUNET_ROOT / "nnUNet_raw" / DATASET_NAME
PREPROCESSED = NNUNET_ROOT / "nnUNet_preprocessed" / DATASET_NAME
RESULTS = NNUNET_ROOT / "nnUNet_results"


def write_raw(*, overwrite: bool = False) -> None:
    """Split each 4D image into 4 single-modality files and binarise the label."""
    images_out, labels_out = RAW / "imagesTr", RAW / "labelsTr"
    if RAW.exists() and not overwrite:
        raise SystemExit(f"{RAW} exists; pass --overwrite to replace it")
    if RAW.exists():
        shutil.rmtree(RAW)
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    ids = C.case_ids("standard")
    for i, cid in enumerate(ids, 1):
        img = nib.load(str(C.IMAGES / f"{cid}.nii.gz"))
        data = np.asanyarray(img.dataobj)
        if data.ndim != 4 or data.shape[3] != C.N_CHANNELS:
            raise SystemExit(f"{cid}: expected (x,y,z,{C.N_CHANNELS}), got {data.shape}")

        # Affine is carried through unchanged. nnU-Net reads spacing from the
        # header to build its plan, so a dropped or altered affine here would
        # silently change the geometry it resamples to.
        for ch in range(C.N_CHANNELS):
            nib.save(
                nib.Nifti1Image(np.ascontiguousarray(data[..., ch]), img.affine, img.header),
                str(images_out / f"{cid}_{ch:04d}.nii.gz"),
            )

        lab = nib.load(str(C.LABELS / f"{cid}.nii.gz"))
        # C.FOREGROUND is "label > 0": the three shipped sub-labels collapsed to
        # the single BraTS whole-tumour class, matching what every other model
        # and baseline in this project is scored against.
        binary = (np.asanyarray(lab.dataobj) > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(binary, lab.affine, lab.header), str(labels_out / f"{cid}.nii.gz"))

        if i % 10 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)} cases written")

    (RAW / "dataset.json").write_text(json.dumps({
        "channel_names": {str(i): m for i, m in enumerate(C.MODALITIES)},
        "labels": {"background": 0, "tumour": 1},
        "numTraining": len(ids),
        "file_ending": ".nii.gz",
        "description": (
            "Whole-tumour (label > 0) binary target, same cohort, same collapse "
            "and same fold assignment as ml/. Reference arm only — not the "
            "pre-registered deliverable; see docs/architecture.md."
        ),
    }, indent=2))
    print(f"\nwrote {RAW}")
    print(f"  imagesTr: {len(ids) * C.N_CHANNELS} files, labelsTr: {len(ids)} files")


def write_splits() -> None:
    """Force nnU-Net onto this project's fold assignment instead of its own."""
    if not PREPROCESSED.exists():
        raise SystemExit(
            f"{PREPROCESSED} does not exist yet — run nnUNetv2_plan_and_preprocess "
            "first, then this command, then training."
        )
    ids = C.case_ids("standard")
    folds = C.fold_assignment(ids)

    splits = []
    for f in range(C.N_FOLDS):
        train = sorted(c for c in ids if folds[c] != f)
        val = sorted(c for c in ids if folds[c] == f)
        assert not set(train) & set(val), f"fold {f}: train/val overlap"
        assert len(train) + len(val) == len(ids)
        splits.append({"train": train, "val": val})

    # Every case validated exactly once across the five folds — the same
    # out-of-fold guarantee ml/train.py relies on.
    seen = [c for s in splits for c in s["val"]]
    assert sorted(seen) == sorted(ids), "each case must be validated exactly once"

    out = PREPROCESSED / "splits_final.json"
    out.write_text(json.dumps(splits, indent=2))
    print(f"wrote {out}")
    for f, s in enumerate(splits):
        print(f"  fold {f}: {len(s['train'])} train / {len(s['val'])} val   val={s['val']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["raw", "splits", "env"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    if args.stage == "raw":
        write_raw(overwrite=args.overwrite)
    elif args.stage == "splits":
        write_splits()
    else:
        print(f"export nnUNet_raw={NNUNET_ROOT / 'nnUNet_raw'}")
        print(f"export nnUNet_preprocessed={NNUNET_ROOT / 'nnUNet_preprocessed'}")
        print(f"export nnUNet_results={RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
