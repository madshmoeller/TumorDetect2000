"""One source of truth for paths, geometry and hyperparameters.

Every other module in `ml/` imports its constants from here. If a number appears
twice in this codebase, one of the two is a bug.
"""

from __future__ import annotations

import os
import pathlib

# ── paths ──────────────────────────────────────────────────────────────────

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "agentic-medical-ai-lab" / "data" / "cases"
IMAGES = DATA / "images"
LABELS = DATA / "labels"
TIER_STANDARD = DATA / "tier_standard.txt"
TIER_TINY = DATA / "tier_tiny.txt"

CACHE = pathlib.Path(os.environ.get("TUMORNET_CACHE", ROOT / "ml" / "cache"))
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "assets" / "figures"
WEB_CASES = ROOT / "assets" / "cases"

# ── dataset facts (asserted, never assumed — see data.audit_geometry) ───────

VOLUME_SHAPE = (240, 240, 155)  # x, y, z as stored in the NIfTI header
N_CHANNELS = 4
MODALITIES = ("FLAIR", "T1w", "T1-Gd", "T2w")
MODALITY_KEYS = ("flair", "t1", "t1gd", "t2")
SUBLABELS = {0: "background", 1: "oedema", 2: "non-enhancing core", 3: "enhancing tumour"}

#: The segmentation target. The dataset ships three tumour sub-labels; we
#: collapse them to a single foreground class, which is the BraTS "whole
#: tumour" region.
FOREGROUND = "label > 0"

VOXEL_VOLUME_MM3 = 1.0  # 1 mm isotropic, so voxel count == volume in mm^3

# ── crop ───────────────────────────────────────────────────────────────────
#
# Arrays are handled in (channel, z, y, x) order throughout — the NIfTI stores
# x fastest, so a straight reshape of the raw buffer to (t, z, y, x) is already
# in this layout and needs no transposes.
#
# The crop below is verified against the brain bounding box of all 60 cases by
# `python -m ml.data --audit`, which asserts per case that not one brain voxel
# is clipped. 192 is divisible by 32, which a depth-5 U-Net needs; the same crop
# is reused for the web atlases so ML space and browser space are the same space.
#
# Measured union bbox over all 60 cases: z[0,148] y[19,203] x[43,199]. An
# earlier 12-case sample suggested y[22,201] and z[0,143] — the audit caught it.
# z is left uncropped: brain reaches slice 148 in at least one case, and the
# in-plane crop is where the compute savings actually are.

CROP_Y = (16, 208)  # 192, covers measured y[19,203]
CROP_X = (24, 216)  # 192, covers measured x[43,199]
CROP_Z = (0, 155)  # full depth
CROP_HW = (CROP_Y[1] - CROP_Y[0], CROP_X[1] - CROP_X[0])
CROP_D = CROP_Z[1] - CROP_Z[0]

# ── web export ─────────────────────────────────────────────────────────────
#
# One PNG atlas per case per channel: every ATLAS_STEP'th axial slice, laid out
# left-to-right / top-to-bottom in an ATLAS_COLS x ATLAS_ROWS grid. The browser
# loads each atlas once and blits tiles out of it, so scrubbing costs no
# network requests.

ATLAS_Z = (6, 150)
ATLAS_STEP = 2
ATLAS_N = len(range(*ATLAS_Z, ATLAS_STEP))  # 72
ATLAS_COLS = 9
ATLAS_ROWS = (ATLAS_N + ATLAS_COLS - 1) // ATLAS_COLS  # 8
ATLAS_TILE = 192  # 1728 x 1536 px per atlas

#: Three viewing planes, all built from the same cropped (Z, Y, X) array.
#: axial slices perpendicular to Z are already square (CROP_HW = 192x192);
#: coronal (perpendicular to Y) and sagittal (perpendicular to X) slices are
#: (CROP_D, 192) = (155, 192) and get zero-padded up to the same 192x192 tile
#: so the frontend's tile math never has to special-case an orientation.
#:
#: Not verified against the NIfTI qform/sform — there's no reliable
#: patient-orientation metadata in this dataset to check against, so "head up,
#: right on the right" for coronal/sagittal is a plausible default, not a
#: confirmed one. Flagged here rather than asserted as correct (rule 4): this
#: is a viewer convenience, not a claim the segmentation model depends on.
ORIENTATION_AXIS = {"axial": 0, "coronal": 1, "sagittal": 2}  # index into (Z, Y, X)
ORIENTATION_RANGE = {
    "axial": ATLAS_Z,  # (6, 150) of 155
    "coronal": (6, 186),  # of 192
    "sagittal": (6, 186),  # of 192
}
ORIENTATION_STEP = {"axial": ATLAS_STEP, "coronal": 2, "sagittal": 2}
ORIENTATION_N = {
    name: len(range(*ORIENTATION_RANGE[name], ORIENTATION_STEP[name])) for name in ORIENTATION_AXIS
}  # axial 72, coronal 90, sagittal 90
ORIENTATION_COLS = 9
ORIENTATION_ROWS = {
    name: (n + ORIENTATION_COLS - 1) // ORIENTATION_COLS for name, n in ORIENTATION_N.items()
}  # 8, 10, 10

#: Display window for the greyscale channels, as percentiles of the brain
#: intensity distribution. Clipping at 99.5 keeps a single hot voxel from
#: crushing the rest of the image to grey.
WINDOW_PCT = (0.5, 99.5)

# ── model: 2.5D U-Net ────────────────────────────────────────────────────────
#
# Kept as a fast, cheap-to-train comparison point (figure F13) even though the
# 3D model below is now the primary architecture — see model_scope.md.

SLICE_CONTEXT = 2  # +/- this many neighbours -> 5-slice stack
IN_CHANNELS_25D = N_CHANNELS * (2 * SLICE_CONTEXT + 1)  # 20

BASE_WIDTH_25D = 24
DEPTH_25D = 5
BATCH_SIZE_25D = 16
EPOCHS_25D = 30
SLICES_PER_CASE_25D = 64  # sampled per case per epoch, not all 155 — see train.py

#: Fraction of sampled slices that are guaranteed to contain tumour. 46.1% of
#: all slices are positive (see eda.py), so this only mildly over-samples.
POS_FRACTION = 0.6

# ── model: 3D U-Net ──────────────────────────────────────────────────────────
#
# Primary architecture, viable now that training runs on a 24 GB RTX 3090 Ti
# rather than an M2 Max — see docs/model_scope.md for the reasoning. Trained on
# random patches, evaluated with sliding-window inference over the full crop.

IN_CHANNELS_3D = N_CHANNELS  # whole-volume 3D convs, no slice-stacking
BASE_WIDTH_3D = 20
DEPTH_3D = 4

#: z, y, x. Divisible by 16 (2^DEPTH_3D), comfortably inside the 155 x 192 x 192
#: crop. Not full-volume: nnU-Net-style patch training generalises better than
#: whole-volume 3D on a cohort this small, and it's what makes batch size > 1
#: affordable on 24 GB.
PATCH_3D = (96, 160, 160)
#: Both measured on this machine's 3090 Ti by `python -m ml.model --probe`
#: (see docs/model_scope.md): this workload is compute-bound, not
#: overhead-bound — ms/sample is ~flat from batch 3 to batch 10 (~88-95ms),
#: so batch size is chosen for memory headroom, not for speed.
#:
#: CAUTION — the probe's 15.3 GB peak is NOT the real training peak, and an
#: earlier version of this comment described it as "a safe memory margin
#: (15.3 GB peak at 8, vs 24 GB total)". Measured on a real run: the process
#: peaked at 23.02 GiB of 23.56 GiB usable, i.e. ~540 MB of headroom, not ~9 GB.
#: The probe has no dataloader, no pinned host memory and no augmentation, so it
#: understates the real run by ~7.7 GB. Confirmed the hard way: a 94 MB
#: allocation from a second process OOMed while training was in flight. Treat
#: batch 8 as very nearly the memory limit, and run nothing else on the GPU.
#: EPOCHS_3D x PATCHES_PER_CASE_3D x N_FOLDS x
#: 88ms projects to ~148 min total (2.5D + 3D) against the 170 min ceiling —
#: `python -m ml.train --dry-run` recomputes this from the live probe number
#: before any real run starts.
BATCH_SIZE_3D = 8
PATCHES_PER_CASE_3D = 6  # random patches sampled per case per epoch
EPOCHS_3D = 60  # early-stopped on validation Dice (patience above) — see train.py

# ── deep supervision ────────────────────────────────────────────────────────
#
# Auxiliary segmentation heads on the decoder's intermediate resolutions, their
# losses summed with the main head's. The one nnU-Net ingredient this codebase
# was missing — the residual encoder, InstanceNorm + LeakyReLU, Dice+BCE
# compound loss, patch training and AMP were already here.
#
# Decided 2026-08-17, BEFORE any 3D training run — at the time of the decision
# `outputs/checkpoints/` contained 2.5D fold checkpoints only, no `3d_fold*.pt`
# and no `train_history_3d.json`, so that claim is checkable rather than
# asserted. The 2.5D folds then on disk were trained WITHOUT deep supervision;
# they are superseded by a re-run (see below), because comparing a no-DS 2.5D
# model against a DS 3D model in figure F13 would conflate dimensionality with
# deep supervision — two changes, one comparison, which is the inadequate
# -baseline error this project is otherwise careful to avoid.
# Architecture was never pre-registered (RULES.md fixes the targets,
# the budget, the folds and the target region; not the layer count), so this is
# a design choice made in advance, not a target moved after seeing a result.
# The targets in TARGETS below are untouched.
#
# Cost is small: the heads are 1x1 convs and their losses run at reduced
# resolution. Measured, not assumed, by `python -m ml.model --probe`, which
# builds the model with deep supervision active so the reported ms/step is the
# real training step and not an understatement of it.
DEEP_SUPERVISION = True

#: One head per decoder stage, EXCLUDING the lowest-resolution stage (nnU-Net's
#: convention: at 1/16 resolution a 7285-voxel lesion is a handful of voxels and
#: the head mostly learns noise). Weights halve as resolution does, main head
#: first, normalised to sum to 1 — so turning deep supervision on does not also
#: silently scale the total loss and thus the effective learning rate.
DS_WEIGHT_DECAY_PER_LEVEL = 0.5

#: Targets are matched to each head's resolution by ADAPTIVE AVERAGE POOLING,
#: not nearest-neighbour subsampling. `soft_dice_loss` never assumes a binary
#: target and `binary_cross_entropy_with_logits` accepts one, so the soft
#: fractional targets average pooling produces keep both loss terms valid.
#:
#: STATUS: the default below is NOT settled, and the two justifications this
#: comment previously carried were both wrong. Written out in full because the
#: errors are more instructive than the conclusion, and because the 3D run of
#: 2026-08-17 17:00 was produced with "avg" and that must stay traceable.
#:
#: Measured on all 60 cached label volumes (rule 2). Counting targets that
#: become all-background despite the source containing tumour — this column is
#: sound and does not depend on any rescaling:
#:
#:            avg emptied   nearest emptied
#:   3D  1/2    0 /  200        0 /  200
#:   3D  1/4    0 /  200        0 /  200
#:   25d 1/2    0 / 4290       35 / 4290
#:   25d 1/4    0 / 4290       94 / 4290
#:   25d 1/8    0 / 4290      229 / 4290   (true areas: median 13 px, max 163)
#:
#: WRONG JUSTIFICATION #1: that nearest-neighbour would delete the cohort's
#: smallest lesion (7285 voxels, case_056) at 1/4 or 1/8. It does not. 7285
#: voxels is a compact blob ~19 voxels across, still ~125 voxels after 4x
#: downsampling. Volume burden was the wrong statistic; per-slice *area* is what
#: gets erased, which is why only the 2D model shows any effect at all.
#:
#: WRONG JUSTIFICATION #2: that avg pooling is "mass-preserving by construction
#: (exactly 1.0000 at every head)". That measurement multiplied the pooled sum
#: back up by factor**3 before dividing, so it was 1.0000 by construction and
#: described nothing the loss ever sees. `adaptive_avg_pool` preserves the MEAN
#: (occupancy fraction), not the sum; the sum scales by 1/factor**ndim.
#: Measured on a 100-voxel lesion: target.sum() = 100.0 / 12.5 / 1.56 / 0.195 at
#: full / 1/2 / 1/4 / 1/8.
#:
#: WHAT THE TESTS ACTUALLY SHOW — and it points the other way. Soft Dice is not
#: minimised at prob == target when the target is soft: with p == t exactly,
#: dice = sum(t**2)/sum(t) < 1 for any non-binary t. Measured loss for a
#: *perfect* soft prediction: 0.0615 (full), 0.2398 (1/2), 0.4814 (1/4), 0.8753
#: (1/8). Its stationary condition requires t_i == dice/2 for every i, which is
#: unreachable for a non-constant target, so the optimum sits on the boundary:
#: the aux heads are driven toward a *thresholded, dilated* binary mask rather
#: than toward the occupancy fraction they were handed. Binary targets make the
#: same loss well-posed (sum(t**2) == sum(t), optimum exactly at p == t), which
#: is why nnU-Net downsamples hard labels rather than averaging them.
#:
#: So "nearest" is arguably the correct choice, and for the 3D model it is
#: strictly better on the evidence available: binary (well-posed) targets AND
#: nothing emptied at either of its 1/2 and 1/4 heads. Only 2.5D's 1/8 head has
#: anything to lose, and 2.5D is the comparison arm, not the deliverable.
#:
#: NOT changed to "nearest" here, deliberately. The in-flight 3D 5-fold run used
#: "avg", and silently flipping the constant would mean the code on disk is not
#: the code that produced the reported numbers. This gets settled by running the
#: ablation the knob exists for — one fold, ~28 min — and reporting both, not by
#: an edit made while a run is in progress.
#: Overridable by env (`TUMORNET_DS_POOLING=nearest`) so the ablation can be run
#: as a separate process without editing this file mid-experiment — an edit
#: during a run would mean the code on disk is not the code that produced the
#: artifacts, which is the thing this project most needs to stay true.
DS_TARGET_POOLING = os.environ.get("TUMORNET_DS_POOLING", "avg")

# ── training (shared) ───────────────────────────────────────────────────────

N_FOLDS = 5
FOLD_SEED = 20260817
#: 3e-3 (a common OneCycle/AdamW max_lr) produced near-total training
#: instability under fp16 autocast on this architecture: from the LR-peak
#: region onward, almost every batch's forward pass produced a non-finite
#: loss (192/192 in some epochs) even with gradient clipping in place —
#: clipping bounds a *finite* gradient, it does not prevent a peak-LR update
#: from pushing weights into a regime whose activations overflow fp16 on the
#: next forward pass. Measured, not assumed: see train.py's isfinite-loss
#: skip-and-log, which is what surfaced this in the first place.
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 20  # epochs without val-Dice improvement

#: Patience does not start counting until this fraction of the schedule has
#: elapsed. OneCycleLR peaks at pct_start (0.3 by default) and only anneals
#: afterwards, and the anneal is where most of the final quality comes from — so
#: a fold stopped before it has thrown away the part that matters.
#:
#: This is a fix, not a tuning knob. In the pre-registered run of 2026-08-17,
#: 3 of 5 folds selected their checkpoint at or before the epoch-18 LR peak
#: (epochs 10, 13, 18) and two of them were terminated by patience before
#: annealing could overturn what turned out to be a warmup fluke on a
#: re-drawn-every-epoch validation set. Both causes are fixed together: this,
#: and `frozen=True` on the validation datasets. See
#: outputs/prereg_20260817/README.md for what the unfixed run produced.
EARLY_STOP_MIN_FRACTION = 0.5

#: Also write `<arch>_fold<k>_final.pt` at the end of every fold, alongside the
#: best-by-validation checkpoint. Costs one torch.save per fold. Exists because
#: the pre-registered run saved only best-by-proxy weights, which made the
#: question "would the annealed weights have been better?" impossible to answer
#: after the fact — it needed a full re-run rather than a comparison.
SAVE_FINAL_CHECKPOINT = True

#: TTA at inference: left-right flip, averaged with the unflipped prediction —
#: legitimate even under strict out-of-fold evaluation, since it only reuses
#: the *same* single model that never saw this case, not any other fold's.
#: Ablated (with vs without) in figure F13, not just applied silently.
#:
#: Deliberately NOT averaging in the other N_FOLDS-1 models' predictions for a
#: held-out case: 4 of those 5 models trained on that exact case, so an
#: "ensemble" Dice computed that way would be leaked, not just optimistic.
#: The 5-fold ensemble is a real artifact (worth shipping for a genuinely new
#: scan) but it has no honest Dice number to report against these 60 cases —
#: see docs/model_scope.md.
TTA_FLIP = True

#: Wall-clock ceilings. MACBOOK_* is what the original plan sized against on
#: an M2 Max; TRAIN_BUDGET_MINUTES is the real ceiling for the runs this
#: project actually executes, now that an RTX 3090 Ti turned out to be this
#: machine (measured via `nvidia-smi`, not assumed). Both numbers are kept so
#: the writeup can state which one the reported runtime was measured against.
MACBOOK_BUDGET_MINUTES = 60
MACBOOK_TRAIN_BUDGET_MINUTES = 45
TRAIN_BUDGET_MINUTES = 170  # 3h ceiling given, minus margin

# ── evaluation ───────────────────────────────────────────────────────────────

#: Pre-registered in RULES.md before any training, revised once (same day,
#: still pre-training) when the 3090 Ti became available. Do not edit after
#: seeing a result — see rule 4.
TARGETS = {
    "mean_dice_floor": 0.85,
    "mean_dice_stretch": 0.90,
    "median_dice_floor": 0.87,
    "median_dice_stretch": 0.90,
    "worst_dice_floor": 0.55,  # unchanged by the compute increase — see RULES.md
    "wilcoxon_p": 0.01,
}

# ── augmentation ─────────────────────────────────────────────────────────────
#
# n = 60 (48 per training fold) is small enough that augmentation is not
# optional polish — it is most of what stands between the model and
# memorising 48 brains. Ranges are deliberately mild: this is MRI, not natural
# images, and a 45-degree rotation or a 2x intensity swing would manufacture
# anatomy that does not occur in vivo.

AUG_FLIP_PROB = 0.5  # independent per axis
AUG_ROTATE_DEG_2D = 15.0  # in-plane, 2.5D model
AUG_ROTATE_DEG_3D = 10.0  # about the z (axial) axis only, 3D model
AUG_SCALE_JITTER = 0.10  # +/- fraction
AUG_INTENSITY_SCALE = 0.15  # +/- fraction, contrast-like jitter in z-score space
AUG_INTENSITY_SHIFT = 0.10  # +/- z-score units, brightness-like jitter
AUG_NOISE_SIGMA = 0.08  # gaussian noise std, z-score units
AUG_BIAS_FIELD_MAGNITUDE = 0.15  # +/- fractional multiplicative field
AUG_BIAS_FIELD_RES = 4  # control-point grid side length before upsampling

DEFAULT_THRESHOLD = 0.5
MIN_COMPONENT_VOXELS = 500

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 7


def case_ids(tier: str = "standard") -> list[str]:
    """Read a cohort file, ignoring comments and blanks."""
    path = TIER_STANDARD if tier == "standard" else TIER_TINY
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def fold_assignment(ids: list[str], n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> dict[str, int]:
    """Deterministic case -> fold index, shared by every model and every baseline.

    Baselines, the 2.5D model and the 3D model are only comparable in figure
    F09 if all three are scored on identical held-out sets — so fold
    membership is computed once, here, and imported everywhere else rather
    than re-derived per script.
    """
    import random

    shuffled = sorted(ids)  # sort first so the shuffle is reproducible regardless of input order
    random.Random(seed).shuffle(shuffled)
    return {cid: i % n_folds for i, cid in enumerate(shuffled)}
