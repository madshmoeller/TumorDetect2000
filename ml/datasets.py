"""PyTorch datasets for the 2.5D and 3D models.

Both sample *from the cached, cropped, per-case-normalised arrays* written by
`ml.data.build_cache` — never from the raw NIfTI files — so training reads are
a `np.load(mmap_mode='r')` slice, not a decompress-and-normalise on every
`__getitem__`.

Both oversample tumour-containing locations (`POS_FRACTION`): tumour is
present in only ~46% of axial slices and ~7.5% of brain voxels (see F04/F05),
so uniform random sampling would spend most of an epoch on easy background.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C
from . import data as D


class SliceDataset25D(Dataset):
    """One sample = a 5-slice neighbourhood, all 4 modalities, at one axial index.

    Returns (20, H, W) float32 image stack and (1, H, W) float32 binary mask
    for the *centre* slice only — the neighbours are context, not a
    prediction target, matching how `infer.py` reassembles a full volume by
    sliding the centre index across every slice exactly once.
    """

    def __init__(self, case_ids: list[str], *, samples_per_case: int = C.SLICES_PER_CASE_25D, seed: int = 0,
                 frozen: bool = False):
        self.case_ids = case_ids
        self.samples_per_case = samples_per_case
        self.rng = np.random.RandomState(seed)
        #: See PatchDataset3D's note — `frozen` draws every sample location once
        #: at construction so the set is identical on every epoch. Required for
        #: the validation loader, whose whole job is to be a FIXED yardstick.
        self.frozen = frozen
        self._frozen_slices: list[int] | None = None

        # Pre-index which slices contain tumour, per case, so sampling a
        # positive slice is an O(1) choice rather than a rejection loop.
        self._pos_idx: dict[str, np.ndarray] = {}
        self._all_idx: dict[str, np.ndarray] = {}
        for cid in case_ids:
            _, labels, _ = D.load_cached(cid)
            per_slice = (labels > 0).sum(axis=(1, 2))
            depth = labels.shape[0]
            lo, hi = C.SLICE_CONTEXT, depth - C.SLICE_CONTEXT  # keep the neighbourhood in-bounds
            self._pos_idx[cid] = np.where(per_slice[lo:hi] > 0)[0] + lo
            self._all_idx[cid] = np.arange(lo, hi)

        if frozen:
            self._frozen_slices = [self._pick_slice(self.case_ids[i % len(self.case_ids)])
                                   for i in range(len(self))]

    def __len__(self) -> int:
        return len(self.case_ids) * self.samples_per_case

    def _pick_slice(self, cid: str) -> int:
        if self.rng.rand() < C.POS_FRACTION and len(self._pos_idx[cid]):
            return int(self.rng.choice(self._pos_idx[cid]))
        return int(self.rng.choice(self._all_idx[cid]))

    def __getitem__(self, idx: int):
        cid = self.case_ids[idx % len(self.case_ids)]
        images, labels, _ = D.load_cached(cid)
        z = self._frozen_slices[idx] if self.frozen else self._pick_slice(cid)
        lo, hi = z - C.SLICE_CONTEXT, z + C.SLICE_CONTEXT + 1

        stack = np.asarray(images[:, lo:hi]).astype(np.float32)  # (4, 5, H, W)
        stack = stack.reshape(-1, *stack.shape[2:])  # (20, H, W)
        mask = (np.asarray(labels[z]) > 0).astype(np.float32)[None]  # (1, H, W)

        return torch.from_numpy(stack), torch.from_numpy(mask)


class PatchDataset3D(Dataset):
    """One sample = a random (Z,Y,X) patch, all 4 modalities, plus its mask patch.

    `POS_FRACTION` of patches are centred (with jitter) on a randomly chosen
    tumour voxel; the rest are uniformly placed within the valid crop region.
    """

    def __init__(self, case_ids: list[str], *, patches_per_case: int = C.PATCHES_PER_CASE_3D, seed: int = 0,
                 frozen: bool = False):
        self.case_ids = case_ids
        self.patches_per_case = patches_per_case
        self.rng = np.random.RandomState(seed)
        self.patch = C.PATCH_3D

        #: `frozen` materialises every patch origin once, here, instead of
        #: drawing one per __getitem__ call. This is load-bearing for validation.
        #:
        #: Unfrozen, `__getitem__` calls `_pick_origin`, which advances self.rng
        #: on every call — so iterating the val loader yields a DIFFERENT patch
        #: from the same index on every epoch. Verified: the same idx returned
        #: foreground counts of 66551 / 73207 / 73590 / 62677 across four calls.
        #: "Best epoch" was therefore chosen against a target that moved every
        #: epoch, and a lucky easy draw during LR warmup registers as a peak for
        #: reasons unrelated to the model. In the 2026-08-17 pre-registered run
        #: that put 3 of 5 folds' selected checkpoints at or before the
        #: OneCycleLR peak (epochs 10, 13, 18), i.e. with no anneal applied.
        #: See outputs/prereg_20260817/README.md.
        self.frozen = frozen
        self._origins: list[tuple[int, int, int]] | None = None

        self._tumour_voxels: dict[str, np.ndarray] = {}
        self._volume_shape = None
        for cid in case_ids:
            _, labels, _ = D.load_cached(cid)
            self._volume_shape = labels.shape
            self._tumour_voxels[cid] = np.argwhere(np.asarray(labels) > 0)

        if frozen:
            self._origins = [self._pick_origin(self.case_ids[i % len(self.case_ids)])
                             for i in range(len(self))]

    def __len__(self) -> int:
        return len(self.case_ids) * self.patches_per_case

    def _pick_origin(self, cid: str) -> tuple[int, int, int]:
        pz, py, px = self.patch
        dz, dy, dx = self._volume_shape
        max_z, max_y, max_x = dz - pz, dy - py, dx - px

        if self.rng.rand() < C.POS_FRACTION and len(self._tumour_voxels[cid]):
            cz, cy, cx = self._tumour_voxels[cid][self.rng.randint(len(self._tumour_voxels[cid]))]
            z = int(np.clip(cz - pz // 2 + self.rng.randint(-8, 9), 0, max_z))
            y = int(np.clip(cy - py // 2 + self.rng.randint(-8, 9), 0, max_y))
            x = int(np.clip(cx - px // 2 + self.rng.randint(-8, 9), 0, max_x))
            return z, y, x
        return (
            int(self.rng.randint(0, max_z + 1)),
            int(self.rng.randint(0, max_y + 1)),
            int(self.rng.randint(0, max_x + 1)),
        )

    def __getitem__(self, idx: int):
        cid = self.case_ids[idx % len(self.case_ids)]
        images, labels, _ = D.load_cached(cid)
        z, y, x = self._origins[idx] if self.frozen else self._pick_origin(cid)
        pz, py, px = self.patch

        patch_img = np.asarray(images[:, z : z + pz, y : y + py, x : x + px]).astype(np.float32)
        patch_mask = (np.asarray(labels[z : z + pz, y : y + py, x : x + px]) > 0).astype(np.float32)[None]

        return torch.from_numpy(patch_img), torch.from_numpy(patch_mask)
