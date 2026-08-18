"""5-fold cross-validation training for both architectures.

Every one of the 60 cases ends up in exactly one validation fold and is
therefore scored, later, by a model that never saw it during training — the
out-of-fold guarantee rule 3/4 depend on.

    python -m ml.train --dry-run             # measured probe -> projected schedule, no training
    python -m ml.train --arch 25d             # 5-fold CV, 2.5D model
    python -m ml.train --arch 3d              # 5-fold CV, 3D model
    python -m ml.train --arch 25d --folds 0   # just fold 0 (useful for the overfit/smoke tests)
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import augment as A
from . import config as C
from . import model as MDL
from .datasets import PatchDataset3D, SliceDataset25D
from .losses import DeepSupervisionLoss, hard_dice

CHECKPOINTS = C.OUTPUTS / "checkpoints"

ARCH = {
    "25d": dict(
        model_cls=MDL.UNet2p5D,
        dataset_cls=SliceDataset25D,
        batch_size=C.BATCH_SIZE_25D,
        epochs=C.EPOCHS_25D,
        samples_per_case=C.SLICES_PER_CASE_25D,
        val_samples_per_case=16,
    ),
    "3d": dict(
        model_cls=MDL.UNet3D,
        dataset_cls=PatchDataset3D,
        batch_size=C.BATCH_SIZE_3D,
        epochs=C.EPOCHS_3D,
        samples_per_case=C.PATCHES_PER_CASE_3D,
        val_samples_per_case=3,
    ),
}


def _loader(dataset_cls, ids: list[str], samples_per_case: int, batch_size: int, *, seed: int, shuffle: bool,
            frozen: bool = False):
    kwarg = "samples_per_case" if dataset_cls is SliceDataset25D else "patches_per_case"
    ds = dataset_cls(ids, **{kwarg: samples_per_case}, seed=seed, frozen=frozen)
    # drop_last=False unconditionally: both models use InstanceNorm (per-sample
    # statistics), so a partial final batch is not a training-stability problem
    # the way it would be with BatchNorm. Tying drop_last to `shuffle` used to
    # mean a dataset smaller than one batch (e.g. the overfit-one-case smoke
    # test) silently trained on zero batches every epoch — caught by that gate,
    # not by a code review, which is exactly why the gate exists.
    return DataLoader(
        ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle, num_workers=min(6, len(ds)),
        pin_memory=True, persistent_workers=len(ds) > 0, drop_last=False,
    )


def train_one_fold(
    arch: str, fold: int, train_ids: list[str], val_ids: list[str], device: torch.device, *,
    verbose: bool = True, epochs: int | None = None, augment: bool = True, save_checkpoint: bool = True,
) -> dict:
    spec = ARCH[arch]
    epochs = spec["epochs"] if epochs is None else epochs
    model = spec["model_cls"]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    # Handles both a deep-supervised model's list of multi-resolution logits and
    # a single tensor, so this line is correct whether C.DEEP_SUPERVISION is on
    # or off and the training loop below needs no branch on it.
    loss_fn = DeepSupervisionLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_loader = _loader(spec["dataset_cls"], train_ids, spec["samples_per_case"], spec["batch_size"],
                            seed=fold, shuffle=True)
    # frozen=True: the validation set must be a FIXED yardstick. Unfrozen it
    # re-draws every patch each epoch, so "best epoch" is judged against a
    # moving target — see PatchDataset3D's docstring and config's
    # EARLY_STOP_MIN_FRACTION for what that cost the 2026-08-17 run.
    val_loader = _loader(spec["dataset_cls"], val_ids, spec["val_samples_per_case"], spec["batch_size"],
                          seed=1000 + fold, shuffle=False, frozen=True)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C.LR, total_steps=max(total_steps, 1))

    history = []
    best_val_dice, best_epoch, patience_left = -1.0, -1, C.EARLY_STOP_PATIENCE
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINTS / f"{arch}_fold{fold}.pt"

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_losses = []
        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            if augment:
                images, masks = A.augment_batch(images, masks)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, masks)

            if not torch.isfinite(loss):
                # Once a nan/inf enters the weights it poisons every later
                # forward pass — an early fold run showed exactly that
                # (loss.item() stayed nan for the rest of training after one
                # bad step). Skipping the step outright is cheap insurance
                # clipping alone doesn't give: clipping bounds *finite*
                # gradients, it does not save you from a step whose loss was
                # already nan/inf before backward() ever ran.
                sched.step()
                train_losses.append(float("nan"))
                continue

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            sched.step()
            train_losses.append(loss.item())

        model.eval()
        val_dices = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                val_dices.append(hard_dice(logits, masks))

        # nanmean, not mean: one skipped (nan-loss) batch in an otherwise fine
        # epoch should not blank out the whole epoch's logged loss — but it
        # does still show up, via n_skipped, as a signal something happened.
        n_skipped = int(np.isnan(train_losses).sum()) if train_losses else 0
        train_loss = float(np.nanmean(train_losses)) if train_losses and n_skipped < len(train_losses) else float("nan")
        val_dice = float(np.mean(val_dices)) if val_dices else float("nan")
        epoch_sec = time.time() - t0
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_dice": val_dice,
             "lr": sched.get_last_lr()[0], "sec": epoch_sec, "n_skipped": n_skipped}
        )
        if verbose:
            skip_note = f"  [{n_skipped} nan-loss batch(es) skipped]" if n_skipped else ""
            print(f"  [{arch} fold{fold}] epoch {epoch:3d}  loss={train_loss:.4f}  "
                  f"val_dice(patch-proxy)={val_dice:.4f}  {epoch_sec:.1f}s{skip_note}")

        if val_dice > best_val_dice:
            best_val_dice, best_epoch, patience_left = val_dice, epoch, C.EARLY_STOP_PATIENCE
            if save_checkpoint:
                # deep_supervision is recorded because it changes the state_dict's
                # key set (aux_heads.*). infer.py's load_state_dict is strict, so
                # a checkpoint trained under one setting and loaded under the
                # other fails loudly rather than silently — recording the flag
                # makes the artifact self-describing when that happens.
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice,
                            "deep_supervision": getattr(model, "deep_supervision", False)}, ckpt_path)
        else:
            # Do not let patience end the fold before OneCycle has annealed —
            # the anneal is where most of the final quality comes from, and a
            # warmup-phase peak on a noisy proxy is not evidence the model has
            # stopped improving. See config.EARLY_STOP_MIN_FRACTION.
            if epoch >= C.EARLY_STOP_MIN_FRACTION * epochs:
                patience_left -= 1
                if patience_left <= 0:
                    if verbose:
                        print(f"  [{arch} fold{fold}] early stop at epoch {epoch} "
                              f"(best {best_val_dice:.4f} @ epoch {best_epoch})")
                    break

    # The final-epoch weights, kept alongside the best-by-validation ones so
    # "would annealing have helped?" is answerable by comparison rather than by
    # a second 100-minute run — see config.SAVE_FINAL_CHECKPOINT.
    final_path = None
    if save_checkpoint and C.SAVE_FINAL_CHECKPOINT and history:
        final_path = CHECKPOINTS / f"{arch}_fold{fold}_final.pt"
        torch.save({"model": model.state_dict(), "epoch": history[-1]["epoch"],
                    "val_dice": history[-1]["val_dice"],
                    "deep_supervision": getattr(model, "deep_supervision", False)}, final_path)

    return {
        "arch": arch, "fold": fold, "train_ids": train_ids, "val_ids": val_ids,
        "history": history, "best_val_dice": best_val_dice, "best_epoch": best_epoch,
        "checkpoint": str(ckpt_path),
        "final_checkpoint": str(final_path) if final_path else None,
        "final_epoch": history[-1]["epoch"] if history else None,
        "ds_target_pooling": C.DS_TARGET_POOLING,
        "deep_supervision": C.DEEP_SUPERVISION,
    }


def run_cv(arch: str, ids: list[str], device: torch.device, *, only_folds: list[int] | None = None) -> dict:
    folds = C.fold_assignment(ids)
    fold_ids = only_folds if only_folds is not None else list(range(C.N_FOLDS))

    all_results = []
    t0 = time.time()
    for f in fold_ids:
        train_ids = [c for c in ids if folds[c] != f]
        val_ids = [c for c in ids if folds[c] == f]
        print(f"\n=== {arch}  fold {f}: {len(train_ids)} train, {len(val_ids)} val ===")
        result = train_one_fold(arch, f, train_ids, val_ids, device)
        all_results.append(result)

    elapsed_min = (time.time() - t0) / 60
    out = {"arch": arch, "folds": all_results, "wall_clock_minutes": elapsed_min, "fold_assignment": folds}
    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    (C.OUTPUTS / f"train_history_{arch}.json").write_text(json.dumps(out, indent=2))
    print(f"\n{arch}: {elapsed_min:.1f} min measured (not estimated) for {len(fold_ids)} fold(s)")
    return out


def dry_run(device: torch.device) -> None:
    """Project the training schedule from a real timing probe, before spending any wall-clock on it."""
    print("measured probe -> projected schedule (not a hand-computed estimate):\n")
    total_minutes = 0.0
    n_train = len(C.case_ids()) - len(C.case_ids()) // C.N_FOLDS  # cases per training fold, approx

    for arch in ("25d", "3d"):
        spec = ARCH[arch]
        model = spec["model_cls"]()
        if arch == "25d":
            shape = (spec["batch_size"], C.IN_CHANNELS_25D, *C.CROP_HW)
        else:
            shape = (spec["batch_size"], C.IN_CHANNELS_3D, *C.PATCH_3D)
        r = MDL.probe(model, shape, device, n_steps=10)

        steps_per_epoch = -(-n_train * spec["samples_per_case"] // spec["batch_size"])  # ceil div
        minutes = MDL.project_schedule(
            r["sec_per_step"], steps_per_epoch=steps_per_epoch, epochs=spec["epochs"], n_folds=C.N_FOLDS
        )
        total_minutes += minutes
        print(f"  {arch:4s}  {r['sec_per_step'] * 1000:6.1f} ms/step  "
              f"{steps_per_epoch:4d} steps/epoch x {spec['epochs']:3d} epochs x {C.N_FOLDS} folds  "
              f"-> {minutes:6.1f} min  (mem {r['peak_mem_gb']})")
        del model

    print(f"\n  total projected: {total_minutes:.1f} min  vs ceiling {C.TRAIN_BUDGET_MINUTES} min "
          f"({'OK' if total_minutes <= C.TRAIN_BUDGET_MINUTES else 'OVER BUDGET'})")
    print("  (early stopping means the real run is typically shorter than this upper bound)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["25d", "3d"], help="required unless --dry-run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--folds", type=str, default=None, help="comma-separated fold indices, e.g. '0' or '0,1'")
    ap.add_argument("--tier", default="standard", choices=["standard", "tiny"])
    args = ap.parse_args(argv)

    device = MDL.pick_device()
    print(f"device: {device}")

    if args.dry_run:
        dry_run(device)
        return 0

    if not args.arch:
        ap.error("--arch is required unless --dry-run")

    ids = C.case_ids(args.tier)
    only_folds = [int(x) for x in args.folds.split(",")] if args.folds else None
    run_cv(args.arch, ids, device, only_folds=only_folds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
