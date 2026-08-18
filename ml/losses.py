"""Soft Dice + BCE, the loss for both the 2.5D and 3D models.

Combined rather than either alone: Dice handles the class imbalance directly
(BCE alone on ~7.5%-foreground brain voxels spends most of its gradient on
background it was already getting right); BCE keeps early training stable
when the Dice surface is flat because the network is predicting near-zero
foreground everywhere.

`DeepSupervisionLoss` wraps that compound loss across a deep-supervised
model's multi-resolution outputs — see `config.DEEP_SUPERVISION`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """logits, target: (B, 1, ...) — any number of trailing spatial dims.

    Computed in fp32 regardless of an enclosing `torch.autocast` region — this
    is not cosmetic. `prob.sum(dim=dims)` is an *unnormalised* per-sample sum
    over every spatial element, and fp16's max representable value is only
    65504. The 3D model's full-resolution patch is 96*160*160 = 2,457,600
    elements: confirmed on this machine that a saturated fp16 tensor of that
    shape (all elements ~0.95, the kind of prediction an early or over-
    confident model produces) sums to `inf`, while the identical tensor cast
    to fp32 first sums to a correct, finite value. `inf / inf` from `inter`
    and `denom` both overflowing the same way is `nan` — and once a nan
    enters the weights it poisons every later forward pass (see the
    isfinite-loss skip-and-log in train.py, which is what surfaced this).
    The 2.5D head (192x192 = 36864 elements) is individually safer but not
    provably safe under saturation either, so the cast applies uniformly
    rather than only where the arithmetic has already been shown to bite.
    """
    logits, target = logits.float(), target.float()
    prob = torch.sigmoid(logits)
    dims = tuple(range(2, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Same fp32-forced reasoning as soft_dice_loss — BCE-with-logits is
        # the numerically-stable log-sum-exp form and is usually fine in fp16,
        # but casting costs nothing next to a conv forward/backward pass and
        # removes it as a variable entirely rather than trusting "usually".
        logits, target = logits.float(), target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = soft_dice_loss(logits, target)
        return self.dice_weight * dice + (1 - self.dice_weight) * bce


@torch.no_grad()
def hard_dice(logits: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5) -> float:
    """Batch-mean hard Dice at a fixed threshold — used for training-loop logging.

    Not the metric used for the reported result (metrics.dice, run per-case on
    full reconstructed volumes) — this is a fast per-batch proxy so training
    curves (F07) are cheap to produce every epoch.
    """
    if not torch.is_tensor(logits):
        # A deep-supervised model returns a list of logits in training mode. The
        # validation loop runs under model.eval() and so already gets a single
        # tensor, but score the full-resolution head rather than crashing (or,
        # worse, silently scoring a 1/4-resolution head) if this is ever called
        # on a train-mode forward pass.
        logits = logits[0]
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = torch.where(denom == 0, torch.ones_like(denom), (2 * inter) / denom.clamp(min=1e-6))
    return float(dice.mean().item())


# ── deep supervision ─────────────────────────────────────────────────────────


def _pool_target(target: torch.Tensor, size: tuple[int, ...], *, mode: str = C.DS_TARGET_POOLING) -> torch.Tensor:
    """Match `target` down to one deep-supervision head's spatial size.

    See `config.DS_TARGET_POOLING` for why "avg" is the default and why the
    choice is load-bearing rather than cosmetic — nearest-neighbour can delete
    this cohort's smallest lesion outright at 1/4 or 1/8 resolution. "nearest"
    is kept so the two can be ablated against each other rather than the
    better-sounding option simply being asserted.
    """
    if tuple(target.shape[2:]) == tuple(size):
        return target
    if mode == "avg":
        pool = F.adaptive_avg_pool2d if target.ndim == 4 else F.adaptive_avg_pool3d
        return pool(target, tuple(size))
    if mode == "nearest":
        return F.interpolate(target, size=tuple(size), mode="nearest")
    raise ValueError(f"unknown deep-supervision target pooling mode: {mode!r}")


class DeepSupervisionLoss(nn.Module):
    """Weighted sum of `base_loss` over a model's multi-resolution outputs.

    Accepts either the list of logits a deep-supervised model returns in
    training mode, or a single tensor — so the training loop never has to branch
    on whether deep supervision is enabled, and inference (always under
    `model.eval()`, which returns one tensor) is unaffected either way.
    """

    def __init__(self, base_loss: nn.Module | None = None, *, decay: float = C.DS_WEIGHT_DECAY_PER_LEVEL):
        super().__init__()
        self.base_loss = DiceBCELoss() if base_loss is None else base_loss
        self.decay = decay

    def weights(self, n: int) -> list[float]:
        """Halving weights, full-resolution head first, normalised to sum to 1.

        The normalisation is not cosmetic: an un-normalised sum would scale the
        total loss with the number of heads, and therefore the gradient
        magnitude and the effective learning rate. Without it, a
        deep-supervision ablation would also be an unintended LR ablation — and
        LR is exactly what this architecture was already shown to be fragile to
        (see the fp16-instability note on `config.LR`).
        """
        raw = [self.decay**i for i in range(n)]
        total = sum(raw)
        return [w / total for w in raw]

    def forward(self, outputs, target: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(outputs):
            return self.base_loss(outputs, target)
        loss = outputs[0].new_zeros(())
        for w, logits in zip(self.weights(len(outputs)), outputs):
            loss = loss + w * self.base_loss(logits, _pool_target(target, tuple(logits.shape[2:])))
        return loss
