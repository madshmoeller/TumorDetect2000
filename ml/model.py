"""Two U-Net variants sharing one building-block vocabulary, plus a timing probe.

    UNet2p5D  — 4 modalities x 5 adjacent slices stacked as 20 input channels,
                depth-5, the fast comparison point (figure F13).
    UNet3D    — 4 modalities, real 3D convolutions on (96,160,160) patches,
                depth-4, the primary architecture now that training runs on
                this machine's RTX 3090 Ti rather than an M2 Max.

Both are residual-encoder U-Nets: InstanceNorm + LeakyReLU (BatchNorm is a bad
fit for fold sizes this small — 48 cases, and a 3D patch batch of 2-3 makes
per-batch statistics noisy), residual blocks in the encoder for gradient flow
at depth, plain conv blocks in the decoder.

    python -m ml.model --probe        # real timing probe on this GPU, both archs
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn

from . import config as C
from .losses import DeepSupervisionLoss


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── shared blocks ────────────────────────────────────────────────────────────


def _conv(in_ch: int, out_ch: int, ndim: int, *, kernel: int = 3, stride: int = 1) -> nn.Module:
    Conv = nn.Conv2d if ndim == 2 else nn.Conv3d
    return Conv(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)


class ResBlock(nn.Module):
    """conv-norm-act-conv-norm, added to a 1x1-projected skip, then a final act.

    conv1/skip both map in_ch -> out_ch at stride 1 with `padding=kernel//2`,
    so the main branch and the skip branch always leave spatial size
    unchanged and land on identical shapes — no shape-matching logic needed
    at the add, unlike at the decoder's upsample+concat (see `_match_and_cat`).
    """

    def __init__(self, in_ch: int, out_ch: int, ndim: int):
        super().__init__()
        Norm = nn.InstanceNorm2d if ndim == 2 else nn.InstanceNorm3d
        self.conv1 = _conv(in_ch, out_ch, ndim)
        self.norm1 = Norm(out_ch, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = _conv(out_ch, out_ch, ndim)
        self.norm2 = Norm(out_ch, affine=True)
        self.skip = _conv(in_ch, out_ch, ndim, kernel=1) if in_ch != out_ch else nn.Identity()
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act2(h + identity)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ndim: int):
        super().__init__()
        Pool = nn.MaxPool2d if ndim == 2 else nn.MaxPool3d
        self.pool = Pool(2)
        self.block = ResBlock(in_ch, out_ch, ndim)

    def forward(self, x):
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, ndim: int):
        super().__init__()
        ConvT = nn.ConvTranspose2d if ndim == 2 else nn.ConvTranspose3d
        self.up = ConvT(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ResBlock(out_ch + skip_ch, out_ch, ndim)

    def forward(self, x, skip):
        x = self.up(x)
        x = _match_and_cat(x, skip)
        return self.block(x)


def _match_and_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Crop `skip` to `x`'s spatial size before concatenating.

    Transposed-conv output size can be off by one voxel from the skip
    connection when an input dimension isn't perfectly divisible by 2**depth
    (true for the 3D model's (96,160,160) patches under some depths) — center
    -crop rather than pad, so the network never sees a border of zeros it
    didn't ask for.
    """
    if x.shape[2:] == skip.shape[2:]:
        return torch.cat([x, skip], dim=1)
    slices = [slice(None), slice(None)]
    for xs, ss in zip(x.shape[2:], skip.shape[2:]):
        start = (ss - xs) // 2
        slices.append(slice(start, start + xs))
    return torch.cat([x, skip[tuple(slices)]], dim=1)


class _UNet(nn.Module):
    """Generic depth-N residual U-Net. Both models below are thin wrappers."""

    def __init__(self, in_channels: int, base_width: int, depth: int, ndim: int,
                 *, deep_supervision: bool = C.DEEP_SUPERVISION):
        super().__init__()
        self.ndim = ndim
        widths = [base_width * (2**i) for i in range(depth + 1)]

        self.stem = ResBlock(in_channels, widths[0], ndim)
        self.downs = nn.ModuleList([DownBlock(widths[i], widths[i + 1], ndim) for i in range(depth)])
        self.ups = nn.ModuleList(
            [UpBlock(widths[i + 1], widths[i], widths[i], ndim) for i in reversed(range(depth))]
        )
        Conv = nn.Conv2d if ndim == 2 else nn.Conv3d
        self.head = Conv(widths[0], 1, kernel_size=1)

        # Deep supervision: `ups[k]` emits widths[depth-1-k] channels at
        # 1/2**(depth-1-k) resolution, so `ups[-1]` is full resolution and is
        # already covered by self.head above. Aux heads go on every *other*
        # decoder stage except ups[0], the lowest-resolution one — see
        # config.DEEP_SUPERVISION for why the coarsest stage is skipped.
        # depth < 3 leaves no intermediate stage to attach to, so the list is
        # empty and the model behaves exactly as it did before.
        self.deep_supervision = deep_supervision
        self.aux_heads = nn.ModuleList(
            [Conv(widths[depth - 1 - k], 1, kernel_size=1) for k in range(1, depth - 1)]
            if deep_supervision else []
        )

    def forward(self, x):
        skips = [self.stem(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips[-1]
        aux = []
        for k, (up, skip) in enumerate(zip(self.ups, reversed(skips[:-1]))):
            x = up(x, skip)
            if self.aux_heads and 1 <= k <= len(self.aux_heads):
                aux.append(self.aux_heads[k - 1](x))
        out = self.head(x)
        # Training mode only, and full resolution first — the order
        # DeepSupervisionLoss's halving weights assume. Under model.eval() this
        # returns a single tensor exactly as before, which is what keeps
        # infer.py, the TTA flip and the checkpoint format untouched.
        if self.training and aux:
            return [out, *reversed(aux)]
        return out


class UNet2p5D(_UNet):
    def __init__(self, *, deep_supervision: bool = C.DEEP_SUPERVISION):
        super().__init__(C.IN_CHANNELS_25D, C.BASE_WIDTH_25D, C.DEPTH_25D, ndim=2,
                         deep_supervision=deep_supervision)


class UNet3D(_UNet):
    def __init__(self, *, deep_supervision: bool = C.DEEP_SUPERVISION):
        super().__init__(C.IN_CHANNELS_3D, C.BASE_WIDTH_3D, C.DEPTH_3D, ndim=3,
                         deep_supervision=deep_supervision)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ── timing probe ─────────────────────────────────────────────────────────────


def probe(model: nn.Module, input_shape: tuple[int, ...], device: torch.device, *, n_steps: int = 15) -> dict:
    """Run real forward+backward steps and measure wall-clock per step.

    Replaces the original plan's hand-computed GFLOPs-per-step estimate (which
    existed only because there was no way to run the model on the target Mac
    in advance). There is no such excuse on a machine that's sitting right
    here — this is a *measurement*, not a projection from a formula.

    The model is left in training mode and scored with the real training loss
    (Dice+BCE, wrapped for deep supervision), not a stand-in BCE on the main
    head alone. With deep supervision on, the cheaper stand-in would skip the
    aux heads' backward pass entirely and hand `train.dry_run` a per-step number
    lower than any step the real run ever takes — a budget projection that
    flatters itself is worse than no projection.
    """
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = DeepSupervisionLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    x = torch.randn(*input_shape, device=device)
    y = (torch.rand(input_shape[0], 1, *input_shape[2:], device=device) > 0.9).float()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        return loss

    for _ in range(3):  # warmup: cuDNN autotune, allocator warmup
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
    return {
        "device": str(device),
        "input_shape": list(input_shape),
        "n_params": n_params(model),
        "sec_per_step": elapsed / n_steps,
        "peak_mem_gb": peak_mem,
        "amp": use_amp,
    }


def project_schedule(sec_per_step: float, *, steps_per_epoch: int, epochs: int, n_folds: int) -> float:
    """Projected total minutes for the full CV run at a given schedule."""
    return sec_per_step * steps_per_epoch * epochs * n_folds / 60


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args(argv)

    device = pick_device()
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

    if not args.probe:
        ap.print_help()
        return 0

    # 2.5D: full in-plane crop, batch = config default
    m25 = UNet2p5D()
    shape25 = (C.BATCH_SIZE_25D, C.IN_CHANNELS_25D, *C.CROP_HW)
    r25 = probe(m25, shape25, device)
    print(f"\nUNet2p5D   params={r25['n_params']:,}  input={r25['input_shape']}  "
          f"{r25['sec_per_step'] * 1000:.1f} ms/step  mem={r25['peak_mem_gb']}")

    # 3D: patch, batch = config default
    m3d = UNet3D()
    shape3d = (C.BATCH_SIZE_3D, C.IN_CHANNELS_3D, *C.PATCH_3D)
    r3d = probe(m3d, shape3d, device)
    print(f"UNet3D     params={r3d['n_params']:,}  input={r3d['input_shape']}  "
          f"{r3d['sec_per_step'] * 1000:.1f} ms/step  mem={r3d['peak_mem_gb']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
