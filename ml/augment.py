"""On-device batched augmentation, shared by the 2.5D and 3D training loops.

Everything here runs on the GPU, after a batch has already been moved there —
that is what "on-device" buys: augmenting on the CPU inside a DataLoader
worker would make the dataloader the bottleneck on a GPU this fast, for a
dataset this small (48 cases can be held in GPU memory whole; there is no
disk-bound reason to leave the augmentation on CPU).

One function, `augment_batch`, handles both models: it infers 2D vs 3D from
how many trailing spatial dims the input has (`x.dim() - 2`), builds one
random affine matrix per sample, and applies it to image and mask together
with a single `affine_grid` + `grid_sample` call — the only way to guarantee
the mask stays registered to the image under rotation/scale/flip.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import config as C


def _random_affine_theta_2d(batch: int, device, *, rotate_deg: float, scale_jitter: float, flip_prob: float):
    angle = (torch.rand(batch, device=device) * 2 - 1) * rotate_deg * (torch.pi / 180)
    scale = 1 + (torch.rand(batch, device=device) * 2 - 1) * scale_jitter
    flip_x = torch.where(torch.rand(batch, device=device) < flip_prob, -1.0, 1.0)

    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0] = cos * scale * flip_x
    theta[:, 0, 1] = -sin * scale
    theta[:, 1, 0] = sin * scale * flip_x
    theta[:, 1, 1] = cos * scale
    return theta


def _random_affine_theta_3d(batch: int, device, *, rotate_deg: float, scale_jitter: float, flip_prob: float):
    """Rotation about the z (axial) axis only — see AUG_ROTATE_DEG_3D in config.

    Free-axis 3D rotation would need Euler-angle composition and risks
    introducing anatomy-implausible tilts; rotating about the axis the patient
    actually varies around (a slightly turned head, same scanner orientation)
    is both simpler and the more defensible augmentation for this data.
    """
    angle = (torch.rand(batch, device=device) * 2 - 1) * rotate_deg * (torch.pi / 180)
    scale = 1 + (torch.rand(batch, device=device) * 2 - 1) * scale_jitter
    flips = torch.where(torch.rand(batch, 3, device=device) < flip_prob, -1.0, 1.0)  # per z,y,x axis

    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros(batch, 3, 4, device=device)
    theta[:, 0, 0] = scale * flips[:, 0]  # z axis untouched by the rotation
    theta[:, 1, 1] = cos * scale * flips[:, 1]
    theta[:, 1, 2] = -sin * scale
    theta[:, 2, 1] = sin * scale
    theta[:, 2, 2] = cos * scale * flips[:, 2]
    return theta


def _spatial_transform(images: torch.Tensor, masks: torch.Tensor, *, ndim: int) -> tuple[torch.Tensor, torch.Tensor]:
    b, device = images.shape[0], images.device
    if ndim == 2:
        theta = _random_affine_theta_2d(
            b, device, rotate_deg=C.AUG_ROTATE_DEG_2D, scale_jitter=C.AUG_SCALE_JITTER, flip_prob=C.AUG_FLIP_PROB
        )
    else:
        theta = _random_affine_theta_3d(
            b, device, rotate_deg=C.AUG_ROTATE_DEG_3D, scale_jitter=C.AUG_SCALE_JITTER, flip_prob=C.AUG_FLIP_PROB
        )
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    images_t = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    masks_t = F.grid_sample(masks, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return images_t, masks_t


def _intensity_transform(images: torch.Tensor) -> torch.Tensor:
    b, device = images.shape[0], images.device
    ndim = images.dim() - 2
    shape = (b, 1) + (1,) * ndim

    scale = 1 + (torch.rand(shape, device=device) * 2 - 1) * C.AUG_INTENSITY_SCALE
    shift = (torch.rand(shape, device=device) * 2 - 1) * C.AUG_INTENSITY_SHIFT
    images = images * scale + shift

    noise = torch.randn_like(images) * C.AUG_NOISE_SIGMA
    images = images + noise

    images = images * _bias_field(images.shape, device, ndim=ndim)
    return images


def _bias_field(shape: tuple[int, ...], device, *, ndim: int) -> torch.Tensor:
    """A smooth per-sample multiplicative field, simulating MRI coil bias.

    Built by upsampling low-resolution random noise rather than a physical
    field model — coarse-to-fine random-then-smooth is enough to make the
    network robust to the low-frequency intensity drift real scanners produce,
    without needing to simulate the coil physics.
    """
    b = shape[0]
    res = (C.AUG_BIAS_FIELD_RES,) * ndim
    coarse = 1 + (torch.rand((b, 1, *res), device=device) * 2 - 1) * C.AUG_BIAS_FIELD_MAGNITUDE
    mode = "bilinear" if ndim == 2 else "trilinear"
    return F.interpolate(coarse, size=shape[2:], mode=mode, align_corners=False)


def augment_batch(images: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """images: (B, C, *spatial) float. masks: (B, 1, *spatial) float in {0, 1}.

    Spatial dims inferred from `images.dim() - 2` — 2 for the 2.5D model's
    (B, 20, H, W), 3 for the 3D model's (B, 4, Z, Y, X).
    """
    ndim = images.dim() - 2
    assert ndim in (2, 3), f"expected a 4D or 5D batch, got {images.shape}"
    images, masks = _spatial_transform(images, masks, ndim=ndim)
    images = _intensity_transform(images)
    return images, (masks > 0.5).float()
