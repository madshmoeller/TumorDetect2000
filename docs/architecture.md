# Architecture, training scheme and hyperparameters

Reference sheet for both models. Every number here is either read directly from
[`ml/config.py`](../ml/config.py) or verified by running the model — nothing is
hand-derived. Where a figure has not been measured yet it says so rather than
carrying an estimate (rule 2).

Generated 2026-08-17, after deep supervision was added and before the 3D
cross-validation run.

---

## 1. The task

```
input                                                     output
─────                                                     ──────
4 co-registered MRI channels                              1 binary map
FLAIR │ T1w │ T1-Gd │ T2w                                 tumour / not tumour
     240 x 240 x 155  (1 mm isotropic)                    same grid as input
            │
            ▼  cropped to the audited brain bounding box
     192 x 192 x 155  (y 16..208, x 24..216, z full)
```

Target is `label > 0` — the three shipped sub-labels (oedema, non-enhancing
core, enhancing tumour) collapsed into one foreground class. This is the BraTS
"whole tumour" region, the easiest of the three, and not comparable to published
core/enhancing scores.

n = 60 cases. 1 voxel = 1 mm³, so voxel count *is* volume in mm³.

---

## 2. Shared building blocks

Both models are residual-encoder U-Nets built from one vocabulary. `InstanceNorm`
rather than `BatchNorm` throughout: fold sizes are 48 cases and the 3D patch
batch is 8, which makes per-batch statistics noisy.

### ResBlock

```
        ┌─ Conv(k3, in→out) ─ InstanceNorm ─ LeakyReLU(0.01) ─┐
   x ───┤                                                     ├─ Conv(k3, out→out) ─ InstanceNorm ─┐
        │                                                                                          (+) ─ LeakyReLU(0.01) ─→
        └─ Conv(k1, in→out)  [Identity when in == out] ────────────────────────────────────────────┘
```

Both branches run at stride 1 with `padding = k//2`, so they always land on
identical shapes and the add needs no shape-matching logic.

### DownBlock / UpBlock

```
DownBlock:   MaxPool(2) ─→ ResBlock(in→out)

UpBlock:     ConvTranspose(k2, s2, in→out) ─→ concat with encoder skip ─→ ResBlock(out+skip→out)
                                                │
                                                └─ skip is CENTRE-CROPPED to match, never zero-padded
                                                   (transposed-conv output can be off by one voxel
                                                    when a dim isn't divisible by 2**depth)
```

---

## 3. Primary model — UNet3D

**8,959,163 parameters** (verified by `ml.model.n_params`). Depth 4, base width
20, real 3D convolutions on random patches.

```
 patch: 4 x 96 x 160 x 160
        │
        │  ENCODER                                                    DECODER
        ▼                                                                  ▲
  ┌───────────────┐                                              ┌──────────────────┐
  │ stem          │  4→20   96x160x160  ──────── skip ──────────▶ │ up3  40→20       │ ─→ HEAD  Conv1x1 20→1
  └───────┬───────┘                                              └────────▲─────────┘    96x160x160  (main)
          │ pool 2                                                        │ ConvT 40→20
  ┌───────▼───────┐                                              ┌────────┴─────────┐
  │ down0         │ 20→40   48x 80x 80  ──────── skip ──────────▶ │ up2  80→40       │ ─→ aux  Conv1x1 40→1
  └───────┬───────┘                                              └────────▲─────────┘    48x80x80   (1/2)
          │ pool 2                                                        │ ConvT 80→40
  ┌───────▼───────┐                                              ┌────────┴─────────┐
  │ down1         │ 40→80   24x 40x 40  ──────── skip ──────────▶ │ up1  160→80      │ ─→ aux  Conv1x1 80→1
  └───────┬───────┘                                              └────────▲─────────┘    24x40x40   (1/4)
          │ pool 2                                                        │ ConvT 160→80
  ┌───────▼───────┐                                              ┌────────┴─────────┐
  │ down2         │ 80→160  12x 20x 20  ──────── skip ──────────▶ │ up0  320→160     │   (no head — lowest
  └───────┬───────┘                                              └────────▲─────────┘    decoder stage, skipped)
          │ pool 2                                                        │ ConvT 320→160
  ┌───────▼─────────────────────────────────────────────────────────────┴─┐
  │ down3   160→320    6 x 10 x 10                            BOTTLENECK  │
  └───────────────────────────────────────────────────────────────────────┘
```

Widths per level: `[20, 40, 80, 160, 320]`.

Trained on patches, evaluated by sliding-window inference over the full
192x192x155 crop. Patch training rather than whole-volume: it generalises better
on a cohort this small, and it is what makes batch size > 1 affordable in 24 GB.

---

## 4. Comparison model — UNet2p5D

**18,286,804 parameters.** Depth 5, base width 24. Consumes 4 modalities x 5
adjacent slices stacked as 20 input channels — one slice plus ±2 neighbours, so
it approximates through-plane context without paying for 3D convolutions.

```
 slice stack: 20 x 192 x 192                                  widths [24,48,96,192,384,768]

   stem   20→24    192x192 ─── skip ───▶  up4   48→24    192x192  ─→ HEAD Conv1x1 24→1  (main)
   down0  24→48     96x 96 ─── skip ───▶  up3   96→48     96x 96  ─→ aux  Conv1x1 48→1  (1/2)
   down1  48→96     48x 48 ─── skip ───▶  up2  192→96     48x 48  ─→ aux  Conv1x1 96→1  (1/4)
   down2  96→192    24x 24 ─── skip ───▶  up1  384→192    24x 24  ─→ aux  Conv1x1 192→1 (1/8)
   down3 192→384    12x 12 ─── skip ───▶  up0  768→384    12x 12     (no head — skipped)
   down4 384→768     6x  6   BOTTLENECK
```

Kept as figure F13's comparison point, not as the deliverable. Trains in minutes.

---

## 5. Deep supervision

Auxiliary 1x1-conv segmentation heads on the decoder's intermediate
resolutions, their losses summed with the main head's. This was the one nnU-Net
ingredient the codebase was missing — residual encoder, InstanceNorm+LeakyReLU,
Dice+BCE compound loss, patch training, AMP and TTA were already present.

```
                                   ┌─ main head  (full res)  weight w0
   decoder stages ─────────────────┼─ aux head   (1/2 res)   weight w1
                                   ├─ aux head   (1/4 res)   weight w2
                                   └─ aux head   (1/8 res)   weight w3   [2.5D only]

   loss = Σ wi · DiceBCE( logits_i , pool(target → resolution_i) )
```

| | heads | resolutions | weights (normalised) |
|---|---|---|---|
| UNet3D | 3 | full, 1/2, 1/4 | 0.5714, 0.2857, 0.1429 |
| UNet2p5D | 4 | full, 1/2, 1/4, 1/8 | 0.5333, 0.2667, 0.1333, 0.0667 |

Weights halve as resolution does and are **normalised to sum to 1**. That is not
cosmetic: an un-normalised sum would scale total loss with head count, and so
the gradient magnitude and effective learning rate — turning a deep-supervision
ablation into an unintended LR ablation. This architecture is already known to be
fragile to LR (see §7).

The lowest-resolution decoder stage carries no head, per nnU-Net convention.

Two properties that keep inference untouched:

- `forward` returns a **list** of logits in training mode and a **single tensor**
  under `model.eval()`. `infer.py` always calls `.eval()` first, so sliding-window
  inference, the TTA flip and the checkpoint contract are unchanged.
- Checkpoints record `deep_supervision`, because it changes the `state_dict` key
  set (`aux_heads.*`). `load_state_dict` is strict, so a mismatch fails loudly.

### Target downsampling: measured, not argued

Targets are matched to each head's resolution by **adaptive average pooling**.
Counting targets that become all-background even though the source contains
tumour, over all 60 cached label volumes:

| head | avg emptied | nearest emptied | nearest mass retained |
|---|---|---|---|
| 3D 1/2 | 0 / 200 | 0 / 200 | 1.0015 |
| 3D 1/4 | 0 / 200 | 0 / 200 | 0.9880 |
| 2.5D 1/2 | 0 / 4290 | 35 / 4290 | 0.9997 |
| 2.5D 1/4 | 0 / 4290 | 94 / 4290 | 1.0089 |
| 2.5D 1/8 | 0 / 4290 | **229 / 4290 (5.34%)** | 1.0380 |

So the choice is load-bearing **for the 2.5D model only**, at its deepest heads.
For the 3D model it is very nearly immaterial.

An earlier draft justified average pooling by claiming nearest-neighbour would
delete the cohort's smallest lesion (7,285 voxels, `case_056`) at 1/4 or 1/8.
**That was wrong.** 7,285 voxels is a compact blob ~19 voxels across, still ~125
voxels after 4x downsampling. Volume-level burden was the wrong statistic;
per-slice *area* is what gets erased, which is why only the 2D model is affected.
Average pooling is kept on the narrower verified grounds that it is
mass-preserving by construction and empties nothing anywhere.

---

## 6. Training scheme

```
   60 cases
      │  fold_assignment(seed=20260817) — computed once, shared by baselines,
      │  2.5D and 3D so figure F09 compares like with like
      ▼
   ┌─────────┬─────────┬─────────┬─────────┬─────────┐
   │ fold 0  │ fold 1  │ fold 2  │ fold 3  │ fold 4  │   12 cases each
   └─────────┴─────────┴─────────┴─────────┴─────────┘

   for f in 0..4:
       train on the other 48 ──▶ predict the held-out 12
                                        │
                                        ▼
   every case scored exactly once, by a model that never saw it
   ── metrics are out-of-fold ONLY ──
```

Per-fold loop:

```
  for epoch in 0..E-1:
      for batch in train_loader:                  # random patches / slices, re-sampled each epoch
          augment(batch)
          with autocast(fp16):
              logits = model(batch)               # list of heads, training mode
              loss   = Σ wi · DiceBCE(...)
          if not isfinite(loss): skip step, log, continue   # see §7
          scaler.scale(loss).backward()
          unscale → clip_grad_norm(1.0) → scaler.step()
          onecycle.step()                         # per BATCH, not per epoch
      val_dice = mean hard_dice over val_loader   # eval mode → single tensor
      if val_dice improved:  save checkpoint;  patience := 20
      else:                  patience -= 1;  break at 0
```

### Steps per fold

| | samples/case/epoch | cases | batch | steps/epoch | epochs | steps/fold |
|---|---|---|---|---|---|---|
| 2.5D | 64 slices | 48 | 16 | 192 | 30 | 5,760 |
| 3D | 6 patches | 48 | 8 | 36 | 60 | 2,160 |

### Not done, deliberately

**The 5-fold ensemble is not the reported number.** Averaging all five models'
predictions for a held-out case would be leaked, not merely optimistic — four of
those five trained on that exact case. The ensemble is a real artifact worth
shipping for a genuinely new scan, but it has no honest Dice against these 60.

**TTA left-right flip is reported**, and is legitimate under strict out-of-fold
evaluation: it reuses only the same single model that never saw the case. It is
ablated with/without in F13 rather than applied silently.

---

## 7. Hyperparameters

### Optimisation (shared)

| knob | value | note |
|---|---|---|
| optimiser | AdamW | |
| `LR` | **1e-3** | OneCycle `max_lr`. Reduced from 3e-3 — see below |
| `WEIGHT_DECAY` | 1e-4 | |
| schedule | `OneCycleLR`, stepped per batch | `total_steps = steps_per_epoch × epochs` |
| loss | `DiceBCE`, `dice_weight = 0.5` | wrapped in `DeepSupervisionLoss` |
| AMP | fp16 autocast + `GradScaler` | CUDA only |
| grad clipping | `max_norm = 1.0` | applied after `unscale_` |
| `EARLY_STOP_PATIENCE` | 20 epochs | on val Dice (patch proxy) |
| `N_FOLDS` / `FOLD_SEED` | 5 / 20260817 | |

**Why LR is 1e-3 and not 3e-3:** 3e-3 produced near-total instability under fp16
autocast — from the LR-peak region onward almost every batch's forward pass
produced a non-finite loss (192/192 in some epochs). Gradient clipping does not
save you from this: clipping bounds a *finite* gradient, it cannot prevent a
peak-LR update from pushing weights into a regime whose activations overflow
fp16 on the next forward pass. Measured, not assumed — the isfinite-loss
skip-and-log in `train.py` is what surfaced it.

**Note on early stopping:** with `OneCycleLR` sized to the full epoch count, val
Dice tends to keep improving as the LR anneals toward zero, so patience-20 rarely
fires. Treat the projected schedule as the expected runtime, not an upper bound.

### Architecture

| | 2.5D | 3D |
|---|---|---|
| base width | 24 | 20 |
| depth | 5 | 4 |
| input channels | 20 (4 mod × 5 slices) | 4 |
| input geometry | 192 × 192 | 96 × 160 × 160 patch |
| batch size | 16 | 8 |
| epochs | 30 | 60 |
| samples/case/epoch | 64 | 6 |
| parameters | 18,286,804 | 8,959,163 |
| deep-supervision heads | 3 aux | 2 aux |

`SLICE_CONTEXT = 2` (±2 neighbours → 5-slice stack). `POS_FRACTION = 0.6` of
sampled slices are guaranteed to contain tumour; 46.1% of all slices are
positive, so this only mildly over-samples.

### Augmentation

At n = 60 (48 per training fold) this is not polish — it is most of what stands
between the model and memorising 48 brains. Ranges are deliberately mild: this
is MRI, and a 45° rotation or a 2× intensity swing would manufacture anatomy
that does not occur in vivo.

| knob | value |
|---|---|
| `AUG_FLIP_PROB` | 0.5, independent per axis |
| `AUG_ROTATE_DEG_2D` | ±15° in-plane |
| `AUG_ROTATE_DEG_3D` | ±10° about z only |
| `AUG_SCALE_JITTER` | ±0.10 |
| `AUG_INTENSITY_SCALE` | ±0.15 (contrast, in z-score space) |
| `AUG_INTENSITY_SHIFT` | ±0.10 z-score units (brightness) |
| `AUG_NOISE_SIGMA` | 0.08 z-score units |
| `AUG_BIAS_FIELD_MAGNITUDE` | ±0.15 multiplicative |
| `AUG_BIAS_FIELD_RES` | 4³ control grid, upsampled |

### Inference / post-processing

| knob | value |
|---|---|
| window | sliding, over the full 192×192×155 crop |
| `TTA_FLIP` | left-right flip, averaged with unflipped |
| `DEFAULT_THRESHOLD` | 0.5 |
| `MIN_COMPONENT_VOXELS` | 500 (connected components below this dropped) |
| `BOOTSTRAP_N` / seed | 10,000 / 7 (for confidence intervals) |

---

## 8. Budget and pre-registered targets

| | floor | stretch |
|---|---|---|
| Mean out-of-fold Dice, 60 cases | ≥ 0.85 | ≥ 0.90 |
| Median Dice | ≥ 0.87 | ≥ 0.90 |
| Worst-case Dice | ≥ 0.55 | ≥ 0.55 (unchanged) |
| Beats FLAIR-threshold baseline | Wilcoxon p < 0.01 | — |
| Training budget | ≤ 170 min (RTX 3090 Ti) | — |

Fixed in [`RULES.md`](../RULES.md) before any training. Deep supervision did not
move them — architecture was never pre-registered, only the targets, the budget,
the fold count and the target region.

### Baselines to beat (already scored, same fold splits)

| | mean Dice | median | min | max |
|---|---|---|---|---|
| FLAIR threshold | 0.680 | 0.735 | 0.106 | 0.917 |
| Random forest | 0.711 | 0.768 | 0.060 | 0.938 |

### Measured runtime

Measured on an idle 3090 Ti with deep supervision active.

| | probe (ms/step) | probe projection | measured |
|---|---|---|---|
| 2.5D, 5 folds | 29.5 | 14.2 min | **17.6 min** (no DS) |
| 3D, per epoch | 646.4 | — | **27.9 s** steady, 32.7 s epoch 0 |
| 3D, 5 folds | — | 116.4 min | **99.5 min** (213 epochs, all folds early-stopped) |

**The probe understates per-epoch wall-clock by ~1.19x** (2.5D independently gave
1.24x): it measures pure step time with no dataloader, augmentation or validation
pass, so `--dry-run` is a floor, not a forecast. The 5-fold *total* nevertheless
came in under the projection because early stopping cut every fold short.

Deep supervision's own cost is negligible: 646.4 ms/step with it against 648.7
without — inside run-to-run noise.

**Memory.** The probe reports 15.3 GB peak at batch 8; the real run peaks at
**23.02 GiB of 23.56 GiB usable**, ~540 MB of headroom, the gap being dataloader,
pinned host memory and augmentation. A 94 MB allocation from a second process
OOMed while training was in flight. Batch 8 is effectively the memory limit — run
nothing else on the GPU, including inference.

### Result (pre-registered, 2026-08-17)

| target | measured | floor | verdict |
|---|---|---|---|
| mean out-of-fold Dice | **0.8869** CI95 [0.8526, 0.9136] | 0.85 | PASS (stretch 0.90 missed) |
| median Dice | **0.9246** | 0.87 | PASS — clears stretch |
| worst-case Dice | **0.1390** | 0.55 | **FAIL** |
| beats FLAIR baseline | 0.6803 → 0.8869, p = 1.63e-11 | p<0.01 | PASS |

Zero non-finite batches across all 213 epochs. Inference: 2.0 min for 60 cases.

Worst case is `case_056`, the smallest lesion (7,285 voxels): **not a miss** —
81% recall — but 78,773 voxels predicted against 7,285 true, i.e. a precision
failure. Dice is flat at 0.11-0.15 across thresholds 0.1-0.7, so it is not a
threshold artifact either.

### Early stopping, and the defect it caused

Early stopping fires for 3D but not for 2.5D: all five 2.5D folds ran their full
30 epochs, while 3D stopped at epochs 51/33/56/30/38 selecting epochs
31/13/36/10/18. `OneCycleLR` peaks at epoch 18, so **three of five folds shipped
weights from at or before the LR peak** — mid-warmup, with none of the anneal
applied.

Two causes, both since fixed:

1. `PatchDataset3D` re-drew validation patches on every `__getitem__`, so "best
   epoch" was judged against a target that moved each epoch (same index returned
   foreground counts 66551/73207/73590/62677). Fixed by `frozen=True` on the
   validation datasets.
2. `EARLY_STOP_PATIENCE = 20` was sized for the 2.5D 30-epoch schedule; on 60
   epochs it ended folds before annealing could overturn a warmup fluke. Fixed by
   `EARLY_STOP_MIN_FRACTION`, which gates patience to the second half.

Annealed folds averaged 0.9119 against 0.8703 for the others (+0.0415), but
**p = 0.2230** — suggestive, not significant, on 5 folds with difficulty
confounded. Not retrospectively measurable either: only best-by-proxy weights
were saved, so there is nothing to A/B. `SAVE_FINAL_CHECKPOINT` now also keeps the
final epoch so this is answerable next time without a re-run.

---

## 9. Known open items

- **2.5D folds on disk predate deep supervision** and must be re-run (~20 min)
  before F13 compares them against the 3D model, or that figure conflates
  dimensionality with deep supervision — two changes, one comparison.
- **`docs/model_scope.md` §3 contradicts `config.py`** on whether the 5-fold
  ensemble is the reported number. `config.py` is correct (it would be leaked);
  model_scope.md needs the correction.
- **nnU-Net reference arm** is planned as a separate, explicitly-labelled
  experiment on fold 0 only, outside the 170-min budget. It cannot fit inside it:
  nnU-Net's epoch is a fixed 250 iterations regardless of dataset size, so its
  default 1000-epoch schedule is ~24 h/fold on this GPU. Running 2% of it and
  calling the result "nnU-Net" would be exactly the inadequate-baseline error
  that framework's own authors warn about.
</content>
</invoke>
