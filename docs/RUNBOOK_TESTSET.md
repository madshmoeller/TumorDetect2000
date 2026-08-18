# Runbook — test set, morning of 2026-08-18

Written the night before, for someone who has just woken up. Commands are
copy-pasteable from the repo root. Nothing here needs the overnight run to have
succeeded; step 0 tells you what you actually have.

---

## 0. First: what happened overnight? (1 min)

```bash
cd /home/mads/tumordetect
tail -40 outputs/overnight/driver.log      # per-stage OK / FAIL / SKIP
python3 -m ml.deploy --report              # every arm, scored, best flagged
cat outputs/deploy_manifest.json           # what got installed, if anything
```

`driver.log` has one `=== OK ===` or `=== FAIL ===` line per stage. Stages are
fault-isolated, so a failure part-way does not invalidate what came before.

### ⚠ Do this before anything else unless the `deploy` stage shows `=== OK ===`

While the overnight run is training, `outputs/checkpoints/` holds a **mix**: the
fold currently being trained has already been overwritten, while the others are
still from the previous arm. A mixed ensemble is not a model — it is five
unrelated networks averaged together, and it will produce plausible-looking
nonsense rather than an error.

`grep deploy outputs/overnight/driver.log`. If it does not show `=== OK deploy ===`,
install a coherent set explicitly before predicting anything:

```bash
python3 -m ml.deploy --report                      # see what completed
python3 -m ml.deploy --install prereg_20260817     # known-good 5-fold set, mean 0.8869
# ...or a completed overnight arm, e.g.:
python3 -m ml.deploy --install 3d_fixed_avg
```

The pre-registered model is always intact at `outputs/prereg_20260817/`, so this
is never unrecoverable.

---

## 1. Verify the inference path before trusting it (2 min)

**First check the GPU is actually free.** Training peaks at ~23.0 GiB of 23.6 GiB,
so if anything is still running there is no room and inference will fail:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

Expect no rows. If the overnight run overran its 07:45 guard, stop it:

```bash
pkill -f 'ml.train'; sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

`ml.predict` checks this itself and refuses with an explicit message rather than
an opaque `CUDA error: out of memory`. A CPU fallback exists but takes minutes per
case: prefix with `CUDA_VISIBLE_DEVICES=''`.

Then:

```bash
python3 -m ml.predict --self-test
```

Must print `PASS: test-set inference path works end-to-end`. It runs three known
cases through the *real* test-set code path (raw NIfTI → brain mask → z-score →
crop → 5-fold ensemble + TTA → postprocess → NIfTI out) and checks shapes and
files. The Dice it prints is **leaked and meaningless as a score** — four of the
five ensembled folds trained on those cases. It is a plumbing check only.

If it fails, do not proceed; read the traceback. Most likely cause is missing or
half-written checkpoints in `outputs/checkpoints/` — fix with the
`--install prereg_20260817` command above.

---

## 2. Run the test set (~5 s per case)

```bash
python3 -m ml.predict --input-dir /path/to/testset --output-dir outputs/testset
```

Expects one 4-modality `.nii.gz` per case, channels last, in the same modality
order as training: **FLAIR, T1w, T1-Gd, T2w**. Confirm that order before
trusting any output — the channel order is silent if wrong and the model will
produce confident nonsense.

Outputs into `outputs/testset/`:

| file | what |
|---|---|
| `<case>_mask.nii.gz` | binary prediction, original 240×240×155 geometry, input affine |
| `<case>.npz` | same mask in `ml/infer.py`'s format, so the web export can publish it |
| `predictions.json` | per case: `volume_mm3`, `confidence`, `verdict`, `n_components`, timings, and any failures |

It is deliberately robust for unattended use: a malformed case is recorded under
`failures` and the batch continues. **Check `n_failed` in `predictions.json`.**

### If the geometry differs from training
`predictions.json` flags it per case. `nonstandard_geometry: true` means the
volume was not 240×240×155; `crop: "none (full volume)"` means the canonical crop
would have clipped brain voxels so the full volume was used instead. Both are
handled (the model is fully convolutional and inference is a sliding window), but
they are worth knowing before quoting results.

### Useful variants
```bash
# single fold instead of the ensemble (faster, weaker; for debugging)
python3 -m ml.predict --input-dir DIR --output-dir OUT --no-ensemble
# the 2.5D model
python3 -m ml.predict --input-dir DIR --output-dir OUT --arch 25d
# no test-time augmentation
python3 -m ml.predict --input-dir DIR --output-dir OUT --no-tta
```

---

## 3. If the test set ships ground-truth labels

Then you can score it, and **this score is honest** — unlike the development
cases, a real test set was in nobody's training fold, so the 5-fold ensemble is
legitimate rather than leaked.

```bash
python3 -m ml.predict --input-dir /path/to/testset/images \
                      --labels-dir /path/to/testset/labels \
                      --output-dir outputs/testset
```

Adds `dice`, `hd95`, `precision`, `recall` per case plus a cohort summary with a
bootstrap CI to `predictions.json`.

**Do not compare that number to the pre-registered 0.8869 as though they measure
the same thing.** They do not: 0.8869 is out-of-fold single-model on the 60
development cases; a test-set score is 5-fold-ensemble on unseen cases. The
ensemble alone should be worth something. Report them side by side, labelled.

---

## 4. Publish to the website

```bash
python3 -m ml.deploy --auto          # picks the best arm, installs it, exports assets
python3 serve.py                     # http://127.0.0.1:8000/
```

`--auto` merges predictions into `assets/cases/manifest.json`, filling the
`prediction` mask slots and the `dice`/`hd95`/`predictedVolumeMm3` fields that are
currently `null`. As of tonight the site is **ground-truth-only** — this is the
first time it will show model output.

To publish a specific arm instead of the best:
```bash
python3 -m ml.deploy --install 3d_nearest
```

---

## 5. What must NOT be claimed

- **The reported result is the pre-registered one: mean out-of-fold Dice
  0.8869, CI95 [0.8526, 0.9136], median 0.9246, worst case 0.1390.** Floor 0.85
  met, stretch 0.90 missed, median clears its stretch, both baselines beaten at
  p = 1.63e-11. Worst-case floor of 0.55 **FAILED**.
- Deploying a better overnight arm does **not** change that. `deploy_manifest.json`
  records the deployed arm and the pre-registered arm separately for this reason.
  Overnight arms are labelled, post-hoc, and fix defects found after the fact —
  legitimate to deploy and to report *as a separate arm*, not to substitute.
- `confidence` in `predictions.json` is the mean predicted probability inside the
  retained mask. It is **not calibrated** and is not a probability that a tumour
  exists. The field `confidence_is_calibrated: false` says so in the data.
- The patch-proxy `val_dice` in training logs is not the reported metric and reads
  optimistic. Use `infer_*.json` / `predictions.json` numbers only.

## Known defects, disclosed

- **Worst-case Dice 0.1390 on `case_056`** (smallest lesion, 7,285 voxels). Not a
  miss — 81% recall — but massive over-segmentation: 78,773 voxels predicted,
  72,843 of them false positives. A precision failure on the smallest lesion.
- **The pre-registered run selected 3 of 5 checkpoints at or before the LR peak**
  (epochs 10, 13, 18), i.e. with none of OneCycle's anneal. Cause: validation
  patches were re-drawn every epoch, so "best epoch" chased a moving target, and
  `EARLY_STOP_PATIENCE=20` (sized for the 2.5D 30-epoch schedule) then ended folds
  early. Both fixed for the overnight arms; see `outputs/prereg_20260817/README.md`.
- **`DS_TARGET_POOLING="avg"`** was justified twice on reasoning that turned out to
  be wrong. Evidence now favours `"nearest"`. The overnight run ablates both.
