# Pre-registered 3D result — 2026-08-17, DO NOT OVERWRITE

Produced by the run started 16:56:33 and finished 18:36:06 on 2026-08-17.
This is the result the RULES.md targets were pre-registered against.

    3D training wall clock   99.5 min measured   (ceiling 170 min)
    mean out-of-fold Dice    0.8869  CI95 [0.8526, 0.9136]   floor 0.85  PASS
    median Dice              0.9246                          floor 0.87  PASS (stretch too)
    worst-case Dice          0.1390                          floor 0.55  FAIL
    vs FLAIR threshold       0.6803 -> 0.8869  Wilcoxon p = 1.63e-11    PASS
    nan-loss batches         0 across 213 epochs

Known defects in THIS run, disclosed rather than fixed retroactively:
  - 3 of 5 folds (epochs 10, 13, 18) selected checkpoints at or before the
    OneCycleLR peak at epoch 18, i.e. with none of the anneal applied.
  - Cause: validation patches were re-drawn every epoch (PatchDataset3D
    resamples per __getitem__), so "best epoch" was judged against a moving
    target, and EARLY_STOP_PATIENCE=20 — sized for the 2.5D 30-epoch schedule —
    then ended folds before annealing could overturn a warmup fluke.
  - Not retrospectively measurable: only best-by-proxy checkpoints were saved.
  - DS_TARGET_POOLING was "avg", whose justification was twice wrong; the
    evidence now favours "nearest". See ml/config.py.

The overnight run of 2026-08-18 fixes these and is a SEPARATE, LABELLED arm.
It does not replace the numbers above.
