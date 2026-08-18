# TumorNet 2000 ™

Brain tumour segmentation on 60 MRI cases, wrapped in a deliberately over-the-top
late-90s "virtual detection lab" website.

The joke is the interface. The model behind it is real, cross-validated, and
reported against targets that were written down before any of it was trained.

```
   4 MRI channels          3D residual U-Net              binary tumour mask
   FLAIR T1w T1-Gd T2w  ──▶  + deep supervision   ──▶     (whole-lesion)
   240×240×155 @ 1mm          5-fold ensemble
```

## Results

Out-of-fold on all 60 cases — every case predicted by a model that never saw it.

| | mean Dice | median | worst case |
|---|---|---|---|
| FLAIR-threshold baseline | 0.680 | 0.735 | 0.106 |
| voxel-wise random forest | 0.711 | 0.768 | 0.060 |
| **3D U-Net + deep supervision** | **0.905** | **0.928** | **0.612** |

Beats both baselines at paired Wilcoxon *p* = 1.6e-11. Training: 99.5 min
measured on one 24 GB RTX 3090 Ti.

**The pre-registered figure was 0.887, and it failed the worst-case floor at
0.139.** The 0.905 above comes from a later run that fixed a genuine
checkpoint-selection defect, and is reported as a separate labelled arm rather
than backfilled over the original. Both are kept:
`outputs/prereg_20260817/README.md` records what changed and why.

## The rules this was built under

From [RULES.md](RULES.md), fixed at the start:

1. Ask for the plan before writing files.
2. Never accept a number you have not seen a figure for.
3. Decide what a good result looks like before you produce one.
4. Report what you found, not what you hoped for.

Rule 3 is why the targets in RULES.md are dated and were revised only *before*
training. Rule 4 is why the failed worst-case floor is in the README rather than a
footnote, and why `docs/architecture.md` documents three mechanisms that were
confirmed in isolation and then turned out not to hold in the real configuration.

## Layout

```
ml/                  the pipeline
  config.py            single source of truth for every constant
  data.py              loading, per-case z-score, crop, cache
  datasets.py          patch/slice samplers  (val sets are frozen — see why inside)
  model.py             2.5D + 3D residual U-Nets, deep supervision, timing probe
  losses.py            Dice+BCE, deep-supervision wrapper
  augment.py           one fused affine + intensity/bias-field jitter
  train.py             5-fold CV
  infer.py             OUT-OF-FOLD scoring of the 60 dev cases (needs labels)
  predict.py           inference on NEW unlabelled scans (5-fold ensemble)
  deploy.py            compare arms, install the best, publish to the site
  export_web.py        website atlases for the labelled cohort
  export_web_eval.py   ...and for unlabelled cases
  export_nnunet.py     nnU-Net reference-arm conversion, with OUR fold splits
  baselines.py metrics.py figures.py eda.py

docs/
  model_overview.md    how a U-Net works, no background assumed, ASCII diagrams
  architecture.md      technical reference: shapes, hyperparameters, measurements
  model_scope.md       why the compute decisions went the way they did
  RUNBOOK_TESTSET.md   procedure for running a fresh test set

scripts/overnight.sh   unattended multi-arm run, deadline-guarded
outputs/**/*.json      every reported number, per case and per fold
assets/figures/        the figures behind those numbers (rule 2)
*.html src/            the website
```

## Running it

```bash
python -m pip install -r requirements.txt
python serve.py                    # → http://127.0.0.1:8000/
```

The site works with no data present — the viewer falls back to a scripted demo,
which is a documented path, not an error. To drive it with the real model you need
the dataset: see **[DATA.md](DATA.md)**, which also explains why no imaging data,
predicted mask, or model weight is tracked here.

## Honest limitations

- **n = 60.** Every metric carries a confidence interval; a small cohort bounds
  what anyone is entitled to conclude from it.
- **Binary whole-lesion target only** — not comparable to published tumour-core
  or enhancing-tumour scores.
- **Provenance withheld** by the course until the lab ends. Nothing here should be
  compared against a named public benchmark as though it were the same test set.
- **`confidence` in prediction output is not calibrated.** It is the mean
  predicted probability inside the retained mask, and the JSON says so in a field.

## Disclaimer

This diagnoses nothing. It is a design demo with a cartoon brain in the corner.
Please consult an actual doctor.
