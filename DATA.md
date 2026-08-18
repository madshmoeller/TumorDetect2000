# Getting the data

This repository ships **no imaging data**. Two separate reasons, neither of them
about repository size.

## 1. The training cohort — attribution is withheld

The 60 labelled cases come from a separate upstream repository:

```bash
git clone https://github.com/kimmouridsen-cloud/agentic-medical-ai-lab
```

This project expects it at `./agentic-medical-ai-lab/` (see `ml/config.py`).

That repo's MIT licence covers its software. The imaging data carries its own
terms, stated in `data/cases/dataset.json`:

> CC-BY-SA 4.0 — source, citation and full attribution published with the course
> material after the lab

CC-BY-SA **requires** attribution, and the attribution has not been published
yet: the upstream repo gitignores `ATTRIBUTION.md` with the note *"held back
until the lab is over"*. Redistributing the data from here would therefore break
the licence we were given it under. Clone it from the source instead, and cite it
from the upstream attribution once that is released.

## 2. The eval cohort — publishing it would leak a live competition

`data/eval/` holds 60 evaluation cases, **images only**: *"no masks are published
for these"* (`data/eval/tier_eval.txt`). They are the test inputs of a private
classroom Kaggle competition that was still running when this was written.

Excluded from this repo, on fairness grounds:

- the eval NIfTI volumes,
- the browsable PNG atlases the web exporter renders from them,
- our own predicted masks and `submission.csv`.

Any of those hands another participant either the test set or a finished answer.
`.gitignore` matches these by content pattern rather than by directory name,
because an earlier pass excluded one directory and a differently-named sibling
carrying the same files walked straight past it.

## Rebuilding everything from a clean clone

```bash
git clone https://github.com/kimmouridsen-cloud/agentic-medical-ai-lab
python -m pip install -r requirements.txt

python -m ml.data --audit          # assert the dataset facts this project relies on
python -m ml.data --build          # preprocessing cache (~3.2 GB, regenerable)
python -m ml.train --dry-run       # measured probe -> projected schedule
python -m ml.train --arch 3d       # 5-fold CV, ~100 min on a 24 GB RTX 3090 Ti
python -m ml.infer  --arch 3d      # out-of-fold Dice
python -m ml.export_web --full     # website assets (~819 MB, regenerable)
```

Model weights are also not tracked — 42 checkpoints exceed GitHub's 50 MB warning
and two exceed the 100 MB hard limit. Every **number** this project reports is
tracked, in `outputs/**/*.json`, so results stay auditable without the weights.
Attach the deployed 5-fold set to a Release if it needs to ship.

## Inference on new scans

```bash
python -m ml.predict --input-dir <DIR> --output-dir outputs/testset
```
Expects one 4-modality `.nii.gz` per case, channels last, ordered
**FLAIR, T1w, T1-Gd, T2w**. See `docs/RUNBOOK_TESTSET.md`.
