# Data, weights, and how to get everything

## The dataset — a submodule, not a copy

The imaging data lives in its own public upstream repository and is referenced
here as a git submodule pinned to an exact commit:

```bash
git clone --recurse-submodules <this repo>
# or, in an existing clone:
git submodule update --init
```

That populates `./agentic-medical-ai-lab/`, which is where `ml/config.py` expects
it. You get the same 60 labelled cases and the same 60 eval cases this project
was built against, at the same commit, without this repository redistributing a
single byte of them.

**Why a submodule rather than a copy.** The data carries its own licence, stated
in `data/cases/dataset.json`:

> CC-BY-SA 4.0 — source, citation and full attribution published with the course
> material after the lab

The upstream repo gitignores `ATTRIBUTION.md` with the note *"held back until the
lab is over"*, so the attribution CC-BY-SA requires has not been published yet.
A submodule sidesteps the question cleanly: the data is distributed by its own
licensor from its own repository, and this project only records which commit it
used. Cite it from the upstream attribution once that is released.

The upstream repo's own MIT licence covers its software, not its data.

## The eval cohort

`agentic-medical-ai-lab/data/eval/` holds 60 evaluation cases, **images only** —
*"no masks are published for these"* (`data/eval/tier_eval.txt`). They are the
test inputs of a classroom Kaggle competition.

Our predictions for them **are** tracked, in `outputs/kaggle_eval/`: 60
`*_mask.nii.gz` volumes plus `submission.csv` (run-length encoded, verified to
round-trip losslessly for all 60 cases). No ground-truth masks exist for these
cases, so nothing here can be scored against them locally.

> A correction worth recording. An earlier version of this file excluded the eval
> set and our predictions on the grounds that publishing them would leak a live
> competition. That was wrong: the upstream repository is **public** — an
> anonymous `git ls-remote` with credentials disabled resolves it — so every
> participant already has those images by cloning the course repo. Republishing
> leaks nothing.

## Model weights

The **deployed 5-fold 3D set is tracked**: `outputs/checkpoints/3d_fold{0..4}.pt`,
34 MB each, so this repository runs inference out of the box with no training.

Not tracked, for size: the 2.5D checkpoints (70 MB each) and the per-arm archives
under `outputs/overnight/*/checkpoints/`, which duplicate the deployed set.
Everything needed to *audit* them is tracked — `outputs/**/*.json` holds every
reported number, per case and per fold.

## Also not tracked, all regenerable

| path | size | rebuild with |
|---|---|---|
| `ml/cache/` | 3.2 GB | `python -m ml.data --build` |
| `assets/cases/` | 819 MB | `python -m ml.export_web --full --arch 3d` then `python -m ml.export_web_eval` |
| `nnunet_tumordetect/` | 1.4 GB | `scripts/nnunet_setup.sh`, `scripts/nnunet_train.sh` |

The website works with `assets/cases/` absent — the viewer falls back to a
scripted demo, which is a documented path rather than an error.

## From a clean clone to a result

```bash
git clone --recurse-submodules <this repo> && cd tumordetect
python -m pip install -r requirements.txt

# inference only, using the tracked weights — no training needed
python -m ml.predict --input-dir agentic-medical-ai-lab/data/eval/images \
                     --output-dir outputs/testset

# or reproduce the whole thing
python -m ml.data --audit          # assert the dataset facts this project relies on
python -m ml.data --build          # preprocessing cache
python -m ml.train --dry-run       # measured probe -> projected schedule
python -m ml.train --arch 3d       # 5-fold CV, ~100 min on a 24 GB RTX 3090 Ti
python -m ml.infer  --arch 3d      # out-of-fold Dice
python -m ml.export_web --full     # website assets
```

## Inference on your own scans

```bash
python -m ml.predict --input-dir <DIR> --output-dir outputs/testset
```

One 4-modality `.nii.gz` per case, channels last, ordered **FLAIR, T1w, T1-Gd,
T2w**. Channel order is silent if wrong — the model will produce confident
nonsense — so verify it before trusting output. Add `--labels-dir` to score
against ground truth if you have it. Full procedure in
[docs/RUNBOOK_TESTSET.md](docs/RUNBOOK_TESTSET.md).
