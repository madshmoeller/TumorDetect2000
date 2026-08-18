# Rules for Claude Code


- Ask Claude for its plan before it writes files.
- Never accept a number you have not seen a figure for.
- Decide what a good result looks like before you produce one.
- Report what you found, not what you hoped for.

---

## Pre-registered, before any training (rule 3)

Fixed 2026-08-17. If these are missed, the targets do not move (rule 4).

Revised same day, still before any deep-model training run: a 24 GB RTX 3090 Ti
turned out to be this machine, with a 3-hour training budget instead of a
1-hour MacBook one. That is roughly a 50-100x compute increase (5-fold instead
of 3-fold CV, 3D instead of 2.5D-only, real AMP, ensembling). n = 60 does not
change, so the floor stays put — but a stretch goal is added rather than
silently raising the bar:

| | floor | stretch |
|---|---|---|
| Mean out-of-fold Dice, 60 cases | ≥ 0.85 | ≥ 0.90 |
| Median Dice | ≥ 0.87 | ≥ 0.90 |
| Worst-case Dice | ≥ 0.55 | ≥ 0.55 (unchanged — outliers don't move with more compute) |
| Beats FLAIR-threshold baseline | paired Wilcoxon, p < 0.01 | — |
| Training budget | ≤ 60 min (M2 Max) | ≤ 180 min (RTX 3090 Ti, this machine) |

Both numbers get reported regardless of which is hit. Baselines (FLAIR
threshold, random forest) are scored **before** the deep model, on the same
fold splits, so the number to beat exists in advance either way.

## How the rules bind here

- **Rule 2** — every reported number traces to a figure in `assets/figures/`,
  collected on `about.html`. No figure, no number.
- **Rule 4** — all 60 cases are browsable on the site, failures included. The
  ablation figure is allowed to conclude the deep model wasn't worth it.
  Quoted runtimes are measured, not estimated.
- **n = 60**, and it is small. Confidence intervals accompany every point
  estimate. Metrics are out-of-fold only; fold membership is asserted disjoint.
- Target is binary whole-lesion (`label > 0`) — the *whole tumour* region, the
  easiest of the three. Not comparable to published core/enhancing scores, and
  the dataset's provenance is withheld by the course until the lab ends.

