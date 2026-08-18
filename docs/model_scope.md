# Model scope: what changed when the 3090 Ti turned up

The research plan was sized against a 1-hour end-to-end budget on an M2 Max
MacBook, 32 GB unified memory. That constraint explicitly ruled out 3D
convolutions: *"a single fold would eat the hour"*.

Partway through Act III it turned out a 24 GB RTX 3090 Ti was the actual
machine running this code (confirmed with `nvidia-smi`, not assumed), with a
3-hour training budget instead of 1. That is roughly a 50-100x increase in
usable compute (3x the wall-clock, and CUDA + a discrete 24 GB GPU is easily
5-15x an M2 Max's MPS throughput for this kind of workload).

## What this changes

1. **3D becomes the primary architecture.** The 2.5D U-Net (4 modalities x 5
   adjacent slices stacked as 20 input channels) stays in the codebase — it
   trains in minutes and is now the *comparison point*, not the deliverable.
   The 3D U-Net operates on real volumetric patches and can use through-plane
   context the 2.5D stack only approximates.
2. **5-fold CV instead of 3-fold.** Every one of the 60 cases still gets an
   out-of-fold prediction; each fold now trains on 48 cases instead of 40.
3. **TTA is affordable as the reported number. Ensembling is not.** Average
   each fold model's sigmoid output against its own left-right flip; that is
   legitimate under strict out-of-fold evaluation, because it reuses only the
   *same* single model that never saw the case. It is ablated against "no TTA"
   in figure F13 rather than applied silently.

   Averaging all five fold models' predictions for a held-out case is a
   different matter and is **not** reported: four of those five trained on that
   exact case, so the resulting Dice would be leaked, not merely optimistic.
   The 5-fold ensemble is still a real artifact — worth shipping for a
   genuinely new scan — it just has no honest Dice number against these 60
   cases. An earlier version of this document said the opposite and committed,
   in writing, to reporting the ensemble as the headline; `ml/config.py` had
   the correct reasoning all along and this section is the one that was wrong.
4. **Real AMP.** CUDA's mixed precision is mature; MPS's is not (the original
   plan explicitly avoided it for that reason). Free throughput, spent on
   patch size and fold count rather than banked.
5. **Timing is measured, not estimated.** The original plan hand-computed
   GFLOPs-per-step because there was no way to run the model on the target
   Mac in advance. That reasoning doesn't transfer to a machine sitting right
   here — `ml/model.py`'s timing probe runs real steps and reports a real
   number.

## What did not change

- **n = 60.** More compute does not buy a bigger cohort. The worst-case Dice
  floor in `RULES.md` is left untouched for exactly this reason — a few
  genuinely hard cases are a sample-size fact, not a compute-budget fact.
- **The floor target (mean Dice >= 0.85).** Kept as the pre-registered bar. A
  stretch target (0.90) was added *before* training, not raised in place of
  the floor — see `RULES.md` for both numbers and how they get reported.
- **The baselines.** FLAIR threshold and per-voxel random forest are still
  scored first, on the same fold splits as everything else, so a number to
  beat exists in advance regardless of which model architecture wins.
