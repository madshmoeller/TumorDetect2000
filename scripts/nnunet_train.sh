#!/usr/bin/env bash
# nnU-Net reference arm: plan, preprocess, force OUR folds, train fold 0.
#
# This is an external reference, not the deliverable, and it is not comparable to
# the pre-registered 170 min budget — nnU-Net's epoch is a fixed 250 iterations
# regardless of dataset size, so its default 1000-epoch schedule is ~24 h/fold on
# this GPU. We run a reduced schedule on ONE fold and label it as reduced. A
# truncated nnU-Net reported as "nnU-Net" would be the inadequate-baseline error
# that framework's own authors wrote a paper about.
set -u
ROOT=/home/mads/tumordetect
VENV=$ROOT/.venv-nnunet
PY=$VENV/bin/python
NN=$ROOT/../nnunet_tumordetect

# Hard wall: leave the morning clear. nnU-Net checkpoints every 50 epochs, so a
# kill here still leaves a usable partial model plus its training log.
STOP_AT="07:30"

export nnUNet_raw=$NN/nnUNet_raw
export nnUNet_preprocessed=$NN/nnUNet_preprocessed
export nnUNet_results=$NN/nnUNet_results
DATASET=501

[ -x "$PY" ] || { echo "no venv at $VENV — nnunet_setup stage must have failed"; exit 1; }
[ -d "$nnUNet_raw/Dataset501_TumorWT" ] || { echo "raw data missing — nnunet_prep stage must have failed"; exit 1; }

echo "raw:          $nnUNet_raw"
echo "preprocessed: $nnUNet_preprocessed"
echo "results:      $nnUNet_results"

echo "=== plan and preprocess ==="
"$VENV/bin/nnUNetv2_plan_and_preprocess" -d $DATASET --verify_dataset_integrity -c 3d_fullres \
  || { echo "plan_and_preprocess failed"; exit 1; }

# MUST happen after preprocessing (which creates the directory) and before
# training (which writes its own splits_final.json if none exists). Without this
# nnU-Net invents its own 5-fold split and its fold 0 is a different 12 cases
# than our baselines and models were scored on, silently breaking F09.
echo "=== forcing our fold assignment (seed 20260817) ==="
cd "$ROOT" && python3 -m ml.export_nnunet splits || { echo "splits step failed"; exit 1; }
echo "splits_final.json:"; head -c 300 "$nnUNet_preprocessed/Dataset501_TumorWT/splits_final.json"; echo

echo "=== printing nnU-Net's chosen plan (this is a finding in itself) ==="
"$PY" - <<PY
import json, os
p = os.path.join(os.environ["nnUNet_preprocessed"], "Dataset501_TumorWT", "nnUNetPlans.json")
d = json.load(open(p))
c = d["configurations"]["3d_fullres"]
print("  patch_size      ", c["patch_size"])
print("  batch_size      ", c["batch_size"])
print("  spacing         ", c.get("spacing"))
print("  n_stages        ", c["architecture"]["arch_kwargs"].get("n_stages"))
print("  features_per_stage", c["architecture"]["arch_kwargs"].get("features_per_stage"))
print("  normalization   ", d.get("foreground_intensity_properties_per_channel", {}).keys())
PY

# READ THIS BEFORE QUOTING ANY NUMBER FROM THIS RUN.
#
# 50 epochs = 12,500 iterations: 5% of nnU-Net's default 1000-epoch schedule.
# This is NOT a fair nnU-Net and cannot answer "is our model leaving anything on
# the table?". A fair answer needs the default schedule, which nnU-Net's own
# fixed 250-iterations-per-epoch makes ~24 h per fold on this GPU — days for five
# folds, against a 170 min pre-registered budget for the whole project.
#
# 50 was chosen over 250 for one specific reason. The window here is ~2 h, and
# 250 epochs would be killed by the 07:30 guard at roughly epoch 72 — stopped
# mid-anneal, with its poly-LR schedule sized for a run that never finished. That
# is exactly the checkpoint-selection defect this project spent tonight fixing in
# its own model; reproducing it in the reference arm would make the comparison
# worse, not better. A short schedule that COMPLETES its anneal is a coherent
# model. A long one that gets killed is not.
#
# So what this arm is actually for:
#   1. proving the whole nnU-Net chain runs here, so a fair multi-day run later
#      is one command away rather than an unknown;
#   2. a labelled LOWER BOUND on nnU-Net's performance on this cohort.
# It is not the reference comparison, and must never be reported as "nnU-Net".
TRAINER=nnUNetTrainer_50epochs
echo "=== training fold 0 with $TRAINER (hard stop $STOP_AT) ==="

"$VENV/bin/nnUNetv2_train" $DATASET 3d_fullres 0 -tr $TRAINER --npz &
TRAIN_PID=$!
echo "nnUNetv2_train PID $TRAIN_PID"

while kill -0 $TRAIN_PID 2>/dev/null; do
  now=$((10#$(date +%H%M))); cut=$((10#$(echo $STOP_AT | tr -d ':')))
  if [ "$now" -ge "$cut" ] && [ "$((10#$(date +%H)))" -lt 12 ]; then
    echo "reached $STOP_AT — stopping nnU-Net so the GPU is free for the test set"
    kill $TRAIN_PID 2>/dev/null; sleep 20; kill -9 $TRAIN_PID 2>/dev/null
    echo "PARTIAL: nnU-Net stopped early; latest checkpoint and log remain under $nnUNet_results"
    break
  fi
  sleep 60
done
wait $TRAIN_PID 2>/dev/null
echo "=== nnU-Net stage finished at $(date +%H:%M:%S) ==="
find "$nnUNet_results" -name "*.pth" -o -name "progress.png" -o -name "training_log*.txt" 2>/dev/null | head
