#!/usr/bin/env bash
# Unattended overnight run — 2026-08-18, ~00:55 to a hard 07:45 stop.
#
# Ground rules this script is built around:
#
#  1. NOTHING here may overwrite the pre-registered result. That lives in
#     outputs/prereg_20260817/ and every stage below archives its own artifacts
#     into outputs/overnight/<tag>/ instead of leaving them in outputs/.
#  2. Every stage is fault-isolated. A stage that fails logs and is skipped; the
#     rest still run. There is nobody awake to fix a crash.
#  3. A hard deadline guard. No new stage starts after $DEADLINE, so the machine
#     is idle and ready for the test set in the morning.
#  4. Stages are ordered by value, so if the deadline truncates the tail we lose
#     the least important work. nnU-Net is last for exactly that reason.
#
# Run:  nohup bash scripts/overnight.sh > outputs/overnight/driver.log 2>&1 &

set -u
cd /home/mads/tumordetect

ROOT=/home/mads/tumordetect
OUT=$ROOT/outputs
ON=$OUT/overnight
DEADLINE="07:45"                 # no NEW stage after this
mkdir -p "$ON"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
mins() { echo "$(( ($(date +%s) - $1) / 60 ))"; }

# 10# forces base 10. Without it bash reads "0800" as an octal literal, which is
# invalid (8 is not an octal digit) and would make the guard error out at exactly
# the hour it exists to handle.
past_deadline() {
  local now=$((10#$(date +%H%M)))
  local cut=$((10#$(echo "$DEADLINE" | tr -d ':')))
  [ "$now" -ge "$cut" ] && [ "$((10#$(date +%H)))" -lt 12 ]
}

# Archive a finished stage's artifacts.
#
# JSON results are MOVED (each stage must produce its own, and an inherited
# stale file would be read as this stage's result). Checkpoints are COPIED, so
# $OUT/checkpoints is never left empty — if this whole script dies at 3am, the
# morning still has a loadable model at the canonical path. ml/deploy.py selects
# the real winner from these archives afterwards.
archive() {
  local tag=$1; local d=$ON/$tag
  mkdir -p "$d/checkpoints"
  for f in train_history_3d.json train_history_25d.json infer_3d.json infer_25d.json; do
    [ -f "$OUT/$f" ] && mv "$OUT/$f" "$d/"
  done
  for f in "$OUT"/checkpoints/*.pt; do [ -e "$f" ] && cp "$f" "$d/checkpoints/"; done
  [ -d "$OUT/predictions" ] && cp -r "$OUT/predictions" "$d/" 2>/dev/null
  log "archived -> $d  ($(ls "$d"/checkpoints 2>/dev/null | wc -l) checkpoints)"
}

# stage <tag> <description> <command...>
stage() {
  local tag=$1; shift
  local desc=$1; shift
  if past_deadline; then log "SKIP $tag ($desc) — past $DEADLINE deadline"; return; fi
  local t0=$(date +%s)
  log "=== START $tag : $desc ==="
  if "$@" > "$ON/$tag.log" 2>&1; then
    log "=== OK    $tag  ($(mins $t0) min) ==="
  else
    log "=== FAIL  $tag  ($(mins $t0) min) — see $ON/$tag.log ==="
    tail -15 "$ON/$tag.log" | sed 's/^/      /'
  fi
}

train() { python3 -u -m ml.train --arch "$1"; }
infer() { python3 -u -m ml.infer --arch "$1"; }

log "###### overnight run start — deadline $DEADLINE ######"
log "GPU: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader)"
log "pre-registered result is archived and read-only at $OUT/prereg_20260817/"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Move the pre-registered run's JSON out of the working paths so no stage can be
# credited with it. Its checkpoints are deliberately LEFT in place as the
# fallback model until a stage produces better ones — a full copy is already
# safe in outputs/prereg_20260817/.
mkdir -p "$ON/_prereg_json_moved_aside"
for f in train_history_3d.json infer_3d.json; do
  [ -f "$OUT/$f" ] && mv "$OUT/$f" "$ON/_prereg_json_moved_aside/"
done
log "pre-registered JSON moved aside; its checkpoints left as morning fallback"

# ── STAGE 1 — the point of the whole night ─────────────────────────────────────
# 3D, deep supervision, avg pooling, WITH the checkpoint-selection fixes
# (frozen validation set + patience gated to the second half of the schedule).
# Same everything else as the pre-registered run, so the comparison isolates
# exactly the defect. Earliest possible stop is now epoch 50, so expect ~130 min.
export TUMORNET_DS_POOLING=avg
stage 3d_fixed_avg   "3D 5-fold, selection fixes, avg pooling"  train 3d
stage 3d_fixed_avg_infer "out-of-fold inference for the above"  infer 3d
archive 3d_fixed_avg

# ── STAGE 2 — F13 needs a 2.5D arm and there is currently none on disk ────────
# The peer deleted the old 2.5D artifacts. This one also carries deep
# supervision, so F13 compares 2.5D-vs-3D rather than conflating dimensionality
# with deep supervision.
stage 25d_fixed_avg  "2.5D 5-fold, selection fixes, avg pooling" train 25d
stage 25d_fixed_avg_infer "out-of-fold inference for the above"  infer 25d
archive 25d_fixed_avg

# ── STAGE 3 — the cheap, high-information ablation ────────────────────────────
# 2.5D with nearest (binary) DS targets. Two questions at once:
#   (a) does it reproduce the epoch-21 nan cascade that avg pooling produced?
#       If nearest survives, the mis-specified soft-target Dice is implicated.
#   (b) the avg-vs-nearest ablation itself, on the arm most exposed to it
#       (2.5D is the only architecture with a 1/8 aux head).
export TUMORNET_DS_POOLING=nearest
stage 25d_nearest    "2.5D 5-fold, nearest DS targets"          train 25d
stage 25d_nearest_infer "out-of-fold inference for the above"    infer 25d
archive 25d_nearest

# ── STAGE 4 — the ablation that matters for the deliverable ───────────────────
# 3D with nearest DS targets. If this beats stage 1, the reported model should
# change and we will have measured it rather than swapped a default on theory.
stage 3d_nearest     "3D 5-fold, nearest DS targets"            train 3d
stage 3d_nearest_infer "out-of-fold inference for the above"      infer 3d
archive 3d_nearest
unset TUMORNET_DS_POOLING

# ── STAGE 5 — pick the winner and make the site deployable ───────────────────
# Runs regardless of what the deadline truncated above: it compares whatever
# arms actually completed, installs the best one's checkpoints as canonical, and
# exports the website assets. Deliberately before nnU-Net so a nnU-Net failure
# cannot leave the morning without a deployable model.
stage deploy "select best arm, install checkpoints, export web assets" \
      python3 -u -m ml.deploy --auto

# ── STAGE 6 — nnU-Net reference arm, last because it is the most fragile ──────
# Needs a network install into an isolated venv; if any of that fails at 4am it
# must cost nothing else. Explicitly NOT the deliverable and not comparable to
# the pre-registered budget — see ml/export_nnunet.py.
stage nnunet_prep "convert cohort to nnU-Net raw format" \
      python3 -u -m ml.export_nnunet raw --overwrite
stage nnunet_venv "create isolated venv and install nnunetv2" \
      bash "$ROOT/scripts/nnunet_setup.sh"
stage nnunet_train "nnU-Net fold 0 reference arm" \
      bash "$ROOT/scripts/nnunet_train.sh"

log "###### overnight run complete ######"
log "GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
python3 -u -m ml.deploy --report 2>&1 | sed 's/^/    /'
log "ready for the test set: python3 -m ml.predict --input-dir <DIR> --output-dir outputs/testset"
