#!/usr/bin/env bash
# DR-pinned exam grid over the lottery corners for one checkpoint.
#
#   scripts/dr_bench.sh <ckpt> [angle] [steps] [gpu]
#
# The clean exam (play cfg) strips all DR, so it says nothing about the
# worst draws the training pool samples. This bench re-arms the DR pieces
# one pinned cell at a time (engage_raster drfriction=/drmass=/drbias=/
# drnoise tokens): a 3x3 friction x mass grid on clean obs, then noise-only,
# bias-only, and the full worst-corner stack. One summary line per cell goes
# to <ckpt_dir>/dr_bench.log; raw dumps land in logs/eval/drbench/.
# Bias/noise cells are stochastic (per-joint draw / per-step noise) — rerun
# interesting cells before concluding.
set -u
CKPT="${1:?usage: dr_bench.sh <ckpt> [angle] [steps] [gpu]}"
ANGLE="${2:-90}"
STEPS="${3:-1500}"
GPU="${4:-2}"
TASK=Mjlab-Climb-Wall-Phybot-C2
PY=/home/spider/miniconda3/envs/env_mjlab15/bin/python
OUT="$(dirname "$CKPT")/dr_bench.log"
mkdir -p logs/eval/drbench

run() { # label token...
  local label="$1"
  shift
  local raw="logs/eval/drbench/${label}.out"
  CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl PYTHONPATH=src "$PY" \
    scripts/engage_raster.py "$CKPT" "$ANGLE" "$STEPS" \
    "logs/eval/drbench/${label}" "task=$TASK" "$@" >"$raw" 2>&1
  local line
  line=$(grep -E "match\(|occ pol|height gained|forced-release reasons" "$raw" |
    tr '\n' ' ')
  [ -z "$line" ] && line="EXAM FAILED :: $(tail -2 "$raw" | tr '\n' ' ')"
  echo "[$(date +%H:%M)] $label :: $line" >>"$OUT"
}

for f in 0.6 0.8 1.0; do
  for m in -2 0 2; do
    run "f${f}_m${m}" "drfriction=$f" "drmass=$m"
  done
done
run "noise_only" drnoise
run "bias_only" "drbias=0.01"
run "worst_f0.6_m2_noise_bias" "drfriction=0.6" "drmass=2" drnoise "drbias=0.01"
echo "ALL DONE -> $OUT"
