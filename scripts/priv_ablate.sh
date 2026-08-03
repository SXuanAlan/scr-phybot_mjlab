#!/usr/bin/env bash
# Route-B RMA step 0: privileged-channel ablation exam for one checkpoint.
#
#   scripts/priv_ablate.sh <ckpt> [angle] [steps] [gpu] [task]
#
# Zeroes each privileged actor channel in turn (engage_raster ablate= token)
# and exams at the target angle on the calibrated band; finishes with the
# all-channels-zeroed run (= the policy with no estimator at all, Route B's
# worst case). Channels that survive zeroing need no phi estimate; channels
# that crater define phi's regression targets. One line per cell goes to
# <ckpt_dir>/priv_ablate.log; raw dumps in logs/eval/ablate/.
set -u
CKPT="${1:?usage: priv_ablate.sh <ckpt> [angle] [steps] [gpu] [task]}"
ANGLE="${2:-90}"
STEPS="${3:-1500}"
GPU="${4:-2}"
TASK="${5:-Mjlab-Climb-Wall-Phybot-C2}"
PY=/home/spider/miniconda3/envs/env_mjlab15/bin/python
OUT="$(dirname "$CKPT")/priv_ablate.log"
mkdir -p logs/eval/ablate

run() { # label [ablate-list]
  local label="$1"
  local ablate="${2:-}"
  local raw="logs/eval/ablate/${label}.out"
  local tok=()
  [ -n "$ablate" ] && tok=("ablate=$ablate")
  CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl PYTHONPATH=src "$PY" \
    scripts/engage_raster.py "$CKPT" "$ANGLE" "$STEPS" \
    "logs/eval/ablate/${label}" "task=$TASK" "${tok[@]}" >"$raw" 2>&1
  local line
  line=$(grep -E "match\(|height gained|resets at" "$raw" | tr '\n' ' ')
  [ -z "$line" ] && line="EXAM FAILED :: $(tail -2 "$raw" | tr '\n' ' ')"
  echo "[$(date +%H:%M)] $label :: $line" >>"$OUT"
}

run baseline
run no_engaged microspine_engaged_state
run no_contact_fraction microspine_contact_fraction
run no_disengage_time microspine_disengage_time
run no_cooldown microspine_cooldown
run no_lin_vel base_lin_vel
run no_height height_above_spawn
run no_all_privileged microspine_engaged_state,microspine_contact_fraction,microspine_disengage_time,microspine_cooldown,base_lin_vel,height_above_spawn
echo "ABLATION DONE -> $OUT"
