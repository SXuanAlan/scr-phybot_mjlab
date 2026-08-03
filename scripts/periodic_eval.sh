#!/usr/bin/env bash
# Periodic exam sweep for a running climb training run.
#
# Every ~250 new iterations, examine the newest checkpoint at the angles that
# matter for the rung it is on. The probe angle is what decides rung mastery:
# the pool average hides the steep tail (a policy scored 0.971 at exact-75
# while failing 85 with 3 falls/30 s). Exams always use the calibrated
# tilt-peel band (play cfg), never the training band.
#
#   scripts/periodic_eval.sh <run_dir> [gpu] [iter_step]
set -u
RUN_DIR="${1:?usage: periodic_eval.sh <run_dir> [gpu] [iter_step]}"
GPU="${2:-2}"
STEP="${3:-250}"
TASK="Mjlab-Climb-Wall-Phybot-C2"
PY=/home/spider/miniconda3/envs/env_mjlab15/bin/python
OUT="$RUN_DIR/periodic_eval.log"
last=0

# v4 revival (user call 2026-08-02 afternoon): every cycle examines the
# WHOLE ladder, mastered angles included — v3's 75/85 regression ran 1500
# iters undetected because the probe angle followed the curriculum. 88/90
# rows before those rungs open just track the frontier. Per-EE occupancy
# lines are captured because a single limb overholding (occ_pol - occ_ref
# > +0.04) is the weld early warning, visible well before match drops.
probe_angles() {
  echo "75 85 88 90"
}

training_band() {
  echo ""  # calibrated band from iter 0
}

while true; do
  CKPT=$(ls -v "$RUN_DIR"/model_*.pt 2>/dev/null | tail -1)
  [ -z "$CKPT" ] && { sleep 120; continue; }
  it=$(basename "$CKPT" .pt); it=${it#model_}
  if [ "$it" -ge $((last + STEP)) ]; then
    band=$(training_band "$it")
    for ang in $(probe_angles "$it"); do
      # Two exams per angle. The calibrated band is the ship criterion; the
      # training band separates "can it climb" from "is its posture tight
      # enough yet" — until the anneal lands, a calibrated-band exam peels
      # the policy for a tolerance it has not been taught.
      for variant in "cal" "train"; do
        if [ "$variant" = "train" ]; then
          [ -z "$band" ] && continue   # anneal done: same as the cal exam
          tok="tiltband=$band"
          label="train($band)"
        else
          tok=""
          label="cal(3.5,4.5)"
        fi
        raw="$RUN_DIR/.periodic_eval_last.out"
        CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl PYTHONPATH=src "$PY" \
          scripts/engage_raster.py "$CKPT" "$ang" 1500 "task=$TASK" $tok \
          > "$raw" 2>&1
        line=$(grep -E "match\(|occ pol|height gained|forced-release reasons" "$raw" |
          tr '\n' ' ')
        # An exam that dies must say so: a bare label reads like a pass.
        [ -z "$line" ] && line="EXAM FAILED :: $(tail -2 "$raw" | tr '\n' ' ')"
        echo "[$(date +%H:%M)] iter=$it angle=$ang band=$label $line" >> "$OUT"
      done
    done
    last=$it
  fi
  sleep 120
done
