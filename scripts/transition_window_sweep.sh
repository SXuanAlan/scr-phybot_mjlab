#!/usr/bin/env bash
# Transition window sweep (user call, 08-19 evening): one round renders the
# FULLRUN raster (engage-schedule png + mp4) AND four per-window probes
# (video + window-level schedule png each) on the newest checkpoint, then
# prints a grep-able summary. Exams run at the checkpoint's CURRICULUM
# gravity (the probes resolve the ladder themselves; gravity=1.0 token for
# graduation exams).
#   scripts/transition_window_sweep.sh <run_dir> <tag> [gpu]
set -u
RUN_DIR="${1:?usage: transition_window_sweep.sh <run_dir> <tag> [gpu]}"
TAG="${2:?tag}"
GPU="${3:-2}"
PY=/home/spider/miniconda3/envs/env_mjlab15/bin/python
TASK=Mjlab-Climb-Transition-Phybot-C2
CK=$(ls -v "$RUN_DIR"/model_*.pt | tail -1)
IT=$(basename "$CK" .pt); IT=${IT#model_}
P="$RUN_DIR/${TAG}_${IT}"
run() { CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl PYTHONPATH=src "$PY" "$@"; }

run scripts/engage_raster.py "$CK" 90 1100 "${P}_fullrun" task=$TASK \
  > "${P}_fullrun.out" 2>&1
run scripts/mount_window_probe.py "$CK" 600 "${P}_nursery" task=$TASK \
  segments=hand_place > "${P}_nursery.out" 2>&1
run scripts/mount_window_probe.py "$CK" 600 "${P}_setleft" task=$TASK \
  segments=hand_set_left > "${P}_setleft.out" 2>&1
run scripts/mount_window_probe.py "$CK" 600 "${P}_rightwin" task=$TASK \
  segments=hand_place_right > "${P}_rightwin.out" 2>&1
run scripts/mount_window_probe.py "$CK" 500 "${P}_footwin" task=$TASK \
  segments=left_foot_mount,foot_switch_hold,right_foot_mount \
  > "${P}_footwin.out" 2>&1

echo "=== SWEEP SUMMARY tag=$TAG iter=$IT ckpt=$CK ==="
for w in fullrun nursery setleft rightwin footwin; do
  echo "--- $w"
  grep -aE "gravity scale|match\(|height gained|resets at|latch lifetimes|resets,|press at latch|release reasons" \
    "${P}_${w}.out" | tail -6
done
ls -la "${P}"_*.mp4 "${P}"_*schedule.png "${P}_fullrun.png" 2>/dev/null
