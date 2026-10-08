"""Evaluate the four-corner climbing actor without hiding resets or seed grips.

Example (CUDA_VISIBLE_DEVICES selects the physical evaluation GPU)::

  python scripts/eval/evaluate_climb_checkpoint.py --checkpoint logs/.../model_250.pt \
    --angles 45 50 --num-envs 32 --duration-s 30 --output-dir logs/eval/model_250

Every angle produces metrics.json, trajectory.npz, engage_schedule.png and a
video of environment 0 from that same physical rollout. Terminations stay
enabled. We disable automatic reset only to record the terminal state, then
explicitly reset done worlds before the next physics step. No capacity,
reward, controller or failure threshold is changed.

The default canonical profile starts at reference frame 0 with ordinary DR
off; empirical per-unit strength/early-loading draws still run. The optional
training-rsi profile retains training DR and height offsets, with uniform
full-cycle phase coverage instead of the training failure-adaptive sampler.
Both profiles use strict actor loading, including observation normalization.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from scr_phybot_mjab.tools.climb_evaluation import TASK, evaluate_checkpoint


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument(
    "--angles",
    "--angle-deg",
    required=True,
    nargs="+",
    type=float,
    help="Explicit current mean and band upper edge (e.g. 45 50); no default 60-degree exam",
  )
  parser.add_argument("--task", default=TASK)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--duration-s", type=float, default=30.0)
  parser.add_argument("--settle-s", type=float, default=0.5)
  parser.add_argument("--profile", choices=("canonical", "training-rsi"), default="canonical")
  parser.add_argument("--seed", type=int, default=20260908)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--video-fps", type=float, default=25.0)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  args = parser.parse_args(argv)
  if args.num_envs < 1 or args.duration_s <= 0 or args.settle_s < 0 or args.video_fps <= 0:
    parser.error("num-envs, duration-s and video-fps must be positive; settle-s must be nonnegative")
  if any(not math.isfinite(angle) or not 0 < angle <= 60 for angle in args.angles):
    parser.error("The reduced-hardware curriculum evaluation supports explicit angles in (0, 60]")
  args.checkpoint = args.checkpoint.resolve(strict=True)
  args.output_dir = args.output_dir.resolve()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("MUJOCO_GL", "egl")
  evaluate_checkpoint(args)


if __name__ == "__main__":
  main()
