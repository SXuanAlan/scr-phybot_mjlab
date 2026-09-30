"""Interactively view a motion .npy clip on the PHYBot C2 model in the MuJoCo viewer.

Kinematically replays any clip in the AMP/motion schema (``joints_list``,
``joint_positions``, ``root_position``, ``root_quaternion`` xyzw, ``fps``) in an
interactive window so you can orbit/zoom while it loops. No physics is stepped.
Needs a display; for headless MP4 rendering use
``scripts/validation/check_amp_dataset.py`` instead.

Usage:
    python scripts/view_motion.py                                  # default squat clip
    python scripts/view_motion.py src/scr_phybot_mjab/assets/motions/mini_walk_slow.npy
    python scripts/view_motion.py <clip.npy> --speed 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (  # noqa: E402
  PHYBOT_C2_STANDING_JOINT_POS,
  get_spec,
)

_DEFAULT_CLIP = _REPO / "src/scr_phybot_mjab/assets/motions/squat.npy"


def _load(motion_file: Path):
  data = np.load(motion_file, allow_pickle=True).item()
  return (
    list(data["joints_list"]),
    np.asarray(data["joint_positions"], dtype=np.float64),
    np.asarray(data["root_position"], dtype=np.float64),
    np.asarray(data["root_quaternion"], dtype=np.float64),  # xyzw
    float(np.asarray(data["fps"]).reshape(-1)[0]),
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "motion_file",
    type=Path,
    nargs="?",
    default=_DEFAULT_CLIP,
    help=f"clip .npy to view (default: {_DEFAULT_CLIP.name})",
  )
  parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
  args = parser.parse_args()

  joints, joint_pos, root_pos, root_quat_xyzw, fps = _load(args.motion_file)

  model = get_spec().compile()
  data = mujoco.MjData(model)
  qadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joints}

  def set_frame(t: int) -> None:
    data.qpos[:] = 0.0
    data.qpos[0:3] = root_pos[t]
    # xyzw (clip) -> wxyz (MuJoCo).
    data.qpos[3:7] = root_quat_xyzw[t, [3, 0, 1, 2]]
    for name, angle in PHYBOT_C2_STANDING_JOINT_POS.items():
      data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = angle
    for i, name in enumerate(joints):
      data.qpos[qadr[name]] = joint_pos[t, i]
    mujoco.mj_forward(model, data)

  print(f"viewing {args.motion_file} ({len(joint_pos)} frames @ {fps:.1f} fps). Close window to quit.")
  dt = 1.0 / (fps * max(args.speed, 1e-3))
  with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0
    while viewer.is_running():
      set_frame(t)
      viewer.sync()
      t = (t + 1) % len(joint_pos)
      time.sleep(dt)


if __name__ == "__main__":
  main()
