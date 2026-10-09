"""Generate the PHYBot C2 squat clip, tasks/squat/motions/squat.npy by default.

The choreography, the IK and the clip schema are in
scr_phybot_mjab.tools.squat_reference. Check the result with:
    python scripts/validate/check_amp_dataset.py --data-dir <output dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_POLICY_JOINT_NAMES
from scr_phybot_mjab.tasks.squat.motions import SQUAT_MOTION
from scr_phybot_mjab.tools.squat_reference import generate_clip


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output",
    type=Path,
    default=SQUAT_MOTION,
  )
  parser.add_argument("--depth", type=float, default=0.22, help="pelvis drop at squat bottom (m)")
  parser.add_argument(
    "--stand-s",
    type=float,
    default=3.0,
    help="standing hold padded before and after the squat (s)",
  )
  parser.add_argument("--descent-s", type=float, default=2.0, help="descend (and ascend) duration (s)")
  parser.add_argument("--hold-s", type=float, default=1.0, help="hold at squat bottom (s)")
  parser.add_argument("--reach-s", type=float, default=1.0, help="arm raise (and lower) duration (s)")
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument("--torso-lean", type=float, default=0.35, help="forward torso pitch at squat bottom (rad)")
  parser.add_argument(
    "--no-balance",
    dest="balance",
    action="store_false",
    help="disable CoM-over-feet pelvis balancing",
  )
  args = parser.parse_args(argv)

  data, max_residual = generate_clip(
    depth_m=args.depth,
    stand_s=args.stand_s,
    descent_s=args.descent_s,
    hold_s=args.hold_s,
    reach_s=args.reach_s,
    fps=args.fps,
    torso_lean=args.torso_lean,
    balance=args.balance,
  )
  output = args.output
  output.parent.mkdir(parents=True, exist_ok=True)
  np.save(output, data)

  depth_m, fps = args.depth, args.fps
  joint_positions = data["joint_positions"]
  frames = len(joint_positions)
  base_z = data["root_position"][:, 2]

  knee = PHYBOT_C2_POLICY_JOINT_NAMES.index("left_knee")
  shoulder = PHYBOT_C2_POLICY_JOINT_NAMES.index("left_shoulder_pitch")
  print(
    f"wrote {output} ({frames} frames, {frames / fps:.2f}s @ {fps:.1f} fps)\n"
    f"  depth={depth_m:.3f} m, base_z [{base_z.min():.3f}, {base_z.max():.3f}], "
    f"left_knee [{joint_positions[:, knee].min():.3f}, {joint_positions[:, knee].max():.3f}] rad\n"
    f"  left_shoulder_pitch [{joint_positions[:, shoulder].min():.3f}, "
    f"{joint_positions[:, shoulder].max():.3f}] rad (arm raise)\n"
    f"  max foot IK residual={max_residual:.2e} m (large -> depth beyond leg reach)"
  )
  if max_residual > 1e-2:
    print("  WARNING: foot IK did not converge; reduce --depth or check foot targets.")


if __name__ == "__main__":
  main()
