"""Validation suite for the microspine connect-equality attachment.

Subcommands:
  hang     16-env hang test on a vertical (90°) wall: all 4 EEs anchored,
           zero actions. Asserts the anchors hold the robot weight without
           sag, jitter, NaNs, or efc overflow.
  reference-hold
           GPU controller-settling stress test from sampled reference
           snapshots. This is intentionally not a static feasibility proof.
           Releases during the artificial gravity ramp are settle
           transients: tolerated on <= 5% of frames (one anchor at most,
           stance stable on the rest). A release at held full gravity, any
           cascade, or a frame ending two anchors short always fails.
  ablation Disable physical grip at the same low ordinary wall friction and
           assert that the robot falls. This proves the equality anchors, not
           incidental contact friction, provide vertical support.
  readout  Cross-checks the efc-row force gather against the kinematic
           anchor error (constraint force ∝ pose error) under a time-varying
           torso push.

Run from the repo root inside the env_mjlab environment, e.g.:
  python scripts/validate_microspine_constraint.py hang --num-envs 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scr_phybot_mjab.tools.microspine.bench import cmd_ablation, cmd_hang, cmd_readout, cmd_reference_hold


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="cmd", required=True)

  p_hang = sub.add_parser("hang")
  p_hang.add_argument("--num-envs", type=int, default=16)
  p_hang.add_argument("--seconds", type=float, default=10.0)
  p_hang.add_argument(
    "--no-limit-surface",
    action="store_true",
    help="Validate the raw anchor mechanics with hold checks disabled "
    "(pre-re-grip behavior; engaged spines stay latched forever).",
  )

  p_reference = sub.add_parser("reference-hold")
  p_reference.add_argument("motion_file", type=Path)
  p_reference.add_argument(
    "--ramp-seconds",
    type=float,
    default=0.8,
    help="gravity-ramp duration for every sampled controller snapshot",
  )
  p_reference.add_argument(
    "--hold-seconds",
    type=float,
    default=0.4,
    help="full-gravity overdamped hold after the ramp",
  )
  p_reference.add_argument(
    "--frame-stride",
    type=int,
    default=1,
    help="sample every Nth cycle frame; segment boundaries are always included",
  )

  p_ablation = sub.add_parser("ablation")
  p_ablation.add_argument("--num-envs", type=int, default=16)
  p_ablation.add_argument("--seconds", type=float, default=1.0)
  p_ablation.add_argument("--minimum-drop", type=float, default=0.25)

  p_read = sub.add_parser("readout")
  p_read.add_argument("--num-envs", type=int, default=8)
  p_read.add_argument("--seconds", type=float, default=10.0)

  args = parser.parse_args(argv)
  ok = {
    "hang": cmd_hang,
    "reference-hold": cmd_reference_hold,
    "ablation": cmd_ablation,
    "readout": cmd_readout,
  }[args.cmd](args)
  print("OVERALL:", "PASS" if ok else "FAIL")
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
