"""Offline geometry and empirical-load audit. No environment steps or training.

Static loading is a necessary-condition relaxation: engaged passive gimbals
are balanced, free swinging gimbals/trajectory accelerations are not. Exported
loads are reference-derived optimization results, never policy or measured
loads. Failure here is useful evidence; it must not be relabelled a pass.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mjlab  # noqa: F401  (initialize mjlab before project imports)
import numpy as np

from scr_phybot_mjab.assets.motions import CAD_CLIMB_MOTION as MOTION
from scr_phybot_mjab.tools.climbing_reference.cad_cycle_audit import audit_clip
from scr_phybot_mjab.tools.paths import CLIMB_REVIEW_DIR as OUT


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion", type=Path, default=MOTION)
  parser.add_argument("--stride", type=int, default=20)
  parser.add_argument("--initial-only", action="store_true")
  parser.add_argument("--output-dir", type=Path, default=OUT)
  parser.add_argument(
    "--wall-angle", type=float, default=None, help="Wall angle from horizontal; overrides clip metadata"
  )
  parser.add_argument("--unit-layout", choices=("all", "corners"), default=None)
  parser.add_argument("--unit-quantile", type=float, default=0.5)
  parser.add_argument(
    "--preload-friction",
    type=float,
    default=0.2,
    help="Pre-latch friction coefficient; .2 matches the climbing contact pairs",
  )
  parser.add_argument(
    "--toe-friction", type=float, default=0.2, help="Toe friction coefficient; .2 matches the climbing contact pairs"
  )
  args = parser.parse_args(argv)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  clip = np.load(args.motion, allow_pickle=True).item()
  if args.wall_angle is not None:
    clip["wall_angle"] = args.wall_angle
  if args.unit_layout is not None:
    clip["unit_layout"] = args.unit_layout
  # Canonical clips without metadata belong to the twelve-unit main task.
  # Corner variants carry their layout, or select it with --unit-layout.
  clip.setdefault("unit_layout", "all")
  if not 0.0 < float(clip.get("wall_angle", 90.0)) <= 90.0:
    parser.error("--wall-angle must be in (0, 90]")
  if not 0.0 <= args.unit_quantile <= 1.0 or min(args.preload_friction, args.toe_friction) < 0.0 or args.stride < 1:
    parser.error("Invalid unit quantile, preload friction, or stride")
  report, loads = audit_clip(
    clip,
    args.motion,
    stride=args.stride,
    initial_only=args.initial_only,
    unit_quantile=args.unit_quantile,
    preload_friction=args.preload_friction,
    toe_friction=args.toe_friction,
  )
  (args.output_dir / "reference_audit.json").write_text(json.dumps(report, indent=2) + "\n")
  if loads:
    with (args.output_dir / "reference_static_loads.csv").open("w") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(loads[0]))
      writer.writeheader()
      writer.writerows(loads)
  raise SystemExit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
  main()
