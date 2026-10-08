"""Generate and audit the planted-foot Phybot brick reference."""

import argparse
import json
from pathlib import Path

from scr_phybot_mjab.assets.brick import BrickCfg
from scr_phybot_mjab.tasks.brick.geometry import DEFAULT_UNIT_LAYOUT
from scr_phybot_mjab.tasks.brick.reference import REFERENCE_PATH
from scr_phybot_mjab.tools.brick_reference import audit_reference, generate_reference


def main(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=REFERENCE_PATH)
  parser.add_argument("--knots", type=int, default=21)
  parser.add_argument("--start-x", type=float, default=0.32)
  parser.add_argument("--start-y", type=float, default=0.0)
  parser.add_argument("--goal-x", type=float, default=0.0)
  parser.add_argument("--goal-y", type=float, default=0.15)
  parser.add_argument("--size", type=float, nargs=3, default=(0.18, 0.32, 0.16), metavar=("X", "Y", "Z"))
  parser.add_argument("--mass", type=float, default=1.0)
  parser.add_argument("--lift-height", type=float, default=0.12)
  parser.add_argument("--unit-layout", choices=("all", "corners"), default=DEFAULT_UNIT_LAYOUT)
  parser.add_argument("--audit-only", action="store_true", help="Re-audit the existing --output file")
  args = parser.parse_args(argv)
  brick = BrickCfg(
    size=tuple(args.size),
    mass=args.mass,
    start_xy=(args.start_x, args.start_y),
    goal_offset_xy=(args.goal_x, args.goal_y),
    lift_height=args.lift_height,
  )
  report = (
    audit_reference(args.output)
    if args.audit_only
    else generate_reference(args.output, brick, args.knots, args.unit_layout)
  )
  print(json.dumps(report, indent=2))
  raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
  main()
