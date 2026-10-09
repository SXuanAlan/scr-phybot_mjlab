"""Generate the climbing stance and cycle on the CAD pads (tools/climbing_reference/cad_cycle.py).

Without --source-motion: solve the stance, write it to --initial-output,
build the cycle on it, project every frame onto the pad faces and write the
clip. With --source-motion: reuse that canonical clip and only record the
wall angle and active unit layout of a variant. This is offline inverse
kinematics, not policy optimization or a rollout.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_XML
from scr_phybot_mjab.tasks.climb_wall.motions import CAD_CLIMB_INITIAL_STATE, CAD_CLIMB_MOTION
from scr_phybot_mjab.tools.climbing_reference.cad_cycle import (
  project_to_pad_faces,
  solve_initial_stance,
  with_variant_metadata,
)
from scr_phybot_mjab.tools.climbing_reference.cycle import CAD_EMPIRICAL, generate_clip


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--initial-only", action="store_true")
  parser.add_argument("--step-height", type=float, default=0.02)
  parser.add_argument("--output", type=Path, default=CAD_CLIMB_MOTION)
  parser.add_argument("--initial-output", type=Path, default=CAD_CLIMB_INITIAL_STATE)
  parser.add_argument("--wall-angle", type=float, default=90.0)
  parser.add_argument("--unit-layout", choices=("all", "corners"), default="all")
  parser.add_argument(
    "--source-motion", type=Path, help="Reuse a current-CAD canonical trajectory for an angle/layout assessment"
  )
  args = parser.parse_args(argv)
  if not 0.0 < args.wall_angle <= 90.0:
    parser.error("--wall-angle must be in (0, 90]")
  if args.source_motion is not None:
    if args.source_motion.resolve() == args.output.resolve():
      parser.error("--source-motion requires a distinct --output path")
    clip = np.load(args.source_motion, allow_pickle=True).item()
    if clip.get("asset_xml_sha256") != hashlib.sha256(PHYBOT_C2_XML.read_bytes()).hexdigest():
      parser.error("Source motion CAD hash differs; regenerate its kinematics first")
    clip["source_motion_sha256"] = hashlib.sha256(args.source_motion.read_bytes()).hexdigest()
    _save_variant(with_variant_metadata(clip, args.wall_angle, args.unit_layout), args.output)
    return
  initial = solve_initial_stance()
  args.initial_output.parent.mkdir(parents=True, exist_ok=True)
  args.initial_output.write_text(json.dumps(initial, indent=2) + "\n")
  print(f"Wrote {args.initial_output}; max C-wall gap {initial['max_center_wall_error_m'] * 1000:.6f} mm", flush=True)
  if args.initial_only:
    return
  clip = generate_clip(dataclasses.replace(CAD_EMPIRICAL, step_height=args.step_height), initial_state=initial)
  print(clip.report[0])
  print("\n".join(clip.report[1:]))
  data = project_to_pad_faces(clip.data, initial)
  _save_variant(with_variant_metadata(data, args.wall_angle, args.unit_layout), args.output)


def _save_variant(clip: dict, output: Path) -> None:
  output.parent.mkdir(parents=True, exist_ok=True)
  np.save(output, clip)
  print(
    f"Wrote {output}: canonical poses, {clip['wall_angle']:g} degree wall, "
    f"{clip['active_units_per_pad']} active units per pad",
    flush=True,
  )


if __name__ == "__main__":
  main()
