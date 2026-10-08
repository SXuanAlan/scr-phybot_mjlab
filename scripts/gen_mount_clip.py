"""Generate the PHYBot C2 walk-to-climb mount clip on a tilted wall via whole-body IK.

The mount bridges the walk task's standing keyframe (feet flat on the floor,
arms down) to the climb cycle's first frame (all four microspine pads
engaged on the wall), so the transition policy hands off to the climb
policy at an exact reference frame. The clip is acyclic; its final segment
is named ``end_hold`` so ``ClimbMotionClip.from_dict`` accepts it unchanged
(everything before ``end_hold`` reads as one cycle, and the tail supplies
the finite-difference velocity frame). ``step_height`` is 0.

The committed clip, assets/motions/climb_mount_prologue_a80.npy, comes from
the command in its recipe file. In it the left hand takes a low rung and
the right hand a middle rung while both feet stand on the floor; the right
foot and then the left foot (a toe pivot) mount the wall on the hand welds;
the right hand regrips once and the left hand twice up to the final rungs;
an all-four dwell at the climb cycle's first frame ends the clip. The
generator lives in tools/climbing_reference/mount, one function per segment
in choreography.py; every number is a MountRecipe field and a command-line
option.

Output schema = the cycle clip (tools.climbing_reference.cycle) plus
``foot_on_ground`` (T, 2) uint8: 1 while that foot (left, right) is a flat
floor support (feet in ``toe_list``/``foot_sites_list`` order).

Validate:
    python scripts/audit_mount_reference.py <clip.npy>
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import mjlab  # noqa: F401  (registers mjlab entry points)
import numpy as np

from scr_phybot_mjab.assets.motions import CAD_CLIMB_MOTION
from scr_phybot_mjab.tools.climbing_reference.mount import MountRecipe, generate_mount_clip


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--output", type=Path, required=True, help="clip to write")
  parser.add_argument(
    "--handoff-motion",
    type=Path,
    default=CAD_CLIMB_MOTION,
    help="cycle whose first complete pose (including passive joints) defines the handoff",
  )
  # One option per recipe field, defaulting to the a80 mount; see MountRecipe for their meaning.
  a80 = MountRecipe()
  for field in dataclasses.fields(MountRecipe):
    default = getattr(a80, field.name)
    parser.add_argument(
      f"--{field.name.replace('_', '-')}",
      type=float,
      nargs=len(default) if isinstance(default, tuple) else None,
      default=default,
      help=f"default {default}",
    )
  args = parser.parse_args(argv)
  values = {field.name: getattr(args, field.name) for field in dataclasses.fields(MountRecipe)}
  recipe = MountRecipe(**{name: tuple(v) if isinstance(v, list) else v for name, v in values.items()})
  clip = generate_mount_clip(recipe, handoff_motion=args.handoff_motion)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.save(args.output, clip)
  print(f"wrote {args.output}")


if __name__ == "__main__":
  main()
