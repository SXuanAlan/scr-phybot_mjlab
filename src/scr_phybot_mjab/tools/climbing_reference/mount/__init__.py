"""The walk-to-climb mount reference on a tilted wall.

``generate_mount_clip(MountRecipe())`` reproduces the committed a80 clip:
the builder sets up the IK context and the standing and handoff stances,
the choreography solves every segment in clip order, and the audit prints
what the solve achieved.
"""

from __future__ import annotations

from pathlib import Path

from scr_phybot_mjab.tasks.climb_wall.motions import CAD_CLIMB_MOTION

from . import audit, choreography, outputs
from .builder import SEGMENT_NAMES, MountBuilder
from .recipe import MountRecipe

__all__ = ("SEGMENT_NAMES", "MountBuilder", "MountRecipe", "generate_mount_clip")


def generate_mount_clip(recipe: MountRecipe | None = None, handoff_motion: Path = CAD_CLIMB_MOTION) -> dict:
  """Solve the mount and return the clip dictionary (np.save it to write the clip).

  ``handoff_motion`` is the climb cycle whose first complete pose, passive
  joints included, is the mount's final stance.
  """
  b = MountBuilder(recipe or MountRecipe(), handoff_motion)
  choreography.perform(b)
  outputs.smooth(b)
  clip = outputs.clip_data(b, handoff_motion)
  audit.report(b, clip)
  return clip
