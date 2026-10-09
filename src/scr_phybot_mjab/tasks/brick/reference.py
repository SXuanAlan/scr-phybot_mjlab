"""The brick task's reference clip: where it lives and how the task loads it.

``tools/brick_reference.py`` generates and audits the clip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_POLICY_JOINT_NAMES as JOINTS,
)
from scr_phybot_mjab.tasks.brick.motions import BRICK_REFERENCE_MOTION

__all__ = ("PHASE_NAMES", "REFERENCE_PATH", "load_reference")

REFERENCE_PATH = BRICK_REFERENCE_MOTION
PHASE_NAMES = ("reach", "engage", "lift", "carry", "place", "release", "stand")


def load_reference(path=REFERENCE_PATH, *, require_audit=True):
  if not Path(path).exists():
    raise FileNotFoundError(
      f"Generate the brick reference first: python scripts/reference/gen_brick_reference.py --output {path}"
    )
  with np.load(path, allow_pickle=False) as archive:
    result = {key: archive[key].copy() for key in archive.files}
  if int(result["schema_version"]) != 1 or tuple(result["joint_names"]) != tuple(JOINTS):
    raise ValueError("Brick reference schema or policy joint order does not match this robot")
  audit = json.loads(str(result["audit_json"]))
  if require_audit and not audit["passed"]:
    raise ValueError("Brick reference failed its geometric audit; inspect the .audit.json before training")
  return result
