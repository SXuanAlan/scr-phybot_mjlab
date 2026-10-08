"""The squat clip (scripts/gen_squat_clip.py) and the joint groups the rewards track.

Every policy joint is tracked. The neck is not policy-controlled; it is
absent from the clip and its actuators hold it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_STANDING_JOINT_POS,
)

LEG_JOINT_NAMES = PHYBOT_C2_POLICY_JOINT_NAMES[:12]
WAIST_JOINT_NAMES = (PHYBOT_C2_POLICY_JOINT_NAMES[12],)
ARM_JOINT_NAMES = PHYBOT_C2_POLICY_JOINT_NAMES[13:21]
assert LEG_JOINT_NAMES + WAIST_JOINT_NAMES + ARM_JOINT_NAMES == tuple(PHYBOT_C2_POLICY_JOINT_NAMES)


def motion_duration_s(path: Path, control_dt: float = 0.02) -> float:
  """The clip's duration, requiring one frame per control step."""
  data = np.load(path, allow_pickle=True).item()
  frames = int(np.asarray(data["joint_positions"]).shape[0])
  fps = float(np.asarray(data["fps"]).reshape(-1)[0])
  duration_s = frames / fps
  control_steps = duration_s / control_dt
  if not np.isclose(control_steps, round(control_steps), atol=1e-9):
    raise ValueError(f"Squat motion duration {duration_s}s is not an integer number of {control_dt}s control steps.")
  if not np.isclose(fps, 1.0 / control_dt, atol=1e-6):
    raise ValueError(f"Squat motion fps {fps} must equal control fps {1.0 / control_dt}.")
  return duration_s


def motion_action_limits(
  path: Path,
  joint_names: tuple[str, ...],
  action_scale: float,
  margin: float = 0.75,
  minimum: float = 1.0,
) -> tuple[float, ...]:
  """Raw-action envelopes that contain the clip's complete trajectory of these joints."""
  data = np.load(path, allow_pickle=True).item()
  clip_joint_names = list(data["joints_list"])
  joint_ids = [clip_joint_names.index(name) for name in joint_names]
  joint_pos = np.asarray(data["joint_positions"], dtype=np.float32)[:, joint_ids]
  default = np.asarray(
    [PHYBOT_C2_STANDING_JOINT_POS[name] for name in joint_names],
    dtype=np.float32,
  )
  reference_action = (joint_pos - default) / action_scale
  limits = np.maximum(np.max(np.abs(reference_action), axis=0) + margin, minimum)
  return tuple(float(limit) for limit in limits)
