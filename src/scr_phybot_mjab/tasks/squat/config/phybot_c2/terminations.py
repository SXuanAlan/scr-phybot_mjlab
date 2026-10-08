"""Squat terminations: the clip's end, a fall and a body on the floor."""

from __future__ import annotations

import math

from mjlab.envs import mdp as base_mdp
from mjlab.managers.termination_manager import TerminationTermCfg

from scr_phybot_mjab.tasks.common.phybot_c2 import ILLEGAL_GROUND_CONTACT_SENSOR
from scr_phybot_mjab.tasks.common.terminations import illegal_contact


def squat_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
    # The reference lean peaks near 20 degrees, so 70 leaves ample squat
    # margin and still ends genuinely unstable poses.
    "fell_over": TerminationTermCfg(func=base_mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}),
    "illegal_ground_contact": TerminationTermCfg(
      func=illegal_contact,
      params={"sensor_name": ILLEGAL_GROUND_CONTACT_SENSOR, "force_threshold": 1.0},
    ),
  }
