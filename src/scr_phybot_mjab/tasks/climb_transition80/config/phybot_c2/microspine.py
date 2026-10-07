"""Microspine pad physics of the transition task: climb's twelve-unit plant, with arm wall contact."""

from __future__ import annotations

import dataclasses

from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.microspine import climb_microspine_cfg

ARM_WALL_GEOMS = (
  "left_shoulder_roll_collision",
  "right_shoulder_roll_collision",
  "left_shoulder_yaw_collision",
  "right_shoulder_yaw_collision",
  "left_elbow_pitch_collision",
  "right_elbow_pitch_collision",
)
"""Arm collision geoms with ordinary wall contact: the arms may lean on the wall."""


def transition80_microspine_cfg() -> MicrospineCfg:
  """The grip event takes a shallow copy of this config, so it must be complete before that event is built."""
  climb = climb_microspine_cfg("all")
  return dataclasses.replace(
    climb,
    arm_wall_contact_geoms=ARM_WALL_GEOMS,
    # Seeded welds are judged from step 7, inside the 25-step birth freeze:
    # an unjudged seed cannot break, and the policy would learn to lean on it.
    limit_surface=dataclasses.replace(climb.limit_surface, seed_hold_judge_guard_steps=6),
  )
