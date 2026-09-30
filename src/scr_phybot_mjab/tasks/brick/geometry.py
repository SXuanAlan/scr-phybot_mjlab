"""Collision geometry shared by reference auditing and live simulation."""

import copy

from scr_phybot_mjab.tasks.walk.config.phybot_c2.env_cfgs import PHYBOT_C2_WALK_COLLISION

DEFAULT_UNIT_LAYOUT = "corners"


def active_unit_count(layout):
  if layout not in ("all", "corners"):
    raise ValueError("Brick grasp unit layout must be 'all' or 'corners'")
  return 12 if layout == "all" else 4


def brick_robot_collision():
  collision = copy.deepcopy(PHYBOT_C2_WALK_COLLISION)
  collision.geom_names_expr = (".*_collision.*", r"^(left|right)_hand_spine_c_[0-9]+_[0-9]+$")
  collision.contype = {r"^(left|right)_foot_spine(?:_.*)?_collision$": 0, ".*": 1}
  collision.conaffinity = dict(collision.contype)
  return collision
