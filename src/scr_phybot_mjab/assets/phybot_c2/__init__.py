"""PHYBot C2 robot configuration."""

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_ACTION_SCALE,
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_STANDING_JOINT_POS,
  configure_phybot_c2_floor_geom,
  configure_phybot_c2_terrain_cfg,
  get_phybot_c2_robot_cfg,
)

__all__ = [
  "get_phybot_c2_robot_cfg",
  "PHYBOT_C2_ACTION_SCALE",
  "PHYBOT_C2_POLICY_JOINT_NAMES",
  "PHYBOT_C2_STANDING_JOINT_POS",
  "configure_phybot_c2_floor_geom",
  "configure_phybot_c2_terrain_cfg",
]
