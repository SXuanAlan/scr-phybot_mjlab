"""PHYBot C2 robot configuration."""

from spider_rl_mjlab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_ACTION_SCALE,
  configure_phybot_c2_floor_geom,
  configure_phybot_c2_terrain_cfg,
  get_phybot_c2_robot_cfg,
)

__all__ = [
  "get_phybot_c2_robot_cfg",
  "PHYBOT_C2_ACTION_SCALE",
  "configure_phybot_c2_floor_geom",
  "configure_phybot_c2_terrain_cfg",
]
