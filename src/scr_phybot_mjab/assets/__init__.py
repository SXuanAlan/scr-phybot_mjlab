"""Asset configurations."""

from scr_phybot_mjab.assets.forearm_hand import (
  FOREARM_HAND_ACTION_SCALE,
  get_forearm_hand_robot_cfg,
)
from scr_phybot_mjab.assets.phybot_big_v2 import (
  PHYBOT_BIG_V2_ACTION_SCALE,
  get_phybot_big_v2_robot_cfg,
)
from scr_phybot_mjab.assets.phybot_c2 import (
  PHYBOT_C2_ACTION_SCALE,
  configure_phybot_c2_floor_geom,
  configure_phybot_c2_terrain_cfg,
  get_phybot_c2_robot_cfg,
)
from scr_phybot_mjab.assets.wall import (
  WALL_OFFSET,
  get_wall_cfg,
)

__all__ = [
  "get_forearm_hand_robot_cfg",
  "FOREARM_HAND_ACTION_SCALE",
  "get_phybot_big_v2_robot_cfg",
  "PHYBOT_BIG_V2_ACTION_SCALE",
  "get_phybot_c2_robot_cfg",
  "PHYBOT_C2_ACTION_SCALE",
  "configure_phybot_c2_floor_geom",
  "configure_phybot_c2_terrain_cfg",
  "get_wall_cfg",
  "WALL_OFFSET",
]
