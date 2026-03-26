"""Asset configurations."""

from spider_rl_mjlab.assets.phybot_big_v2 import (
  PHYBOT_BIG_V2_ACTION_SCALE,
  get_phybot_big_v2_robot_cfg,
)
from spider_rl_mjlab.assets.wall import (
  WALL_OFFSET,
  get_wall_cfg,
)

__all__ = [
  "get_phybot_big_v2_robot_cfg",
  "PHYBOT_BIG_V2_ACTION_SCALE",
  "get_wall_cfg",
  "WALL_OFFSET",
]
