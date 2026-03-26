"""Asset configurations."""

from spider_rl_mjlab.assets.phybot_c2 import (
  PHYBOT_C2_ACTION_SCALE,
  get_phybot_c2_robot_cfg,
)
from spider_rl_mjlab.assets.phybot_v8 import (
  PHYBOT_V8_ACTION_SCALE,
  get_phybot_v8_robot_cfg,
)
from spider_rl_mjlab.assets.wall import (
  WALL_OFFSET,
  get_wall_cfg,
)

__all__ = [
  "get_phybot_c2_robot_cfg",
  "PHYBOT_C2_ACTION_SCALE",
  "get_phybot_v8_robot_cfg",
  "PHYBOT_V8_ACTION_SCALE",
  "get_wall_cfg",
  "WALL_OFFSET",
]
