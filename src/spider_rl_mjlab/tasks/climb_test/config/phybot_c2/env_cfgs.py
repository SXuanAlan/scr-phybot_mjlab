"""PHYBot C2 climb-test environment.

This task intentionally clones ``Mjlab-Climb-Wall-Phybot-C2`` and only changes
the gravity curriculum: gravity angle is pinned at 0 degrees for the whole run.
All robot setup, observations, rewards, microspine adhesion, terminations, and
play-time debug behavior stay inherited from the active C2 climb-wall config.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from spider_rl_mjlab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)


_FIXED_GRAVITY_ANGLE_DEG = 0.0


def phybot_c2_climb_test_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the C2 climb-test config as climb-wall C2 with fixed 0° gravity."""
  cfg = phybot_c2_climb_wall_env_cfg(play=play)
  cfg.curriculum["gravity_angle"].params["angle_stages"] = [
    {"iter": 0, "angle_deg": _FIXED_GRAVITY_ANGLE_DEG},
  ]
  return cfg
