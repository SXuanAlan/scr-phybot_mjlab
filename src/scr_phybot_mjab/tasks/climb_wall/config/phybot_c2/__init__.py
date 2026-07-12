"""Constraint-based PHYBot C2 wall-climbing task registration."""

import dataclasses

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import phybot_c2_climb_wall_env_cfg
from .rl_cfg import phybot_c2_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Climb-Wall-Phybot-C2",
  env_cfg=phybot_c2_climb_wall_env_cfg(),
  play_env_cfg=phybot_c2_climb_wall_env_cfg(play=True),
  rl_cfg=dataclasses.replace(
    phybot_c2_ppo_runner_cfg(),
    wandb_tags=("phybot_c2", "microspine_constraint"),
  ),
)
