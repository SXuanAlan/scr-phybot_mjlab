"""Constraint-based PHYBot C2 wall-climbing task registration."""

import dataclasses

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  phybot_c2_climb_wall_corners60_env_cfg,
  phybot_c2_climb_wall_corners80_env_cfg,
  phybot_c2_climb_wall_env_cfg,
)
from .rl_cfg import (
  phybot_c2_ppo_runner_cfg,
)

# Main climbing task: all twelve CAD units per pad and the legacy ladder
# ending at 90 degrees. Hardware variants share its rewards, RSI and PD.
# The optional unscheduled-release drill is disabled for fresh training.
register_mjlab_task(
  task_id="Mjlab-Climb-Wall-Phybot-C2",
  env_cfg=phybot_c2_climb_wall_env_cfg(),
  play_env_cfg=phybot_c2_climb_wall_env_cfg(play=True),
  rl_cfg=dataclasses.replace(
    phybot_c2_ppo_runner_cfg(),
    wandb_tags=("phybot_c2", "microspine_constraint", "microspine_empirical", "units12", "curriculum90"),
  ),
)

register_mjlab_task(
  task_id="Mjlab-Climb-Wall-Phybot-C2-Corners-A60",
  env_cfg=phybot_c2_climb_wall_corners60_env_cfg(),
  play_env_cfg=phybot_c2_climb_wall_corners60_env_cfg(play=True),
  rl_cfg=dataclasses.replace(
    phybot_c2_ppo_runner_cfg(),
    experiment_name="phybot_c2_climb_wall_corners60",
    wandb_tags=("phybot_c2", "microspine_empirical", "units4", "curriculum60"),
  ),
)

register_mjlab_task(
  task_id="Mjlab-Climb-Wall-Phybot-C2-Corners-A80",
  env_cfg=phybot_c2_climb_wall_corners80_env_cfg(),
  play_env_cfg=phybot_c2_climb_wall_corners80_env_cfg(play=True),
  rl_cfg=dataclasses.replace(
    phybot_c2_ppo_runner_cfg(),
    wandb_tags=("phybot_c2", "microspine_corners", "wall80"),
  ),
)
