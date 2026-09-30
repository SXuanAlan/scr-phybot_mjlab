"""Constraint-based PHYBot C2 wall-climbing task registration."""

import dataclasses

from mjlab.tasks.registry import register_mjlab_task

from scr_phybot_mjab.rl import MjlabDistillationRunner

from .env_cfgs import (
  phybot_c2_climb_wall_corners60_env_cfg,
  phybot_c2_climb_wall_corners80_env_cfg,
  phybot_c2_climb_wall_distill_env_cfg,
  phybot_c2_climb_wall_env_cfg,
  phybot_c2_climb_wall_polish_env_cfg,
)
from .rl_cfg import (
  phybot_c2_distill_runner_cfg,
  phybot_c2_ppo_polish_runner_cfg,
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

# Student distillation off a climb checkpoint (pass
# --agent.teacher-checkpoint <model.pt>). Separate id because it swaps the
# algorithm and runner class, not because it is a different task.
register_mjlab_task(
  task_id="Mjlab-Climb-Wall-Phybot-C2-Distill",
  env_cfg=phybot_c2_climb_wall_distill_env_cfg(),
  play_env_cfg=phybot_c2_climb_wall_distill_env_cfg(play=True),
  rl_cfg=phybot_c2_distill_runner_cfg(),
  runner_cls=MjlabDistillationRunner,
)

# Phase-3 polish: ordinary PPO, but the actor consumes the student group
# (BC-init via scripts/train_polish.py, privileged critic seeded from the
# teacher checkpoint). Separate id because it needs the student group in the
# env AND a different obs_groups mapping — same exception class as -Distill.
register_mjlab_task(
  task_id="Mjlab-Climb-Wall-Phybot-C2-Polish",
  env_cfg=phybot_c2_climb_wall_polish_env_cfg(),
  play_env_cfg=phybot_c2_climb_wall_polish_env_cfg(play=True),
  rl_cfg=phybot_c2_ppo_polish_runner_cfg(),
)
