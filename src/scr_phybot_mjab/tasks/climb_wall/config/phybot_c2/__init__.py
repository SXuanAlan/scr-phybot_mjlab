"""Constraint-based PHYBot C2 wall-climbing task registration."""

import dataclasses

from mjlab.tasks.registry import register_mjlab_task

from scr_phybot_mjab.rl import MjlabDistillationRunner

from .env_cfgs import (
  phybot_c2_climb_wall_distill_env_cfg,
  phybot_c2_climb_wall_env_cfg,
)
from .rl_cfg import phybot_c2_distill_runner_cfg, phybot_c2_ppo_runner_cfg

# The climb task IS the deployable recipe: hardware-exact PD and the s2r
# domain-randomization package are on by default (2026-08-01). The
# unscheduled-release drill is deliberately excluded from that package —
# enable robustness_release on a checkpoint that already masters exact-90.
register_mjlab_task(
  task_id="Mjlab-Climb-Wall-Phybot-C2",
  env_cfg=phybot_c2_climb_wall_env_cfg(),
  play_env_cfg=phybot_c2_climb_wall_env_cfg(play=True),
  rl_cfg=dataclasses.replace(
    phybot_c2_ppo_runner_cfg(),
    wandb_tags=("phybot_c2", "microspine_constraint"),
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
