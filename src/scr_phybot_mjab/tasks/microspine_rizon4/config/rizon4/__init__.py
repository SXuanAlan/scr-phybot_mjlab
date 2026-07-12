"""Constraint-based Rizon4 microspine task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import rizon4_microspine_env_cfg
from .rl_cfg import rizon4_microspine_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Microspine-Rizon4",
  env_cfg=rizon4_microspine_env_cfg(),
  play_env_cfg=rizon4_microspine_env_cfg(play=True),
  rl_cfg=rizon4_microspine_ppo_runner_cfg(),
)
