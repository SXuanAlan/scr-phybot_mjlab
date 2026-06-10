"""Rizon4 microspine task registration.

Same A/B arms as the climb-wall task:

  * ``Mjlab-Microspine-Rizon4`` — constraint mode (virtual anchor +
    limit-surface release; the anisotropic microspine model).
  * ``Mjlab-Microspine-Rizon4-Legacy`` — legacy adhesion mode (600 N external
    force + mu=30 contacts).
"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import rizon4_microspine_env_cfg
from .rl_cfg import rizon4_microspine_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Microspine-Rizon4",
  env_cfg=rizon4_microspine_env_cfg(attachment_mode="constraint"),
  play_env_cfg=rizon4_microspine_env_cfg(play=True, attachment_mode="constraint"),
  rl_cfg=rizon4_microspine_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Microspine-Rizon4-Legacy",
  env_cfg=rizon4_microspine_env_cfg(attachment_mode="adhesion"),
  play_env_cfg=rizon4_microspine_env_cfg(play=True, attachment_mode="adhesion"),
  rl_cfg=rizon4_microspine_ppo_runner_cfg(),
)
