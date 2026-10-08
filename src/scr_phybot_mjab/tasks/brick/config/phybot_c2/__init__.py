"""PHYBot C2 brick task registration: six variants of one configuration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import brick_env_cfg
from .rl_cfg import brick_ppo_runner_cfg

# (suffix, training stage, randomization, grip enabled)
for suffix, stage, randomized, enabled in (
  ("", "full", False, True),
  ("-DR", "full", True, True),
  ("-Hold", "hold", False, True),
  ("-Engage", "engage", False, True),
  ("-Place", "place", False, True),
  ("-NoGrip", "full", False, False),
):
  register_mjlab_task(
    task_id="Mjlab-Microspine-Brick-Phybot-C2" + suffix,
    env_cfg=brick_env_cfg(stage=stage, randomize=randomized, grip_enabled=enabled),
    play_env_cfg=brick_env_cfg(play=True, stage=stage, randomize=randomized, grip_enabled=enabled),
    rl_cfg=brick_ppo_runner_cfg(),
  )
