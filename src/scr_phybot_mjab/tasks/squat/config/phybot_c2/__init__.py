"""PHYBot C2 squat task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import phybot_c2_squat_env_cfg
from .rl_cfg import phybot_c2_squat_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Squat-Phybot-C2",
  env_cfg=phybot_c2_squat_env_cfg(),
  play_env_cfg=phybot_c2_squat_env_cfg(play=True),
  rl_cfg=phybot_c2_squat_ppo_runner_cfg(),
)
