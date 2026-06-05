"""PHYBot Big V2 wall-climbing task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import phybot_big_v2_climb_wall_env_cfg
from .rl_cfg import phybot_big_v2_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Climb-Wall-Phybot-Big-V2",
  env_cfg=phybot_big_v2_climb_wall_env_cfg(),
  play_env_cfg=phybot_big_v2_climb_wall_env_cfg(play=True),
  rl_cfg=phybot_big_v2_ppo_runner_cfg(),
)
