"""PHYBot C2 80-degree transition task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import phybot_c2_climb_transition80_env_cfg
from .rl_cfg import phybot_c2_climb_transition80_ppo_runner_cfg

# Stand at an 80-degree wall and latch all four pads, tracking the a80 mount
# clip with the real robot's PD table (actor 238, critic 419).
register_mjlab_task(
  task_id="Mjlab-Climb-Transition80-Phybot-C2",
  env_cfg=phybot_c2_climb_transition80_env_cfg(),
  play_env_cfg=phybot_c2_climb_transition80_env_cfg(play=True),
  rl_cfg=phybot_c2_climb_transition80_ppo_runner_cfg(),
)
