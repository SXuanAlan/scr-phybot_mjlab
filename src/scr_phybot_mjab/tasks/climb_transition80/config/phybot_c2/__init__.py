"""PHYBot C2 80-degree transition task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import phybot_c2_climb_transition80_env_cfg
from .rl_cfg import phybot_c2_climb_transition80_ppo_runner_cfg

# The v2 observation contract (actor 238: proprio + BeyondMimic reference
# lanes + pad faces + wall face corners; critic 419), on the a80 prologue at
# a wall pinned to 80 deg, real-robot PD table: BeyondMimic tracking
# rewards, one engage-progress potential, termination-first legality with
# the engage constraint cone in CaT mode.
register_mjlab_task(
  task_id="Mjlab-Climb-Transition80-Phybot-C2",
  env_cfg=phybot_c2_climb_transition80_env_cfg(),
  play_env_cfg=phybot_c2_climb_transition80_env_cfg(play=True),
  rl_cfg=phybot_c2_climb_transition80_ppo_runner_cfg(),
)
