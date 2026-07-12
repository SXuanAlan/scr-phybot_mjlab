"""PHYBot C2 walk task registration."""

from mjlab.tasks.registry import register_mjlab_task

from scr_phybot_mjab.amp_rsl_rl import MjlabAmpRunner

from .env_cfgs import (
  phybot_c2_stand_default_env_cfg,
  phybot_c2_walk_env_cfg,
  phybot_c2_walk_preset_amp_realpd_env_cfg,
)
from .amp_rl_cfg import phybot_c2_walk_amp_ppo_runner_cfg
from .rl_cfg import (
  phybot_c2_stand_default_ppo_runner_cfg,
  phybot_c2_walk_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Walk-Phybot-C2",
  env_cfg=phybot_c2_walk_env_cfg(),
  play_env_cfg=phybot_c2_walk_env_cfg(play=True),
  rl_cfg=phybot_c2_walk_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Stand-Default-Phybot-C2",
  env_cfg=phybot_c2_stand_default_env_cfg(),
  play_env_cfg=phybot_c2_stand_default_env_cfg(play=True),
  rl_cfg=phybot_c2_stand_default_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Walk-Amp-Phybot-C2",
  env_cfg=phybot_c2_walk_preset_amp_realpd_env_cfg(),
  play_env_cfg=phybot_c2_walk_preset_amp_realpd_env_cfg(play=True),
  rl_cfg=phybot_c2_walk_amp_ppo_runner_cfg(),
  runner_cls=MjlabAmpRunner,
)
