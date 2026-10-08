"""PHYBot C2 walk, stand and AMP walk registration: three variants of one configuration."""

from mjlab.tasks.registry import register_mjlab_task

from scr_phybot_mjab.amp_rsl_rl import MjlabAmpRunner

from .amp_rl_cfg import phybot_c2_walk_amp_ppo_runner_cfg
from .env_cfgs import phybot_c2_walk_env_cfg
from .rl_cfg import phybot_c2_stand_default_ppo_runner_cfg, phybot_c2_walk_ppo_runner_cfg

for task_id, variant, rl_cfg, runner_cls in (
  ("Mjlab-Walk-Phybot-C2", "walk", phybot_c2_walk_ppo_runner_cfg(), None),
  ("Mjlab-Stand-Default-Phybot-C2", "stand", phybot_c2_stand_default_ppo_runner_cfg(), None),
  ("Mjlab-Walk-Amp-Phybot-C2", "amp", phybot_c2_walk_amp_ppo_runner_cfg(), MjlabAmpRunner),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=phybot_c2_walk_env_cfg(variant=variant),
    play_env_cfg=phybot_c2_walk_env_cfg(play=True, variant=variant),
    rl_cfg=rl_cfg,
    runner_cls=runner_cls,
  )
