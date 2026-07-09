"""PHYBot C2 wall-climbing task registration.

Two A/B arms of the same task:

  * ``Mjlab-Climb-Wall-Phybot-C2`` — constraint mode (per-EE connect-equality
    virtual anchors + limit-surface release; the anisotropic microspine
    model). Default going forward.
  * ``Mjlab-Climb-Wall-Phybot-C2-Legacy`` — legacy adhesion mode (600 N
    external wall-normal force + mu=30 contacts), kept as the baseline.

Note: checkpoints do not transfer between arms or from pre-M4 runs — the
critic observation group gained privileged microspine terms.
"""

import dataclasses

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import phybot_c2_climb_wall_env_cfg
from .rl_cfg import phybot_c2_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Climb-Wall-Phybot-C2",
  env_cfg=phybot_c2_climb_wall_env_cfg(attachment_mode="constraint"),
  play_env_cfg=phybot_c2_climb_wall_env_cfg(play=True, attachment_mode="constraint"),
  rl_cfg=dataclasses.replace(
    phybot_c2_ppo_runner_cfg(),
    wandb_tags=("phybot_c2", "microspine_constraint"),
  ),
)

register_mjlab_task(
  task_id="Mjlab-Climb-Wall-Phybot-C2-Legacy",
  env_cfg=phybot_c2_climb_wall_env_cfg(attachment_mode="adhesion"),
  play_env_cfg=phybot_c2_climb_wall_env_cfg(play=True, attachment_mode="adhesion"),
  rl_cfg=dataclasses.replace(
    phybot_c2_ppo_runner_cfg(),
    wandb_tags=("phybot_c2", "microspine_legacy_adhesion"),
  ),
)
