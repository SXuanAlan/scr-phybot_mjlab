"""RSL-RL configuration for the PHYBot C2 walk->climb transition task.

The climb PPO recipe verbatim — the env contract is identical by
construction, so the hyperparameters transfer with it. Only the logging
identity changes.
"""

import dataclasses

from mjlab.rl import RslRlOnPolicyRunnerCfg
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.rl_cfg import (
  phybot_c2_ppo_runner_cfg,
)


def phybot_c2_climb_transition_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return dataclasses.replace(
    phybot_c2_ppo_runner_cfg(),
    experiment_name="phybot_c2_climb_transition",
    wandb_tags=("phybot_c2", "transition"),
  )
