"""RSL-RL configuration for the PHYBot C2 80-degree transition task.

Climb's PPO recipe, including its 96-step rollout that the curricula's
STEPS_PER_ITER counts in; only the logging identity changes.
"""

import dataclasses

from mjlab.rl import RslRlOnPolicyRunnerCfg

from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.rl_cfg import (
  phybot_c2_ppo_runner_cfg,
)


def phybot_c2_climb_transition80_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return dataclasses.replace(
    phybot_c2_ppo_runner_cfg(),
    experiment_name="phybot_c2_climb_transition80",
    wandb_tags=("phybot_c2", "transition80"),
  )
