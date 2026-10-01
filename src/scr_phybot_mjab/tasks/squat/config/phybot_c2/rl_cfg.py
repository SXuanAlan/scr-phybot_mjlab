"""RSL-RL configuration for the PHYBot C2 squat tracking task.

DeepMimic-style tracking needs no AMP discriminator, so this is the plain PPO
baseline shared with walk.
"""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from scr_phybot_mjab.tasks.common.rl_cfg import plain_ppo_runner_cfg


def phybot_c2_squat_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return plain_ppo_runner_cfg(
    experiment_name="phybot_c2_squat",
    wandb_project="scr_phybot_mjab-squat",
    wandb_tags=("phybot_c2", "squat"),
    max_iterations=15_000,
  )
