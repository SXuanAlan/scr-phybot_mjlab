"""RSL-RL configuration for the PHYBot C2 walk and stand tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from scr_phybot_mjab.tasks.common.rl_cfg import plain_ppo_runner_cfg


def phybot_c2_walk_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return plain_ppo_runner_cfg(
    experiment_name="phybot_c2_walk",
    wandb_project="scr_phybot_mjab-walk",
    wandb_tags=("phybot_c2", "walk"),
    max_iterations=30_000,
  )


def phybot_c2_stand_default_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Walk's plain PPO, logged under the stand names."""
  return plain_ppo_runner_cfg(
    experiment_name="phybot_c2_stand_default",
    wandb_project="scr_phybot_mjab-stand",
    wandb_tags=("phybot_c2", "stand"),
    max_iterations=30_000,
  )
