"""RSL-RL configuration of the brick task."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from scr_phybot_mjab.tasks.common.rl_cfg import plain_ppo_runner_cfg


def brick_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = plain_ppo_runner_cfg(
    experiment_name="phybot_c2_microspine_brick",
    wandb_project="scr_phybot_mjab-brick",
    wandb_tags=("phybot_c2", "microspine", "brick"),
    max_iterations=15_000,
  )
  cfg.logger = "tensorboard"
  # Actions are residuals around the reference, so exploration starts small.
  cfg.actor.distribution_cfg["init_std"] = 0.2
  return cfg
