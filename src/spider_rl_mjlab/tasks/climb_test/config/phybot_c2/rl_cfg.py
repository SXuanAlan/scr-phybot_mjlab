"""RL configuration for PHYBot C2 climb-test."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from spider_rl_mjlab.tasks.climb_wall.config.phybot_c2.rl_cfg import (
  phybot_c2_ppo_runner_cfg as phybot_c2_climb_wall_ppo_runner_cfg,
)


def phybot_c2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = phybot_c2_climb_wall_ppo_runner_cfg()
  cfg.experiment_name = "phybot_c2_climb_test"
  cfg.wandb_project = "spider-climb-test"
  cfg.wandb_tags = ("phybot_c2", "climb_wall_clone", "gravity_0deg")
  return cfg
