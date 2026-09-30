"""Dual-hand microspine brick task registration."""

from mjlab.tasks.registry import register_mjlab_task

from scr_phybot_mjab.tasks.squat.config.phybot_c2.rl_cfg import phybot_c2_squat_ppo_runner_cfg

from .env_cfg import brick_env_cfg


def runner_cfg():
  cfg = phybot_c2_squat_ppo_runner_cfg()
  cfg.experiment_name = "phybot_c2_microspine_brick"
  cfg.logger = "tensorboard"
  cfg.actor.distribution_cfg["init_std"] = 0.2
  cfg.wandb_project = "scr_phybot_mjab-brick"
  cfg.wandb_tags = ("phybot_c2", "microspine", "brick")
  return cfg


for suffix, stage, randomized, enabled in (
  ("", "full", False, True),
  ("-DR", "full", True, True),
  ("-Hold", "hold", False, True),
  ("-Engage", "engage", False, True),
  ("-Place", "place", False, True),
  ("-NoGrip", "full", False, False),
):
  register_mjlab_task(
    task_id="Mjlab-Microspine-Brick-Phybot-C2" + suffix,
    env_cfg=brick_env_cfg(stage=stage, randomize=randomized, grip_enabled=enabled),
    play_env_cfg=brick_env_cfg(play=True, stage=stage, randomize=randomized, grip_enabled=enabled),
    rl_cfg=runner_cfg(),
  )
