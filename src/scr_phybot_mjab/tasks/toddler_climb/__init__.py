"""ToddlerBot climbing with one microspine unit per limb."""

from mjlab.tasks.registry import register_mjlab_task

from scr_phybot_mjab.tasks.squat.config.phybot_c2.rl_cfg import phybot_c2_squat_ppo_runner_cfg
from scr_phybot_mjab.tools.climbing_reference.toddlerbot import REFERENCE_PATH

from .env_cfg import toddler_climb_env_cfg


def runner_cfg(angle=90):
    cfg = phybot_c2_squat_ppo_runner_cfg()
    cfg.experiment_name = f"toddlerbot_climb_single_unit_a{angle}"
    cfg.logger = "tensorboard"
    cfg.wandb_project = "scr_phybot_mjab-toddlerbot-climb"
    cfg.wandb_tags = ("toddlerbot", "climb", "single-unit")
    cfg.actor.distribution_cfg["init_std"] = 0.15
    cfg.num_steps_per_env = 96
    return cfg


register_mjlab_task(
    task_id="Mjlab-Climb-Wall-ToddlerBot-SingleUnit",
    env_cfg=toddler_climb_env_cfg(),
    play_env_cfg=toddler_climb_env_cfg(play=True),
    rl_cfg=runner_cfg(90),
)

register_mjlab_task(
    task_id="Mjlab-Climb-Wall-ToddlerBot-SingleUnit-A60",
    env_cfg=toddler_climb_env_cfg(
        angle_deg=60.0, reference_file=str(REFERENCE_PATH.with_name("toddlerbot_climb_a60.npz"))
    ),
    play_env_cfg=toddler_climb_env_cfg(
        play=True, angle_deg=60.0, reference_file=str(REFERENCE_PATH.with_name("toddlerbot_climb_a60.npz"))
    ),
    rl_cfg=runner_cfg(60),
)
