"""Forearm+hand microspine test task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import forearm_hand_microspine_test_env_cfg
from .rl_cfg import forearm_hand_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Microspine-Test-Forearm-Hand",
  env_cfg=forearm_hand_microspine_test_env_cfg(),
  play_env_cfg=forearm_hand_microspine_test_env_cfg(play=True),
  rl_cfg=forearm_hand_ppo_runner_cfg(),
)
