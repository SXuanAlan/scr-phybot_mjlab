"""Default friction test task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import friction_test_env_cfg
from .rl_cfg import friction_test_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Friction-Test",
  env_cfg=friction_test_env_cfg(),
  play_env_cfg=friction_test_env_cfg(play=True),
  rl_cfg=friction_test_ppo_runner_cfg(),
)
