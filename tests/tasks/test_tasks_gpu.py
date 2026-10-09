"""Every registered task builds, resets and steps on MuJoCo Warp (run with pytest --gpu).

The config tests check what each task is; this checks that it runs: the
scene compiles, every manager term finds its entities, sensors and commands,
and zero and random actions keep observations and rewards finite.
"""

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

import scr_phybot_mjab.tasks  # noqa: F401  (registers the project tasks)

pytestmark = pytest.mark.gpu

TASKS = sorted(task for task in list_tasks() if "Phybot" in task)


def _assert_finite(value, where: str) -> None:
  if isinstance(value, dict):
    for key, item in value.items():
      _assert_finite(item, f"{where}/{key}")
  else:
    assert torch.isfinite(value).all(), where


@pytest.mark.parametrize("play", [False, True], ids=["train", "play"])
@pytest.mark.parametrize("task", TASKS)
@torch.inference_mode()
def test_task_builds_resets_and_steps(task, play):
  cfg = load_env_cfg(task, play=play)
  cfg.scene.num_envs, cfg.seed = 2, 0
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    obs, _ = env.reset(seed=0)
    _assert_finite(obs, "reset")
    shape = (env.num_envs, env.action_manager.total_action_dim)
    # A private CPU generator, so the actions never shift the env's RNG streams.
    generator = torch.Generator().manual_seed(1)
    for step in range(12):
      actions = torch.zeros(shape) if step < 6 else 0.3 * torch.randn(shape, generator=generator)
      obs, reward, terminated, truncated, _ = env.step(actions.to(env.device))
      _assert_finite(obs, f"step {step}")
      assert torch.isfinite(reward).all(), f"reward at step {step}"
      done = (terminated | truncated).nonzero().flatten()
      if len(done):
        env.reset(env_ids=done)
  finally:
    env.close()
