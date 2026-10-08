"""Build a task's environment and load a trained actor, as the evaluation and export scripts do.

Typical use::

  cfg = load_env_cfg(task, play=True)
  ...  # adjust cfg (env count, ranges)
  env = make_env(task, cfg, device)
  runner, policy = load_policy(task, env, checkpoint, device)
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

__all__ = ("load_policy", "make_env")


def make_env(
  task: str, cfg: ManagerBasedRlEnvCfg, device: str, *, render_mode: str | None = None
) -> RslRlVecEnvWrapper:
  """The task's environment, wrapped for RSL-RL with the task's action clip."""
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=render_mode)
  return RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task).clip_actions)


def load_policy(task: str, env: RslRlVecEnvWrapper, checkpoint: str | Path, device: str):
  """The task's runner holding the checkpoint's actor, and its inference policy.

  The actor loads strictly, observation normalizer included, so a
  checkpoint that does not match the task's observation layout fails here.
  """
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(load_rl_cfg(task)), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  return runner, runner.get_inference_policy(device=device)
