"""Build a task's environment and load a trained actor, as the evaluation and export scripts do.

Typical use::

  cfg = load_env_cfg(task, play=True)
  ...  # adjust cfg (env count, ranges)
  env = make_env(task, cfg, device)
  runner, policy = load_policy(task, env, checkpoint, device)

resolve_checkpoint finds a local or W&B checkpoint, and pin_climb_wall_angle
evaluates a climb checkpoint on the wall angle its training reached.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path

from scr_phybot_mjab.tasks.climb_wall.mdp.curriculums import resolve_wall_angle

__all__ = ("checkpoint_iteration", "load_policy", "make_env", "pin_climb_wall_angle", "resolve_checkpoint")


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


def resolve_checkpoint(
  experiment_name: str,
  checkpoint_file: str | None,
  wandb_run_path: str | None,
  wandb_checkpoint_name: str | None = None,
) -> tuple[Path, str]:
  """The checkpoint to load and a one-line description of where it came from.

  A local ``checkpoint_file`` wins; otherwise the checkpoint comes from the
  W&B run, cached under logs/rsl_rl/<experiment_name>.
  """
  if checkpoint_file is not None:
    path = Path(checkpoint_file)
    if not path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path, path.name
  if wandb_run_path is None:
    raise ValueError("`wandb_run_path` is required when `checkpoint_file` is not provided.")
  log_root = (Path("logs") / "rsl_rl" / experiment_name).resolve()
  path, was_cached = get_wandb_checkpoint_path(log_root, Path(wandb_run_path), wandb_checkpoint_name)
  return path, f"{path.name} (run: {path.parent.name}, {'cached' if was_cached else 'downloaded'})"


def checkpoint_iteration(path_or_name: str | Path | None) -> int | None:
  """The training iteration in a checkpoint name such as model_1250.pt, or None."""
  if path_or_name is None:
    return None
  name = Path(path_or_name).name
  match = re.search(r"(?:model|checkpoint|ckpt)[_-](\d+)", name)
  if match is None:
    match = re.search(r"(\d+)(?=\.[^.]+$)", name)
  return int(match.group(1)) if match is not None else None


def pin_climb_wall_angle(
  env_cfg: ManagerBasedRlEnvCfg,
  *,
  wall_angle_deg: float | None,
  checkpoint: str | Path | None,
  checkpoint_hint: str | None = None,
) -> None:
  """Pin a climb task's wall-angle curriculum to one angle and print it.

  ``wall_angle_deg`` wins; otherwise the angle is the curriculum stage the
  checkpoint's iteration reached (read from its name, or from
  ``checkpoint_hint``). Tasks without a wall-angle curriculum are untouched.
  """
  curriculum = getattr(env_cfg, "curriculum", None)
  if not curriculum or "wall_angle" not in curriculum:
    return
  term_cfg = curriculum["wall_angle"]
  stages = term_cfg.params.get("angle_stages")
  if not stages:
    return
  if wall_angle_deg is not None:
    angle_deg, source = float(wall_angle_deg), "CLI override"
  else:
    iteration = checkpoint_iteration(checkpoint)
    if iteration is None:
      iteration = checkpoint_iteration(checkpoint_hint)
    if iteration is None:
      print(
        "[WARN]: Could not infer the wall angle because the checkpoint name does "
        "not look like model_<iter>.pt; keeping the curriculum at its initial angle."
      )
      return
    angle_deg, source = float(resolve_wall_angle(iteration, stages)), f"checkpoint iter {iteration}"
  term_cfg.params["angle_stages"] = [{"iter": 0, "angle_deg": angle_deg}]
  print(f"[INFO]: Wall angle set to {angle_deg:.2f} deg ({source})")
