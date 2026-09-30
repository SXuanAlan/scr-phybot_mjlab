"""Record one fixed-seed rollout of a task; run as a module in a fresh process.

  python -m scr_phybot_mjab.tools.equivalence.rollout TASK --mode train \
      --actions zero --steps 150 --num-envs 8 --seed 1234 --out FILE.npz

MuJoCo Warp is not bitwise deterministic, so tiny run-to-run differences grow
chaotically after contact events. Every ``--segment-steps`` steps all envs are
therefore reset with a fresh per-segment seed; each short segment stays in
the low-noise regime while RSI spreads the segments over the whole clip.

The npz holds one array per signal stacked over steps (``reset/*`` stacked
over segments, ``log/*`` from ``extras["log"]`` with NaN where absent); the
json sidecar holds the run settings and term names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scr_phybot_mjab.tools.equivalence.tasks import configure_determinism

ACTION_SOURCES = ("zero", "random", "reference")
RANDOM_ACTION_STD = 0.3


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("task")
  parser.add_argument("--mode", choices=("train", "play"), default="train")
  parser.add_argument("--actions", choices=ACTION_SOURCES, default="zero")
  parser.add_argument("--steps", type=int, default=150)
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--segment-steps", type=int, default=5, help="reset every N steps; 0 = never")
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args(argv)
  configure_determinism()
  record_rollout(
    args.task, mode=args.mode, actions=args.actions, steps=args.steps,
    num_envs=args.num_envs, seed=args.seed, segment_steps=args.segment_steps, out=args.out,
  )


def record_rollout(
  task: str, *, mode: str, actions: str, steps: int, num_envs: int, seed: int, segment_steps: int, out: Path
) -> None:
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(task, play=mode == "play")
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  next_action = _action_source(actions, env, steps=steps, seed=seed)
  recorder = _Recorder(env)
  with torch.inference_mode():
    for step in range(steps):
      if step == 0 or (segment_steps and step % segment_steps == 0):
        env.reset(seed=seed + (step // segment_steps if segment_steps else 0))
        recorder.capture_reset()
      env.step(next_action(step))
      recorder.capture_step()
  meta = {
    "task": task, "mode": mode, "actions": actions, "steps": steps,
    "num_envs": num_envs, "seed": seed, "segment_steps": segment_steps,
  }
  recorder.save(out, meta)
  env.close()


def _action_source(kind: str, env: Any, *, steps: int, seed: int) -> Callable[[int], Any]:
  import torch

  shape = (env.num_envs, env.action_manager.total_action_dim)
  if kind == "zero":
    zeros = torch.zeros(shape, device=env.device)
    return lambda step: zeros
  if kind == "random":
    # A private CPU generator: drawing these never shifts the env's RNG streams.
    generator = torch.Generator().manual_seed(seed + 1)
    table = (torch.randn((steps, *shape), generator=generator) * RANDOM_ACTION_STD).to(env.device)
    return lambda step: table[step]
  return _ReferenceActions(env)


class _ReferenceActions:
  """Joint-position actions that replay the motion clip's reference pose."""

  def __init__(self, env: Any) -> None:
    import torch

    self._env = env
    self._motion = _motion_command(env)
    clip_column = {joint: column for column, joint in enumerate(self._motion._joint_ids.tolist())}
    self._terms = []
    start = 0
    for term in env.action_manager._terms.values():
      targets = getattr(term, "_target_ids", None)
      if targets is not None:
        joints = targets.tolist()
        columns = torch.tensor([clip_column.get(j, 0) for j in joints], device=env.device)
        in_clip = torch.tensor([j in clip_column for j in joints], device=env.device)
        self._terms.append((start, term, columns, in_clip))
      start += term.action_dim
    self._action_dim = start

  def __call__(self, step: int) -> Any:
    import torch

    action = torch.zeros(self._env.num_envs, self._action_dim, device=self._env.device)
    reference = self._motion.ref_joint_pos
    for start, term, columns, in_clip in self._terms:
      target = (reference[:, columns] - term._offset) / term._scale
      action[:, start:start + term.action_dim] = torch.where(in_clip, target, 0.0)
    return action


def _motion_command(env: Any) -> Any:
  for term in env.command_manager._terms.values():
    if hasattr(term, "ref_joint_pos") and hasattr(term, "_joint_ids"):
      return term
  raise RuntimeError("Task has no clip-driven motion command; use zero or random actions")


class _Recorder:
  def __init__(self, env: Any) -> None:
    self._env = env
    self._resets: dict[str, list[np.ndarray]] = {}
    self._steps: dict[str, list[np.ndarray]] = {}
    self._logs: list[dict[str, float]] = []

  def capture_reset(self) -> None:
    for key, value in _to_numpy(self._state()).items():
      self._resets.setdefault(key, []).append(value)

  def capture_step(self) -> None:
    for key, value in _to_numpy({**self._state(), **self._signals()}).items():
      self._steps.setdefault(key, []).append(value)
    self._logs.append(_log_scalars(self._env.extras.get("log", {})))

  def _state(self) -> dict[str, Any]:
    env = self._env
    state = {}
    for group, value in env.obs_buf.items():
      if isinstance(value, dict):
        state.update({f"obs/{group}/{term}": tensor for term, tensor in value.items()})
      else:
        state[f"obs/{group}"] = value
    for name, term in env.command_manager._terms.items():
      state.update(_command_state(name, term, env.num_envs))
    data = env.sim.data
    state.update({"sim/qpos": data.qpos, "sim/qvel": data.qvel, "sim/ctrl": data.ctrl})
    state["env/episode_length"] = env.episode_length_buf
    return state

  def _signals(self) -> dict[str, Any]:
    env = self._env
    signals = {
      "reward/terms": env.reward_manager._step_reward,
      "reward/total": env.reward_buf,
      "done/terminated": env.reset_terminated,
      "done/time_outs": env.reset_time_outs,
    }
    for name in env.termination_manager.active_terms:
      signals[f"done/{name}"] = env.termination_manager.get_term(name)
    metrics = getattr(env.metrics_manager, "_step_values", None)
    if metrics is not None:
      signals["metrics"] = metrics
    return signals

  def save(self, path: Path, meta: dict) -> None:
    env = self._env
    arrays = {f"reset/{key}": np.stack(values) for key, values in self._resets.items()}
    arrays.update({key: np.stack(values) for key, values in self._steps.items()})
    arrays.update(_log_arrays(self._logs))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    meta.update(
      reward_terms=list(env.reward_manager.active_terms),
      termination_terms=list(env.termination_manager.active_terms),
      metric_terms=list(getattr(env.metrics_manager, "active_terms", [])),
      observation_terms={k: list(v) for k, v in env.observation_manager.active_terms.items()},
      command_terms=list(env.command_manager._terms),
    )
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")


def _command_state(name: str, term: Any, num_envs: int) -> dict[str, Any]:
  """Per-env tensors a command term holds; private ones are kept apart."""
  import torch

  state = {}
  command = term.command
  if isinstance(command, torch.Tensor):
    state[f"cmd/{name}/command"] = command
  for attr, value in vars(term).items():
    if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == num_envs:
      prefix = "cmd_private" if attr.startswith("_") else "cmd"
      state[f"{prefix}/{name}/{attr}"] = value
  return state


def _to_numpy(tensors: dict[str, Any]) -> dict[str, np.ndarray]:
  return {key: value.detach().cpu().numpy().copy() for key, value in tensors.items()}


def _log_arrays(logs: list[dict[str, float]]) -> dict[str, np.ndarray]:
  """One float64 array per extras-log key, NaN on steps that did not log it."""
  keys = sorted({key for log in logs for key in log})
  return {f"log/{key}": np.array([log.get(key, np.nan) for log in logs]) for key in keys}


def _log_scalars(log: dict) -> dict[str, float]:
  scalars = {}
  for key, value in log.items():
    if hasattr(value, "numel"):
      if value.numel() != 1:
        continue
      value = value.item()
    if isinstance(value, (int, float)):
      scalars[key] = float(value)
  return scalars


if __name__ == "__main__":
  main()
