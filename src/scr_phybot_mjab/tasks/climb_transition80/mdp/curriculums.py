"""Transition80 curricula: generic stepwise schedulers (own copies).

Stage lookup convention (shared by every scheduler here): ``stages`` is a
list of ``{"iter": int, "value": float}``; the active value is the last
stage whose ``iter`` <= the cumulative learning iteration
``common_step_counter // steps_per_iter``. The resume path RESTORES the
counter, so stage iters are cumulative across a lineage (08-18 finding).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "microspine_param_schedule",
  "pad_gimbal_travel_schedule",
  "reward_param_schedule",
  "stage_interp_at",
  "stage_value_at",
  "termination_param_schedule",
)


def stage_value_at(stages: list[dict], iteration: int) -> float:
  """Pure stage lookup (also the exam scripts' helper)."""
  current = stages[0]["value"]
  for stage in sorted(stages, key=lambda item: item["iter"]):
    if iteration >= stage["iter"]:
      current = stage["value"]
    else:
      break
  return float(current)


def stage_interp_at(stages: list[dict], iteration: int) -> float:
  """Piecewise-LINEAR stage lookup: holds the first value before the first
  stage, interpolates between consecutive stages, holds the last after."""
  ordered = sorted(stages, key=lambda item: item["iter"])
  if iteration <= ordered[0]["iter"]:
    return float(ordered[0]["value"])
  for lo, hi in zip(ordered[:-1], ordered[1:]):
    if lo["iter"] <= iteration <= hi["iter"]:
      span = max(int(hi["iter"]) - int(lo["iter"]), 1)
      u = (iteration - int(lo["iter"])) / span
      return float(lo["value"]) + u * (float(hi["value"]) - float(lo["value"]))
  return float(ordered[-1]["value"])


def microspine_param_schedule(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  param_name: str,
  stages: list[dict],
  command_name: str = "microspine_affordance",
  steps_per_iter: int = 96,
) -> torch.Tensor:
  """Stage a live MicrospineCfg scalar by training iteration (writes the
  affordance's live ``_mcfg`` — the weld-shear ignition ladder's path)."""
  del env_ids
  current = stage_value_at(stages, env.common_step_counter // steps_per_iter)
  affordance = env.command_manager.get_term(command_name)
  setattr(affordance._mcfg, param_name, current)
  return torch.tensor([current], device=env.device)


def termination_param_schedule(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  termination_name: str,
  param_name: str,
  stages: list[dict],
  steps_per_iter: int = 96,
) -> torch.Tensor:
  """Stage a termination term's PARAM by training iteration (mutates the
  live termination cfg's params dict)."""
  del env_ids
  current = stage_value_at(stages, env.common_step_counter // steps_per_iter)
  env.termination_manager.get_term_cfg(termination_name).params[param_name] = current
  return torch.tensor([current], device=env.device)


def reward_param_schedule(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  param_name: str,
  stages: list[dict],
  steps_per_iter: int = 96,
  interpolate: bool = False,
) -> torch.Tensor:
  """Stage a reward term's PARAM by training iteration (mutates the live
  reward cfg's params dict). ``interpolate`` = piecewise-linear ramp between
  the stages instead of steps (v5 rip bill)."""
  del env_ids
  iteration = env.common_step_counter // steps_per_iter
  current = stage_interp_at(stages, iteration) if interpolate else stage_value_at(stages, iteration)
  env.reward_manager.get_term_cfg(reward_name).params[param_name] = current
  return torch.tensor([current], device=env.device)


@requires_model_fields("jnt_range")
def pad_gimbal_travel_schedule(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  stages: list[dict],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*_gimbal_[xyz]",)),
  steps_per_iter: int = 96,
) -> None:
  """EVENT (startup + reset): pad-gimbal travel ladder, +-deg by iteration.

  v5 (2026-08-23, user design): angle compliance as a curriculum — the
  passive wrist ball joints open at +-15 deg and anneal (piecewise-linear)
  to the +-3 deg real stop, the way climb annealed its tilt-peel band.
  Writes the plant's per-world ``jnt_range`` of the 12 gimbal hinges for the
  envs being reset (every episode is <= 21 s, so the whole fleet follows the
  ladder within one episode); ``startup`` seeds all worlds. The stops'
  solref stays the constant block's. Clips, the validator and the probes
  are untouched (they read the constants module).
  """
  iteration = env.common_step_counter // steps_per_iter
  travel = math.radians(stage_interp_at(stages, iteration))
  asset = env.scene[asset_cfg.name]
  joint_ids = asset.indexing.joint_ids[asset_cfg.joint_ids]
  joint_ids = torch.as_tensor(joint_ids, device=env.device, dtype=torch.int)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)
  field = env.sim.model.jnt_range
  env_grid, jnt_grid = torch.meshgrid(env_ids, joint_ids, indexing="ij")
  field[env_grid, jnt_grid] = torch.tensor([-travel, travel], device=field.device, dtype=field.dtype)
