"""Curriculum functions for the wall-climbing task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class RewardWeightStage(TypedDict):
  iter: int
  weight: float


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
  steps_per_iter: int = 24,
) -> torch.Tensor:
  """Update a reward term's weight based on training iteration stages.

  Args:
    steps_per_iter: Number of env.step() calls per training iteration
      (must match num_steps_per_env in rl_cfg).
  """
  del env_ids  # Unused.
  current_iter = env.common_step_counter // steps_per_iter
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if current_iter > stage["iter"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


class WallAngleStage(TypedDict):
  iter: int
  angle_deg: float  # effective wall angle from horizontal. 90 = vertical wall
  # (gravity straight down). 30 = gentle 30°-from-horizontal "slope" — modeled
  # by tilting GRAVITY while keeping the wall geometry vertical.


def _interp_angle(current_iter: int, stages: list[WallAngleStage]) -> float:
  """Piecewise-linear interpolation of wall angle over iterations."""
  if current_iter <= stages[0]["iter"]:
    return stages[0]["angle_deg"]
  for i in range(len(stages) - 1):
    a, b = stages[i], stages[i + 1]
    if a["iter"] <= current_iter <= b["iter"]:
      span = max(b["iter"] - a["iter"], 1)
      alpha = (current_iter - a["iter"]) / span
      return a["angle_deg"] + alpha * (b["angle_deg"] - a["angle_deg"])
  return stages[-1]["angle_deg"]


def gravity_angle(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  angle_stages: list[WallAngleStage],
  gravity_magnitude: float = 9.81,
  steps_per_iter: int = 24,
) -> torch.Tensor:
  """Curriculum that varies gravity direction to simulate climbing on a slope
  of angle ``angle_deg`` from horizontal, while keeping the wall geometry
  vertical and removing the need for a floor.

  At ``angle_deg = 90°``: gravity = (0, 0, -g) — standard vertical wall climb.
  At ``angle_deg = 30°``: gravity = g · (cos 30°, 0, -sin 30°) — tilted so it
  mostly pushes the robot INTO the wall (+x), with only a small component
  along the wall's up direction. This is physically equivalent to the robot
  sitting on a gentle 30° slope.

  Writes gravity in-place on ``env.sim.wp_model.opt.gravity`` via a torch
  view — never reallocates the warp array (CUDA graphs would otherwise still
  read from the old array). ``gravity_vec_w`` on each entity is deliberately
  NOT updated: the orientation rewards (``torso_orientation``,
  ``flat_orientation_l2``) continue to measure alignment with world-vertical,
  which is the correct "up-the-wall" reference for climbing.
  """
  del env_ids  # Unused.
  import warp as wp

  current_iter = env.common_step_counter // steps_per_iter
  angle_deg = _interp_angle(current_iter, angle_stages)
  rad = math.radians(angle_deg)
  gx = gravity_magnitude * math.cos(rad)
  gz = -gravity_magnitude * math.sin(rad)

  grav_view = wp.to_torch(env.sim.wp_model.opt.gravity)  # (N, 3) view, N ∈ {1, num_envs}
  grav_view[:, 0] = gx
  grav_view[:, 1] = 0.0
  grav_view[:, 2] = gz

  return torch.tensor([angle_deg], device=env.device)
