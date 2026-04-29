"""Curriculum functions for the wall-climbing task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict, cast

import torch

from .commands import ClimbVelocityCommandCfg

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


class ClimbVelocityRangeStage(TypedDict, total=False):
  iter: int
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  lin_vel_z: tuple[float, float]
  vx_range: tuple[float, float]
  vy_range: tuple[float, float]
  vz_range: tuple[float, float]


def _interp_angle(current_iter: int, stages: list[WallAngleStage]) -> float:
  """Piecewise interpolation of wall angle over iterations, in sin(angle)
  space.

  Since along-wall gravity load equals g·sin(angle), interpolating the
  sin linearly gives a *uniform* physics-load ramp between waypoints.
  Linear-in-angle interpolation (the naive approach) makes load change
  fast early and slow late — exactly backwards for stable learning.
  """
  if current_iter <= stages[0]["iter"]:
    return stages[0]["angle_deg"]
  for i in range(len(stages) - 1):
    a, b = stages[i], stages[i + 1]
    if a["iter"] <= current_iter <= b["iter"]:
      span = max(b["iter"] - a["iter"], 1)
      alpha = (current_iter - a["iter"]) / span
      sin_a = math.sin(math.radians(a["angle_deg"]))
      sin_b = math.sin(math.radians(b["angle_deg"]))
      sin_mid = sin_a + alpha * (sin_b - sin_a)
      # clamp for numerical safety at the endpoint
      sin_mid = max(-1.0, min(1.0, sin_mid))
      return math.degrees(math.asin(sin_mid))
  return stages[-1]["angle_deg"]


def resolve_gravity_angle(current_iter: int, angle_stages: list[WallAngleStage]) -> float:
  """Resolve the effective gravity curriculum angle for a training iteration."""
  return _interp_angle(current_iter, angle_stages)


def resolve_climb_velocity_ranges(
  current_iter: int,
  range_stages: list[ClimbVelocityRangeStage],
) -> ClimbVelocityRangeStage:
  """Resolve command ranges for a training iteration.

  Stages are cumulative: later stages can specify only the axes they change.
  ``vx_range``/``vy_range``/``vz_range`` aliases are accepted for readability
  in climb-wall configs.
  """
  if not range_stages:
    raise ValueError("range_stages must contain at least one stage.")

  resolved: dict = {}
  for stage in sorted(range_stages, key=lambda item: item["iter"]):
    if current_iter >= stage["iter"]:
      resolved.update(stage)
  if not resolved:
    resolved.update(range_stages[0])
  return cast(ClimbVelocityRangeStage, resolved)


def _stage_range(
  stage: ClimbVelocityRangeStage, *keys: str
) -> tuple[float, float] | None:
  for key in keys:
    if key in stage:
      value = stage[key]  # type: ignore[literal-required]
      return (float(value[0]), float(value[1]))
  return None


def _apply_climb_velocity_range_stage(
  cfg: ClimbVelocityCommandCfg,
  stage: ClimbVelocityRangeStage,
) -> None:
  if (value := _stage_range(stage, "lin_vel_x", "vx_range")) is not None:
    cfg.ranges.lin_vel_x = value
  if (value := _stage_range(stage, "lin_vel_y", "vy_range")) is not None:
    cfg.ranges.lin_vel_y = value
  if (value := _stage_range(stage, "lin_vel_z", "vz_range")) is not None:
    cfg.ranges.lin_vel_z = value


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
  NOT updated: orientation terms that read ``EntityData.projected_gravity_b``
  continue to measure alignment with world-vertical, which is the correct
  "up-the-wall" reference for climbing. Observation terms that need the
  physical gravity direction should read ``env.sim.wp_model.opt.gravity``.
  """
  del env_ids  # Unused.
  import warp as wp

  current_iter = env.common_step_counter // steps_per_iter
  angle_deg = resolve_gravity_angle(current_iter, angle_stages)
  rad = math.radians(angle_deg)
  gx = gravity_magnitude * math.cos(rad)
  gz = -gravity_magnitude * math.sin(rad)

  grav_view = wp.to_torch(env.sim.wp_model.opt.gravity)  # (N, 3) view, N ∈ {1, num_envs}
  grav_view[:, 0] = gx
  grav_view[:, 1] = 0.0
  grav_view[:, 2] = gz

  return torch.tensor([angle_deg], device=env.device)


def climb_velocity_command_ranges(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  range_stages: list[ClimbVelocityRangeStage],
  steps_per_iter: int = 24,
) -> dict[str, torch.Tensor]:
  """Curriculum that updates climb velocity command ranges by iteration."""
  del env_ids  # Unused.
  current_iter = env.common_step_counter // steps_per_iter
  stage = resolve_climb_velocity_ranges(current_iter, range_stages)

  command_term = env.command_manager.get_term(command_name)
  cfg = command_term.cfg
  if not isinstance(cfg, ClimbVelocityCommandCfg):
    raise TypeError(
      f"Command term '{command_name}' must use ClimbVelocityCommandCfg, "
      f"got {type(cfg).__name__}."
    )
  _apply_climb_velocity_range_stage(cfg, stage)

  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0], device=env.device),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1], device=env.device),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0], device=env.device),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1], device=env.device),
    "lin_vel_z_min": torch.tensor(cfg.ranges.lin_vel_z[0], device=env.device),
    "lin_vel_z_max": torch.tensor(cfg.ranges.lin_vel_z[1], device=env.device),
  }
