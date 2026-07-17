"""Curriculum functions for the wall-climbing task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

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


class ClimbDifficultyStage(TypedDict):
  """One performance-gated climbing stage.

  ``gait_ee_names`` restricts the regrip cycle to a limb subset (feet first:
  compression-loaded pads are the low-risk limbs to practice release/re-latch
  on). ``step_height`` shrinks the commanded upward step so early regrips are
  reachable by exploration. Omitted keys fall back to the task defaults.
  """

  angle_deg: float
  initial_engaged_ee_names: tuple[str, ...]
  required_productive_regrips: int
  randomize_start_limb: bool
  gait_ee_names: NotRequired[tuple[str, ...]]
  step_height: NotRequired[float]


class performance_gated_climb:
  """Advance grip initialization and wall angle from episode success rate."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del env
    self.level = 0
    self.evaluated_episodes = 0
    self.successful_episodes = 0
    self.last_success_rate = 0.0
    stages = cfg.params.get("stages", [])
    if not stages:
      raise ValueError("performance_gated_climb requires at least one stage.")

  @staticmethod
  def _set_gravity(
    env: ManagerBasedRlEnv,
    angle_deg: float,
    gravity_magnitude: float,
  ) -> None:
    import warp as wp

    rad = math.radians(angle_deg)
    grav_view = wp.to_torch(env.sim.wp_model.opt.gravity)
    grav_view[:, 0] = gravity_magnitude * math.cos(rad)
    grav_view[:, 1] = 0.0
    grav_view[:, 2] = -gravity_magnitude * math.sin(rad)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    *,
    stages: list[ClimbDifficultyStage],
    gait_command_name: str = "climb_gait",
    affordance_command_name: str = "microspine_affordance",
    min_evaluated_episodes: int = 512,
    advance_success_rate: float = 0.7,
    gravity_magnitude: float = 9.81,
  ) -> dict[str, torch.Tensor]:
    if not 0.0 <= advance_success_rate <= 1.0:
      raise ValueError("advance_success_rate must be in [0, 1].")
    if min_evaluated_episodes < 1:
      raise ValueError("min_evaluated_episodes must be positive.")

    gait = env.command_manager.get_term(gait_command_name)
    completed = env.episode_length_buf[env_ids] > 0
    successes = gait.episode_success[env_ids] & completed
    self.evaluated_episodes += int(completed.sum().item())
    self.successful_episodes += int(successes.sum().item())

    if self.evaluated_episodes >= min_evaluated_episodes:
      self.last_success_rate = (
        self.successful_episodes / self.evaluated_episodes
      )
      if (
        self.last_success_rate >= advance_success_rate
        and self.level + 1 < len(stages)
      ):
        self.level += 1
      self.evaluated_episodes = 0
      self.successful_episodes = 0

    stage = stages[self.level]
    affordance = env.command_manager.get_term(affordance_command_name)
    affordance.set_initial_engaged_ee_names(stage["initial_engaged_ee_names"])
    gait.set_training_stage(
      required_productive_regrips=stage["required_productive_regrips"],
      randomize_start_limb=stage["randomize_start_limb"],
      step_height=stage.get("step_height"),
      gait_ee_names=stage.get("gait_ee_names"),
    )
    self._set_gravity(env, stage["angle_deg"], gravity_magnitude)

    device = env.device
    return {
      "stage": torch.tensor(float(self.level), device=device),
      "angle_deg": torch.tensor(stage["angle_deg"], device=device),
      "success_rate": torch.tensor(self.last_success_rate, device=device),
      "required_regrips": torch.tensor(
        float(stage["required_productive_regrips"]), device=device
      ),
    }


class ClimbVelocityRangeStage(TypedDict, total=False):
  iter: int
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  lin_vel_z: tuple[float, float]
  vx_range: tuple[float, float]
  vy_range: tuple[float, float]
  vz_range: tuple[float, float]


def resolve_gravity_angle(current_iter: int, angle_stages: list[WallAngleStage]) -> float:
  """Resolve the effective gravity curriculum angle by stepwise stage lookup."""
  if not angle_stages:
    raise ValueError("angle_stages must contain at least one stage.")

  current_angle = angle_stages[0]["angle_deg"]
  for stage in sorted(angle_stages, key=lambda item: item["iter"]):
    if current_iter >= stage["iter"]:
      current_angle = stage["angle_deg"]
    else:
      break
  return current_angle


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
