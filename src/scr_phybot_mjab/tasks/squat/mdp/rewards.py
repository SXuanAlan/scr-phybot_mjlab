"""Reward terms for full-body squat motion tracking.

Every policy joint in the motion (legs, waist, and arms) is tracked.  The neck is
not a policy joint and is intentionally absent from the motion dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .motion_command import SquatMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def track_joint_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  joint_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Track the reference joint angles (optionally a subset, e.g. the legs)."""
  command = cast(SquatMotionCommand, env.command_manager.get_term(command_name))
  if joint_names is None:
    ref_ids = slice(None)
    robot_ids = command.joint_ids
  else:
    ref_ids = torch.tensor(
      [command.cfg.joint_names.index(n) for n in joint_names],
      dtype=torch.long,
      device=env.device,
    )
    robot_ids = command.joint_ids[ref_ids]
  error = torch.sum(
    torch.square(command.joint_pos_ref[:, ref_ids] - command.robot.data.joint_pos[:, robot_ids]),
    dim=-1,
  )
  return torch.exp(-error / std**2)


def track_joint_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  joint_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Track reference joint velocities for the selected motion joints."""
  command = cast(SquatMotionCommand, env.command_manager.get_term(command_name))
  if joint_names is None:
    ref_ids = slice(None)
    robot_ids = command.joint_ids
  else:
    ref_ids = torch.tensor(
      [command.cfg.joint_names.index(n) for n in joint_names],
      dtype=torch.long,
      device=env.device,
    )
    robot_ids = command.joint_ids[ref_ids]
  error = torch.sum(
    torch.square(command.joint_vel_ref[:, ref_ids] - command.robot.data.joint_vel[:, robot_ids]),
    dim=-1,
  )
  return torch.exp(-error / std**2)


def _selected_joint_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  joint_names: tuple[str, ...],
  *,
  velocity: bool,
) -> torch.Tensor:
  """Return per-joint reference error in motion joint order."""
  command = cast(SquatMotionCommand, env.command_manager.get_term(command_name))
  ref_ids = torch.tensor(
    [command.cfg.joint_names.index(name) for name in joint_names],
    dtype=torch.long,
    device=env.device,
  )
  robot_ids = command.joint_ids[ref_ids]
  if velocity:
    return command.robot.data.joint_vel[:, robot_ids] - command.joint_vel_ref[:, ref_ids]
  return command.robot.data.joint_pos[:, robot_ids] - command.joint_pos_ref[:, ref_ids]


def _smooth_l1(absolute_error: torch.Tensor, beta: float) -> torch.Tensor:
  """Elementwise smooth-L1 with a non-saturating linear tail."""
  if beta <= 0.0:
    raise ValueError(f"beta must be positive, got {beta}.")
  quadratic = torch.clamp(absolute_error, max=beta)
  return 0.5 * torch.square(quadratic) / beta + absolute_error - quadratic


def track_joint_pos_cauchy(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  joint_names: tuple[str, ...],
  metric_prefix: str,
  worst_joint_weight: float = 0.0,
) -> torch.Tensor:
  """Dense independent joint-position tracking with polynomial tails.

  Scoring joints independently avoids the saturation of a grouped exponential.
  A worst-joint component prevents seven good joints from hiding one visibly
  incorrect joint in the arithmetic mean.
  """
  if not 0.0 <= worst_joint_weight <= 1.0:
    raise ValueError(f"worst_joint_weight must be in [0, 1], got {worst_joint_weight}.")
  error = _selected_joint_error(env, command_name, joint_names, velocity=False)
  squared_error = torch.square(error)
  rmse = torch.sqrt(torch.mean(squared_error, dim=1))
  env.extras["log"][f"Metrics/motion/{metric_prefix}_pos_rmse"] = rmse.mean()
  env.extras["log"][f"Metrics/motion/{metric_prefix}_pos_error_max"] = torch.abs(error).max()
  mean_absolute_error = torch.abs(error).mean(dim=0)
  for joint_id, joint_name in enumerate(joint_names):
    env.extras["log"][f"Metrics/motion/{metric_prefix}_{joint_name}_pos_mae"] = mean_absolute_error[joint_id]
  joint_score = 1.0 / (1.0 + squared_error / std**2)
  mean_score = torch.mean(joint_score, dim=1)
  worst_score = torch.min(joint_score, dim=1).values
  env.extras["log"][f"Metrics/motion/{metric_prefix}_worst_joint_score"] = worst_score.mean()
  return (1.0 - worst_joint_weight) * mean_score + worst_joint_weight * worst_score


def joint_pos_tracking_huber_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  beta: float,
  joint_names: tuple[str, ...],
  reduction: str = "mean",
) -> torch.Tensor:
  """Dense smooth-L1 position error for motion joints.

  Its linear tail keeps pulling heavily displaced joints back toward the motion
  after bounded similarity rewards have become small.
  """
  error = torch.abs(_selected_joint_error(env, command_name, joint_names, velocity=False))
  loss = _smooth_l1(error, beta)
  if reduction == "mean":
    return torch.mean(loss, dim=1)
  if reduction == "sum":
    return torch.sum(loss, dim=1)
  raise ValueError(f"Unsupported reduction {reduction!r}; expected 'mean' or 'sum'.")


def track_joint_vel_cauchy(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  joint_names: tuple[str, ...],
  metric_prefix: str,
) -> torch.Tensor:
  """Dense independent velocity tracking for selected motion joints."""
  error = _selected_joint_error(env, command_name, joint_names, velocity=True)
  squared_error = torch.square(error)
  env.extras["log"][f"Metrics/motion/{metric_prefix}_vel_rmse"] = torch.sqrt(torch.mean(squared_error))
  return torch.mean(1.0 / (1.0 + squared_error / std**2), dim=1)


def joint_action_limit_cost(
  env: ManagerBasedRlEnv,
  action_name: str,
  joint_names: tuple[str, ...],
  max_abs_actions: tuple[float, ...],
  metric_prefix: str,
  reduction: str = "mean",
) -> torch.Tensor:
  """Penalize selected raw actions outside configured safe envelopes."""
  action_term = env.action_manager.get_term(action_name)
  action_ids = torch.tensor(
    [action_term.target_names.index(name) for name in joint_names],
    dtype=torch.long,
    device=env.device,
  )
  if len(max_abs_actions) != len(joint_names):
    raise ValueError(f"max_abs_actions has {len(max_abs_actions)} values for {len(joint_names)} joints.")
  limits = torch.tensor(max_abs_actions, device=env.device)
  action = action_term.raw_action[:, action_ids]
  excess = torch.clamp(torch.abs(action) - limits, min=0.0)

  env.extras["log"][f"Metrics/motion/{metric_prefix}_action_abs_max"] = torch.abs(action).max()
  env.extras["log"][f"Metrics/motion/{metric_prefix}_action_limit_fraction"] = (
    (torch.abs(action) > limits).float().mean()
  )

  # Squat currently has one action term, so its raw-action order is the action
  # manager order. Keep this explicit instead of silently logging wrong rates
  # if another action term is added later.
  if env.action_manager.total_action_dim != len(action_term.target_names):
    raise ValueError("joint_action_limit_cost requires a single action term.")
  previous_action = env.action_manager.prev_action[:, action_ids]
  env.extras["log"][f"Metrics/motion/{metric_prefix}_action_rate_rms"] = torch.sqrt(
    torch.mean(torch.square(action - previous_action))
  )
  cost = torch.square(excess)
  if reduction == "mean":
    return torch.mean(cost, dim=1)
  if reduction == "sum":
    return torch.sum(cost, dim=1)
  raise ValueError(f"Unsupported reduction {reduction!r}; expected 'mean' or 'sum'.")


def _selected_joint_action_and_reference(
  env: ManagerBasedRlEnv,
  command_name: str,
  action_name: str,
  joint_names: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return raw actions and the raw actions that target the current reference."""
  command = cast(SquatMotionCommand, env.command_manager.get_term(command_name))
  ref_ids = torch.tensor(
    [command.cfg.joint_names.index(name) for name in joint_names],
    dtype=torch.long,
    device=env.device,
  )
  robot_ids = command.joint_ids[ref_ids]
  action_term = env.action_manager.get_term(action_name)
  action_ids = torch.tensor(
    [action_term.target_names.index(name) for name in joint_names],
    dtype=torch.long,
    device=env.device,
  )

  scale = action_term.scale
  if isinstance(scale, torch.Tensor):
    scale = scale[:, action_ids]
  offset = action_term.offset
  if isinstance(offset, torch.Tensor):
    offset = offset[:, action_ids]
  encoder_bias = command.robot.data.encoder_bias[:, robot_ids]
  reference_action = (command.joint_pos_ref[:, ref_ids] - offset + encoder_bias) / scale
  return action_term.raw_action[:, action_ids], reference_action


def joint_action_reference_tube_cost(
  env: ManagerBasedRlEnv,
  command_name: str,
  action_name: str,
  joint_names: tuple[str, ...],
  max_abs_deviation: tuple[float, ...],
  beta: float,
  metric_prefix: str,
) -> torch.Tensor:
  """Keep selected actions near the current motion target, with correction room."""
  if len(max_abs_deviation) != len(joint_names):
    raise ValueError(f"max_abs_deviation has {len(max_abs_deviation)} values for {len(joint_names)} joints.")
  action, reference_action = _selected_joint_action_and_reference(env, command_name, action_name, joint_names)
  deviation = torch.abs(action - reference_action)
  tolerance = torch.tensor(max_abs_deviation, device=env.device)
  excess = torch.clamp(deviation - tolerance, min=0.0)
  env.extras["log"][f"Metrics/motion/{metric_prefix}_action_reference_rmse"] = torch.sqrt(
    torch.mean(torch.square(action - reference_action))
  )
  env.extras["log"][f"Metrics/motion/{metric_prefix}_action_tube_fraction"] = (deviation > tolerance).float().mean()
  return torch.sum(_smooth_l1(excess, beta), dim=1)


def joint_pair_symmetry_huber_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  left_joint_names: tuple[str, ...],
  right_joint_names: tuple[str, ...],
  beta: float,
  metric_prefix: str,
) -> torch.Tensor:
  """Penalize left/right position-error differences relative to the motion."""
  if len(left_joint_names) != len(right_joint_names):
    raise ValueError("Left and right joint groups must have equal length.")
  left_error = _selected_joint_error(env, command_name, left_joint_names, velocity=False)
  right_error = _selected_joint_error(env, command_name, right_joint_names, velocity=False)
  symmetry_error = torch.abs(left_error - right_error)
  env.extras["log"][f"Metrics/motion/{metric_prefix}_pos_symmetry_rmse"] = torch.sqrt(
    torch.mean(torch.square(symmetry_error))
  )
  return torch.sum(_smooth_l1(symmetry_error, beta), dim=1)


def joint_action_pair_symmetry_huber_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  action_name: str,
  left_joint_names: tuple[str, ...],
  right_joint_names: tuple[str, ...],
  beta: float,
  metric_prefix: str,
) -> torch.Tensor:
  """Penalize asymmetric action corrections around symmetric motion targets."""
  if len(left_joint_names) != len(right_joint_names):
    raise ValueError("Left and right joint groups must have equal length.")
  left_action, left_reference = _selected_joint_action_and_reference(env, command_name, action_name, left_joint_names)
  right_action, right_reference = _selected_joint_action_and_reference(
    env, command_name, action_name, right_joint_names
  )
  symmetry_error = torch.abs((left_action - left_reference) - (right_action - right_reference))
  env.extras["log"][f"Metrics/motion/{metric_prefix}_action_symmetry_rmse"] = torch.sqrt(
    torch.mean(torch.square(symmetry_error))
  )
  return torch.sum(_smooth_l1(symmetry_error, beta), dim=1)


def track_base_position(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """Track the full reference base position relative to the environment origin."""
  command = cast(SquatMotionCommand, env.command_manager.get_term(command_name))
  position = command.robot.data.root_link_pos_w - env.scene.env_origins
  error = torch.sum(torch.square(command.root_position_ref - position), dim=-1)
  return torch.exp(-error / std**2)


def track_base_lin_vel(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """Track the reference world-frame base linear velocity."""
  command = cast(SquatMotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.root_lin_vel_ref - command.robot.data.root_link_lin_vel_w),
    dim=-1,
  )
  return torch.exp(-error / std**2)


def track_base_orientation(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """Track the complete reference base orientation, including yaw."""
  command = cast(SquatMotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.root_quat_ref, command.robot.data.root_link_quat_w)
  return torch.exp(-torch.square(error) / std**2)


def track_feet_stationary(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward both reference feet remaining planted and motionless.

  The generated squat pins both foot sites for the complete clip.  Requiring
  contact and tracking zero XY foot velocity closes the foot-tapping loophole
  that allowed rapid foot taps while joint angles stayed near the reference.
  The score is computed independently per foot with a Cauchy kernel so a large
  early-training slip does not saturate the complete two-foot learning signal.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  foot_velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  squared_speed = torch.sum(torch.square(foot_velocity), dim=-1)
  both_in_contact = (sensor.data.found > 0).all(dim=1).float()
  foot_speed = torch.sqrt(squared_speed)
  env.extras["log"]["Metrics/motion/foot_speed_xy_mean"] = foot_speed.mean()
  env.extras["log"]["Metrics/motion/foot_speed_xy_worst_mean"] = foot_speed.max(dim=1).values.mean()
  env.extras["log"]["Metrics/motion/both_feet_contact_fraction"] = both_in_contact.mean()
  score = torch.mean(1.0 / (1.0 + squared_speed / std**2), dim=1)
  return score * both_in_contact


def feet_xy_velocity_huber_error(
  env: ManagerBasedRlEnv,
  beta: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Non-saturating cost for horizontal foot slip, applied to both feet."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  foot_speed = torch.norm(foot_velocity, dim=-1)
  return torch.mean(_smooth_l1(foot_speed, beta), dim=1)


def track_feet_anchor(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Track the motion's absolute planted-foot XY positions.

  Unlike a velocity-only reward, this accumulates slow creep into a growing
  position error. It is intentionally not contact-gated so a lifted foot keeps
  a dense gradient back toward its reference foothold. Each foot is scored
  independently with a Cauchy kernel to retain signal after a large drift.
  """
  command = cast(SquatMotionCommand, env.command_manager.get_term(command_name))
  foot_xy = command.robot.data.site_pos_w[:, asset_cfg.site_ids, :2]
  reference_xy = command.foot_position_ref_w[:, :, :2]
  error_xy = torch.norm(foot_xy - reference_xy, dim=-1)
  env.extras["log"]["Metrics/motion/error_foot_xy_mean"] = error_xy.mean()
  env.extras["log"]["Metrics/motion/error_foot_xy_max"] = error_xy.max()
  return torch.mean(1.0 / (1.0 + torch.square(error_xy) / std**2), dim=1)


def feet_xy_anchor_huber_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  beta: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Non-saturating cost for each foot's horizontal anchor displacement."""
  command = cast(SquatMotionCommand, env.command_manager.get_term(command_name))
  foot_xy = command.robot.data.site_pos_w[:, asset_cfg.site_ids, :2]
  reference_xy = command.foot_position_ref_w[:, :, :2]
  error_xy = torch.norm(foot_xy - reference_xy, dim=-1)
  return torch.mean(_smooth_l1(error_xy, beta), dim=1)
