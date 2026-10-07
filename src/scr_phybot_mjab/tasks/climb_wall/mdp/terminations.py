"""Custom termination functions for the wall-climbing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "bad_state",
  "motion_body_tracking_divergence",
  "prolonged_support_loss",
  "root_below_wall_altitude",
)


def root_below_wall_altitude(
  env: ManagerBasedRlEnv,
  minimum_altitude: float = 0.3,
  motion_command_name: str = "climb_motion",
) -> torch.Tensor:
  """Along-wall analog of ``root_height_below_minimum`` (the fall guard).

  ``(root - wall_base_line) . wall_up < minimum_altitude`` — world z on a
  vertical wall (bit-identical to the mjlab builtin there). The builtin's
  absolute world-z bar is a trap under wall tilt: the RSI spawn's world z
  shrinks with the angle (the whole reference tips toward the base line), so
  a legitimately-placed robot on a 45° wall could terminate at t=0.
  """
  motion = env.command_manager.get_term(motion_command_name)
  robot_pos = env.scene["robot"].data.root_link_pos_w
  altitude = ((robot_pos - motion.wall_pivot_w) * motion.wall_up_w).sum(dim=-1)
  return altitude < minimum_altitude


def motion_body_tracking_divergence(
  env,
  command_name="climb_motion",
  position_threshold=0.3,
  orientation_threshold=1.0,
  height_threshold=0.25,
):
  """End large body-pose departures; retain a wall-frame progress check.

  Pose errors use the same XY/yaw alignment as body tracking. Along-wall
  height and global root orientation prevent that alignment hiding a fall.
  These are failure thresholds, independent of reward error scales.
  """
  from mjlab.utils.lab_api.math import quat_error_magnitude

  motion = env.command_manager.get_term(command_name)
  reference_pos, _ = motion.aligned_body_reference()
  position_error = (reference_pos - motion.robot_body_pos_w).norm(dim=-1).amax(-1)
  orientation_error = quat_error_magnitude(motion.ref_root_quat_w, motion.robot_root_quat_w)
  height_error = ((motion.robot.data.root_link_pos_w - motion.ref_root_pos_w) * motion.wall_up_w).sum(-1).abs()
  return (
    (position_error > position_threshold)
    | (orientation_error > orientation_threshold)
    | (height_error > height_threshold)
  )


def prolonged_support_loss():
  """End a continuous no-anchor interval, allowing a short recovery attempt."""
  elapsed = None

  def call(env, command_name="microspine_affordance", timeout_s=0.4):
    nonlocal elapsed
    engaged = env.command_manager.get_term(command_name).engaged
    if elapsed is None:
      elapsed = torch.zeros(env.num_envs, device=env.device)
    elapsed = torch.where(engaged.any(-1), torch.zeros_like(elapsed), elapsed + env.step_dt)
    return elapsed >= timeout_s - 1.0e-7

  def reset(env_ids=None):
    if elapsed is not None:
      elapsed[slice(None) if env_ids is None else env_ids] = 0.0
    return {}

  call.reset = reset
  return call


def bad_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  max_displacement: float = 10.0,
  max_velocity: float = 50.0,
  max_ang_velocity: float = 200.0,
  max_joint_velocity: float = 200.0,
) -> torch.Tensor:
  """Terminate when simulation diverges (NaN/Inf or magnitude explodes).

  Pure numerical safety — covers every state tensor that feeds actor
  observations so divergence is caught before NaN propagates to the
  policy. Uses env-local position to avoid false positives from env
  spacing.
  """
  asset: Entity = env.scene[asset_cfg.name]
  pos = asset.data.root_link_pos_w - env.scene.env_origins
  lin_vel = asset.data.root_link_lin_vel_w
  ang_vel = asset.data.root_link_ang_vel_w
  quat = asset.data.root_link_quat_w
  jpos = asset.data.joint_pos
  jvel = asset.data.joint_vel

  def _bad(t: torch.Tensor, limit: float | None = None) -> torch.Tensor:
    b = (~torch.isfinite(t)).any(dim=1)
    if limit is not None:
      b = b | (t.abs() > limit).any(dim=1)
    return b

  return (
    _bad(pos, max_displacement)
    | _bad(lin_vel, max_velocity)
    | _bad(ang_vel, max_ang_velocity)
    | _bad(quat)
    | _bad(jpos)
    | _bad(jvel, max_joint_velocity)
  )
