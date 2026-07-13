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
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  foot_velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  velocity_error = torch.sum(torch.square(foot_velocity), dim=(1, 2))
  both_in_contact = (sensor.data.found > 0).all(dim=1).float()
  foot_speed = torch.norm(foot_velocity, dim=-1)
  env.extras["log"]["Metrics/motion/foot_speed_xy_mean"] = foot_speed.mean()
  env.extras["log"]["Metrics/motion/both_feet_contact_fraction"] = (
    both_in_contact.mean()
  )
  return torch.exp(-velocity_error / std**2) * both_in_contact


def track_feet_anchor(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Track the motion's absolute planted-foot XY positions.

  Unlike a velocity-only reward, this accumulates slow creep into a growing
  position error. It is intentionally not contact-gated so a lifted foot keeps
  a dense gradient back toward its reference foothold.
  """
  command = cast(SquatMotionCommand, env.command_manager.get_term(command_name))
  foot_xy = command.robot.data.site_pos_w[:, asset_cfg.site_ids, :2]
  reference_xy = command.foot_position_ref_w[:, :, :2]
  error_xy = torch.norm(foot_xy - reference_xy, dim=-1)
  env.extras["log"]["Metrics/motion/error_foot_xy_mean"] = error_xy.mean()
  env.extras["log"]["Metrics/motion/error_foot_xy_max"] = error_xy.max()
  return torch.exp(-torch.sum(torch.square(error_xy), dim=1) / std**2)
