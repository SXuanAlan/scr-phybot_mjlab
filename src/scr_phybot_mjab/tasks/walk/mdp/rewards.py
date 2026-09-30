from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-xy_squared / std**2)


def feet_flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward keeping the foot soles flat while in ground contact.

  For each foot body (asset_cfg.body_ids) the gravity vector is projected into
  the body frame; a flat sole means zero xy components. Feet not in contact
  contribute nothing, so swing feet are free to pitch.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
  num_envs, num_feet = body_quat_w.shape[:2]
  gravity_w = asset.data.gravity_vec_w
  if gravity_w.dim() == 1:
    gravity_w = gravity_w.unsqueeze(0).expand(num_envs, 3)
  gravity_w = gravity_w.unsqueeze(1).expand(num_envs, num_feet, 3)
  projected = quat_apply_inverse(
    body_quat_w.reshape(num_envs * num_feet, 4),
    gravity_w.reshape(num_envs * num_feet, 3),
  ).reshape(num_envs, num_feet, 3)
  xy_squared = torch.sum(torch.square(projected[..., :2]), dim=-1)  # [B, N]
  flatness = torch.exp(-xy_squared / std**2)  # [B, N]
  return torch.sum(flatness * in_contact, dim=1) / num_feet


def feet_lateral_distance(
  env: ManagerBasedRlEnv,
  min_dist: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize the lateral (base-frame y) foot separation dropping below min_dist.

  The policy otherwise converges to standing with the feet touching: the shoe
  shells then clip each other on micro-shuffles and trip the self-collision
  termination. Uses |y_left - y_right| so site ordering does not matter;
  crossed feet pass through zero separation and get the maximum penalty.
  """
  asset: Entity = env.scene[asset_cfg.name]
  pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, 2, 3]
  rel = pos_w - asset.data.root_link_pos_w.unsqueeze(1)
  num_envs, num_sites = rel.shape[:2]
  quat = asset.data.root_link_quat_w.unsqueeze(1).expand(num_envs, num_sites, 4)
  pos_b = quat_apply_inverse(quat.reshape(-1, 4), rel.reshape(-1, 3)).reshape(num_envs, num_sites, 3)
  separation = (pos_b[:, 0, 1] - pos_b[:, 1, 1]).abs()
  return (min_dist - separation).clamp(min=0)


def stand_still_joint_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.05,
  start_time_s: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint motion when the commanded velocity is (near) zero."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  standing = (total_command < command_threshold).float()
  active = (env.episode_length_buf.float() * env.step_dt >= start_time_s).float()
  joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return standing * active * torch.sum(torch.abs(joint_vel), dim=1)


def feet_contact_when_standing(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Reward having all feet on the ground when commanded to stand still."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  standing = (total_command < command_threshold).float()
  all_feet_down = (contact_sensor.data.found > 0).all(dim=1).float()
  return standing * all_feet_down


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    env.extras["log"]["Metrics/self_collision_force_max"] = force_mag.max()
    env.extras["log"]["Metrics/self_collision_hit_rate"] = hit.any(dim=-1).float().mean()
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  env.extras["log"]["Metrics/self_collision_hit_rate"] = (
    data.found.reshape(data.found.shape[0], -1).any(dim=-1).float().mean()
  )
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(num_in_air, min=1)
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros((env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32)
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def base_height(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward keeping the base at a target height (anti-crouch).

  Uses absolute world z, so only valid on flat terrain.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2]
  error = torch.square(height - target_height)
  return torch.exp(-error / std**2)


def preset_forward_velocity_band(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.05,
  low_ratio: float = 0.6,
  high_ratio: float = 1.5,
  std: float = 0.25,
  lateral_std: float = 0.35,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward broad forward-speed bands for preset commands, not exact tracking."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_x = command[:, 0]
  active = (torch.abs(cmd_x) > command_threshold).float()
  target_speed = torch.abs(cmd_x)
  signed_actual = asset.data.root_link_lin_vel_b[:, 0] * torch.sign(cmd_x).clamp(min=-1.0, max=1.0)
  low = target_speed * low_ratio
  high = target_speed * high_ratio
  outside_band = (low - signed_actual).clamp(min=0.0) + (signed_actual - high).clamp(min=0.0)
  lateral_vel = asset.data.root_link_lin_vel_b[:, 1]
  env.extras["log"]["Metrics/preset_forward_actual"] = torch.mean(signed_actual * active)
  return torch.exp(-torch.square(outside_band) / std**2 - torch.square(lateral_vel) / lateral_std**2) * active


def preset_yaw_velocity_band(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.05,
  low_ratio: float = 0.5,
  high_ratio: float = 1.6,
  std: float = 0.25,
  roll_pitch_std: float = 0.35,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward broad yaw-rate bands for turn presets, not exact tracking."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_yaw = command[:, 2]
  active = (torch.abs(cmd_yaw) > command_threshold).float()
  target_rate = torch.abs(cmd_yaw)
  signed_actual = asset.data.root_link_ang_vel_b[:, 2] * torch.sign(cmd_yaw).clamp(min=-1.0, max=1.0)
  low = target_rate * low_ratio
  high = target_rate * high_ratio
  outside_band = (low - signed_actual).clamp(min=0.0) + (signed_actual - high).clamp(min=0.0)
  roll_pitch = torch.norm(asset.data.root_link_ang_vel_b[:, :2], dim=1)
  env.extras["log"]["Metrics/preset_yaw_actual"] = torch.mean(signed_actual * active)
  return torch.exp(-torch.square(outside_band) / std**2 - torch.square(roll_pitch) / roll_pitch_std**2) * active


class energy:
  """Penalize mechanical power: L2 norm of |torque * joint_vel| over joints.

  Matches the official PhyBot Lab ``mdp.energy`` reward. Actuator forces are
  reported in actuator order while joint velocities are in joint order, so the
  two are aligned by name at init (builtin actuators are named after their
  target joint).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    actuator_names = cfg.params["asset_cfg"].actuator_names or (".*",)
    actuator_ids, matched_names = asset.find_actuators(actuator_names)
    joint_ids, joint_names = asset.find_joints(matched_names, preserve_order=True)
    assert list(joint_names) == list(matched_names), f"Actuator/joint name mismatch: {matched_names} vs {joint_names}"
    self.actuator_ids = torch.tensor(actuator_ids, device=env.device)
    self.joint_ids = torch.tensor(joint_ids, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    torque = asset.data.actuator_force[:, self.actuator_ids]
    joint_vel = asset.data.joint_vel[:, self.joint_ids]
    return torch.norm(torque * joint_vel, dim=-1)


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(num_in_contact, min=1)
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def feet_stationary_when_standing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.05,
  missing_contact_cost: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot XY motion and contact loss while commanded to stand."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  standing = (total_command < command_threshold).float()
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  foot_speed = torch.norm(foot_vel_xy, dim=-1)
  motion_cost = torch.sum(torch.square(foot_speed), dim=1)
  assert contact_sensor.data.found is not None
  missing_contact = (contact_sensor.data.found == 0).float()
  contact_cost = missing_contact_cost * torch.sum(missing_contact, dim=1)
  env.extras["log"]["Metrics/stand_foot_speed_mean"] = torch.sum(foot_speed * standing.unsqueeze(1)) / torch.clamp(
    standing.sum() * foot_speed.shape[1], min=1.0
  )
  env.extras["log"]["Metrics/stand_missing_contact_mean"] = torch.sum(
    missing_contact * standing.unsqueeze(1)
  ) / torch.clamp(standing.sum() * missing_contact.shape[1], min=1.0)
  return standing * (motion_cost + contact_cost)


def base_motion_when_standing(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.05,
  start_time_s: float = 0.0,
  lin_std: float = 0.2,
  yaw_std: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize base drift/turning under the stand preset."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  standing = (total_command < command_threshold).float()
  active = (env.episode_length_buf.float() * env.step_dt >= start_time_s).float()
  lin_xy = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
  yaw = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
  cost = torch.square(lin_xy / lin_std) + torch.square(yaw / yaw_std)
  return standing * active * cost


class feet_lock_after_recovery:
  """Stateful stand reward that freezes the final footholds.

  During the recovery window the policy may step and the anchor follows the
  feet. At the boundary, each environment freezes its current world-frame foot
  XY positions. From then until reset, foot speed, accumulated drift, and lost
  contact are penalized. The frozen anchor closes the slow-creep loophole that
  an instantaneous slip penalty cannot detect.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg, env
    self.anchor_xy: torch.Tensor | None = None
    self.frozen: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    *,
    sensor_name: str,
    recovery_time_s: float,
    speed_scale: float,
    drift_scale: float,
    max_scaled_cost: float,
    missing_contact_cost: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None

    foot_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]
    shape = foot_xy.shape
    if self.anchor_xy is None or tuple(self.anchor_xy.shape) != tuple(shape):
      self.anchor_xy = foot_xy.clone()
      self.frozen = torch.zeros(shape[0], dtype=torch.bool, device=env.device)

    anchor_xy = self.anchor_xy
    frozen = self.frozen
    assert anchor_xy is not None and frozen is not None

    elapsed_s = env.episode_length_buf.float() * env.step_dt
    strict = elapsed_s >= recovery_time_s
    recovering = ~strict
    anchor_xy[recovering] = foot_xy[recovering]
    newly_strict = strict & ~frozen
    anchor_xy[newly_strict] = foot_xy[newly_strict]
    frozen[newly_strict] = True

    foot_speed = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
    foot_drift = torch.norm(foot_xy - anchor_xy, dim=-1)
    speed_cost = torch.square(foot_speed / speed_scale).clamp(max=max_scaled_cost)
    drift_cost = torch.square(foot_drift / drift_scale).clamp(max=max_scaled_cost)
    missing_contact = (sensor.data.found == 0).float()
    cost = torch.sum(speed_cost + drift_cost + missing_contact_cost * missing_contact, dim=1)
    strict_f = strict.float()

    denom = torch.clamp(strict_f.sum() * foot_speed.shape[1], min=1.0)
    env.extras["log"]["Metrics/stand_foot_speed_mean"] = torch.sum(foot_speed * strict_f.unsqueeze(1)) / denom
    env.extras["log"]["Metrics/stand_foot_drift_mean"] = torch.sum(foot_drift * strict_f.unsqueeze(1)) / denom
    env.extras["log"]["Metrics/stand_missing_contact_mean"] = torch.sum(missing_contact * strict_f.unsqueeze(1)) / denom
    # Keep the per-step training diagnostic reduction cheap. The checkpoint
    # acceptance script computes the strict-window p95 over complete rollouts.
    env.extras["log"]["Metrics/stand_foot_drift_max"] = torch.max(foot_drift * strict_f.unsqueeze(1))
    return strict_f * cost

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self.anchor_xy is None or self.frozen is None:
      return
    if env_ids is None:
      env_ids = slice(None)
    self.anchor_xy[env_ids] = 0.0
    self.frozen[env_ids] = False


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(std_standing, device=env.device, dtype=torch.float32)

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = ((total_speed >= walking_threshold) & (total_speed < running_threshold)).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
