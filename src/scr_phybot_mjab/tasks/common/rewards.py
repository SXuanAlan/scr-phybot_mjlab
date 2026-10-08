"""Reward terms shared by the PHYBot C2 walk and squat tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ("energy", "feet_lateral_distance", "self_collision_cost")

_ROBOT_ASSET_CFG = SceneEntityCfg("robot")


def feet_lateral_distance(
  env: ManagerBasedRlEnv,
  min_dist: float,
  asset_cfg: SceneEntityCfg = _ROBOT_ASSET_CFG,
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
    asset_cfg: SceneEntityCfg = _ROBOT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    torque = asset.data.actuator_force[:, self.actuator_ids]
    joint_vel = asset.data.joint_vel[:, self.joint_ids]
    return torch.norm(torque * joint_vel, dim=-1)
