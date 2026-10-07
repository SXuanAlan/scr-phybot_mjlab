"""Observation terms shared by the PHYBot C2 tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

ROBOT_ASSET_CFG = SceneEntityCfg("robot")
"""Default asset of the robot observation terms."""


def projected_gravity_from_framequat_sensor(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = ROBOT_ASSET_CFG,
) -> torch.Tensor:
  """Gravity direction in the base frame, from the base frame-quaternion sensor.

  Uses the asset's nominal world-down vector: no task changes gravity
  (climbing tilts the wall instead).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  asset: Entity = env.scene[asset_cfg.name]
  quat_w = sensor.data
  quat_w = quat_w / torch.linalg.norm(quat_w, dim=-1, keepdim=True).clamp_min(1.0e-6)
  return quat_apply_inverse(quat_w, asset.data.gravity_vec_w)
