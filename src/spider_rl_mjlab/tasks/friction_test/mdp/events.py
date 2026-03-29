"""Custom events for the friction test task.

Applies a constant force perpendicular to the slope surface (pressing block into slope).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def apply_normal_force(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  force_magnitude: float = 10.0,
  slope_angle_deg: float = 30.0,
) -> None:
  """Apply a constant force perpendicular to the slope surface, pressing the block into it.

  The force direction is computed from the slope angle:
    - normal to slope = (-sin(angle), 0, -cos(angle)) in world frame
  This pushes the block into the slope surface.

  Args:
    env: The RL environment.
    env_ids: Environment indices (unused — applied to all envs).
    asset_cfg: The block entity.
    force_magnitude: Magnitude of the normal force [N].
    slope_angle_deg: Slope inclination angle [degrees], must match the scene slope.
  """
  asset: Entity = env.scene[asset_cfg.name]
  num_envs = env.num_envs

  rad = math.radians(slope_angle_deg)
  # Normal pointing into slope surface (away from block, into the slope)
  nx = -math.sin(rad)
  nz = -math.cos(rad)
  fx = force_magnitude * nx
  fz = force_magnitude * nz

  forces = torch.tensor(
    [fx, 0.0, fz], device=env.device, dtype=torch.float32
  ).expand(num_envs, 1, 3).clone()

  torques = torch.zeros(num_envs, 1, 3, device=env.device)
  asset.write_external_wrench_to_sim(forces, torques, body_ids=[0])
