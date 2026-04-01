"""Custom termination functions for the wall-climbing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def bad_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  max_height: float = 10.0,
  max_velocity: float = 50.0,
) -> torch.Tensor:
  """Terminate when simulation diverges (position or velocity explodes)."""
  asset: Entity = env.scene[asset_cfg.name]
  pos = asset.data.root_link_pos_w
  vel = asset.data.root_link_lin_vel_w

  pos_bad = torch.isnan(pos).any(dim=1) | (pos.abs() > max_height).any(dim=1)
  vel_bad = torch.isnan(vel).any(dim=1) | (vel.abs() > max_velocity).any(dim=1)
  return pos_bad | vel_bad
