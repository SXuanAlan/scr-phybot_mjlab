"""Custom reward functions for the microspine test task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def hand_on_wall(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  wall_x: float = 0.0,
) -> torch.Tensor:
  """Reward for keeping the root body near wall_x (proxy for hand contact).

  Forearm hangs vertically; root (elbow) at x=0 means hand sphere is pressing
  against wall inner face at x=0. Returns exp(-|dx|/0.05).
  """
  asset: Entity = env.scene[asset_cfg.name]
  root_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  dist = torch.abs(root_x - wall_x)
  return torch.exp(-dist / 0.05)


def hand_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward for maintaining height (resisting the downward force)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2]
