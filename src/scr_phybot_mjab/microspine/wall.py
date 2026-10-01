"""World-frame geometry of the climbing wall entity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def wall_inward_normal(env: ManagerBasedRlEnv, wall_entity_name: str) -> torch.Tensor:
  """Wall inward normal (robot-to-wall, the wall's local +x) in world frame, (E, 3)."""
  wquat = env.scene[wall_entity_name].data.root_link_quat_w
  ww, wx, wy, wz = wquat[:, 0], wquat[:, 1], wquat[:, 2], wquat[:, 3]
  inward_x = 1.0 - 2.0 * (wy * wy + wz * wz)
  inward_y = 2.0 * (wx * wy + ww * wz)
  inward_z = 2.0 * (wx * wz - ww * wy)
  return torch.stack([inward_x, inward_y, inward_z], dim=-1)
