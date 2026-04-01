"""Custom reward functions for the wall-climbing task.

Ported from IsaacLab spider_rl_lab to mjlab API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def base_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Reward upward velocity (climbing speed) rather than absolute height.

  Returns clipped vz so only upward movement is rewarded.
  """
  asset: Entity = env.scene[asset_cfg.name]
  vz = asset.data.root_link_lin_vel_w[:, 2]
  return vz.clamp(min=0.0)


def torso_orientation(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Reward upright orientation using projected gravity.

  Returns exp(-||g_xy||^2) where g_xy are x,y components of gravity in the
  base frame. Value is 1.0 when perfectly upright, decays toward 0 when tilted.
  """
  asset: Entity = env.scene[asset_cfg.name]
  gravity_b = asset.data.projected_gravity_b
  return torch.exp(-torch.sum(torch.square(gravity_b[:, :2]), dim=1))


def near_wall(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, wall_x: float = 1.5
) -> torch.Tensor:
  """Reward proximity to the wall in the x-direction.

  Returns exp(-dist/0.1), highest when robot base is close to wall_x.
  """
  asset: Entity = env.scene[asset_cfg.name]
  root_pos_x_env = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  dist_to_wall = torch.abs(root_pos_x_env - wall_x)
  return torch.exp(-dist_to_wall / 0.1)


def torso_ang_vel_xy(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Penalize angular velocity around x and y axes (body shaking)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_w[:, :2]), dim=1)


def lin_vel_xy(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Penalize lateral linear velocity (x and y)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_lin_vel_w[:, :2]), dim=1)


def com_near_wall(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  wall_x: float = 0.3,
) -> torch.Tensor:
  """Reward COM proximity to wall inner face.

  Returns exp(-|dx|/0.1). Highest when base x equals wall_x.
  """
  asset: Entity = env.scene[asset_cfg.name]
  root_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  dist = torch.abs(root_x - wall_x)
  return torch.exp(-dist / 0.1)


class _SphereGeomCache:
  """Lazily caches per-sphere geom IDs and wall face position."""
  _instance: _SphereGeomCache | None = None
  geom_ids: torch.Tensor
  wall_face_x: float
  wall_sign: float
  wall_axis: int
  initialized: bool = False

  @classmethod
  def get(cls, env: ManagerBasedRlEnv) -> _SphereGeomCache:
    if cls._instance is None or not cls._instance.initialized:
      cls._instance = cls()
      mj = env.sim.mj_model
      ids = []
      for side in ("left", "right"):
        for i in range(2):
          for j in range(4):
            ids.append(mj.geom(f"robot/{side}_hand_c_{i}_{j}").id)
      cls._instance.geom_ids = torch.tensor(ids, dtype=torch.long, device=env.device)
      # Wall face: wall_normal default (-1,0,0) → axis=0, sign=-1
      cls._instance.wall_axis = 0
      cls._instance.wall_sign = -1.0
      wall = env.scene["wall"]
      wall_x = wall.data.root_link_pos_w[0, 0].item()
      cls._instance.wall_face_x = wall_x + cls._instance.wall_sign * 0.5
      cls._instance.initialized = True
    return cls._instance


def hands_on_wall(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  contact_threshold: float = 0.015,
) -> torch.Tensor:
  """Reward fraction of hand spheres in contact with wall.

  Checks all 16 spheres (8 per hand). Returns ratio within contact_threshold
  of the wall face (0=none, 1=all touching).
  """
  cache = _SphereGeomCache.get(env)
  ax = cache.wall_axis
  geom_x = env.sim.data.geom_xpos[:, cache.geom_ids, ax]  # (E, 16)
  dist = (cache.wall_face_x - geom_x) * cache.wall_sign
  in_contact = (dist.abs() < contact_threshold).float()
  return in_contact.mean(dim=1)


def base_lin_vel_z_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Penalize vertical velocity oscillation (promotes smooth climbing)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_w[:, 2])
