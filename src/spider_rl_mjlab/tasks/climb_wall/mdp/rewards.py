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
  """Reward the robot's z-position (height). Encourages climbing higher."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2]


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


def wall_contact_reward(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, wall_x: float = 1.5, threshold: float = 0.1
) -> torch.Tensor:
  """Reward for being very close to the wall (proxy for hand contact).

  Without gecko adhesion controller, this uses proximity as a stand-in for
  actual contact detection. Returns 1.0 when within threshold, else 0.0.
  """
  asset: Entity = env.scene[asset_cfg.name]
  root_pos_x_env = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  distance_to_wall = torch.abs(root_pos_x_env - wall_x)
  return torch.where(
    distance_to_wall < threshold,
    torch.ones_like(distance_to_wall),
    torch.zeros_like(distance_to_wall),
  )


def microspine_adhesion_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward based on total microspine adhesion force capacity.

  Returns the mean adhesion activation across all 4 end-effectors,
  derived from the stochastic spine engagement model.
  """
  controller = env.event_manager.get_term_cfg("microspine_adhesion").func
  obs = controller.get_privileged_obs()  # (num_envs, 20)
  # Index 1 of each EE's 5-dim block is shear_capacity (activation).
  shear_caps = obs[:, 1::5]  # (num_envs, 4)
  return shear_caps.mean(dim=1)


def spine_engagement_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for having spines engaged on the wall.

  Returns the mean engaged ratio across all 4 end-effectors.
  Higher when more spines are successfully hooked onto asperities.
  """
  controller = env.event_manager.get_term_cfg("microspine_adhesion").func
  obs = controller.get_privileged_obs()  # (num_envs, 20)
  # Index 0 of each EE's 5-dim block is engaged_ratio.
  engaged_ratios = obs[:, 0::5]  # (num_envs, 4)
  return engaged_ratios.mean(dim=1)
