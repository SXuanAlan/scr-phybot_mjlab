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


def hands_on_wall(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  adhesion_actuator_names: tuple[str, ...] = (
    "robot/adhesion_left_hand",
    "robot/adhesion_right_hand",
  ),
) -> torch.Tensor:
  """Reward hands maintaining contact with wall via adhesion actuator ctrl.

  Returns the mean adhesion ctrl value across hands (0=disengaged, 1=fully engaged).
  The ctrl is set by the MicrospineController based on actual contact state.
  """
  total = torch.zeros(env.num_envs, device=env.device)
  for act_name in adhesion_actuator_names:
    act_id = env.sim.mj_model.actuator(act_name).id
    total += env.sim.data.ctrl[:, act_id]
  return total / len(adhesion_actuator_names)


def base_lin_vel_z_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Penalize vertical velocity oscillation (promotes smooth climbing)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_w[:, 2])
