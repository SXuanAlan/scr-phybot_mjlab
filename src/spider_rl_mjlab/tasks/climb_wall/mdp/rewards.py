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
  """Lazily caches per-sphere geom IDs, radii, and wall geometry."""
  _instance: _SphereGeomCache | None = None
  geom_ids: torch.Tensor
  radii: torch.Tensor  # per-geom radius for surface distance
  wall_half: float
  wall_sign: float
  wall_axis: int
  initialized: bool = False

  @classmethod
  def get(cls, env: ManagerBasedRlEnv) -> _SphereGeomCache:
    if cls._instance is None or not cls._instance.initialized:
      cls._instance = cls()
      mj = env.sim.mj_model
      import mujoco
      ids = []
      radii = []
      for side in ("left", "right"):
        for i in range(2):
          for j in range(4):
            gid = mj.geom(f"robot/{side}_hand_c_{i}_{j}").id
            ids.append(gid)
            geom_type = int(mj.geom_type[gid])
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
              radii.append(float(mj.geom_size[gid, 0]))
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
              radii.append(float(mj.geom_size[gid, 0]))
            else:
              radii.append(float(mj.geom_size[gid, 0]))
      cls._instance.geom_ids = torch.tensor(ids, dtype=torch.long, device=env.device)
      cls._instance.radii = torch.tensor(radii, dtype=torch.float32, device=env.device)
      cls._instance.wall_axis = 0
      cls._instance.wall_sign = -1.0
      wall_geom_id = mj.geom("wall/wall_geom").id
      cls._instance.wall_half = float(mj.geom_size[wall_geom_id, 0])
      cls._instance.initialized = True
    return cls._instance

  @classmethod
  def wall_face(cls, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Per-env wall face position in world coords. Shape (E,)."""
    c = cls.get(env)
    wall_pos = env.scene["wall"].data.root_link_pos_w[:, c.wall_axis]
    return wall_pos + c.wall_sign * c.wall_half

  @classmethod
  def surface_distances(cls, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Per-sphere surface-to-wall distance. (E, 16).
    Positive=gap (sphere surface hasn't reached wall yet).
    Negative=penetration (sphere surface past wall face).
    Zero=touching.
    """
    c = cls.get(env)
    wall_f = cls.wall_face(env)  # (E,)
    geom_x = env.sim.data.geom_xpos[:, c.geom_ids, c.wall_axis]  # (E, 16)
    # Sphere surface closest to wall = geom_origin + radius toward wall.
    # Wall is in +x direction (wall_sign=-1 means wall normal points -x,
    # so wall is in +x from the robot). Surface point toward wall:
    #   surface_x = geom_x + radius (toward +x / wall)
    # Gap = wall_face - surface_x (positive when surface hasn't reached wall).
    surface_x = geom_x + c.radii.unsqueeze(0)  # surface closest to wall
    return wall_f.unsqueeze(1) - surface_x


def hands_on_wall(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  contact_threshold: float = 0.02,
) -> torch.Tensor:
  """Reward fraction of hand spheres in contact with wall.

  Uses surface distance. A sphere is in contact when its surface
  is within contact_threshold (20mm) of the wall.
  """
  surf_dist = _SphereGeomCache.surface_distances(env)  # (E, 16)
  in_contact = (surf_dist.abs() < contact_threshold).float()
  return in_contact.mean(dim=1)


def both_hands_bonus(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  contact_threshold: float = 0.02,
) -> torch.Tensor:
  """Bonus reward when BOTH hands have contact with wall.

  Computes per-hand contact ratio (0~1), then returns their product.
  Only gives reward when both hands are touching:
    left=0.8, right=0.0 → 0.0 (no bonus)
    left=0.8, right=0.5 → 0.4 (partial bonus)
    left=1.0, right=1.0 → 1.0 (full bonus)
  """
  surf_dist = _SphereGeomCache.surface_distances(env)  # (E, 16)
  in_contact = (surf_dist.abs() < contact_threshold).float()
  # First 8 = left hand, last 8 = right hand.
  left_ratio = in_contact[:, :8].mean(dim=1)
  right_ratio = in_contact[:, 8:].mean(dim=1)
  return left_ratio * right_ratio


def hands_toward_wall(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Continuous shaping reward: negative mean surface distance.

  Returns -mean(|surface_dist|) so that closer = less negative = higher reward.
  Clipped to [-0.5, 0] to avoid huge penalties when hands are very far.
  This provides linear gradient at all distances (no saturation).
  """
  surf_dist = _SphereGeomCache.surface_distances(env)  # (E, 16)
  return -surf_dist.abs().mean(dim=1).clamp(max=0.5)


def hand_orientation_toward_wall(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  hand_body_names: tuple[str, ...] = ("left_hand", "right_hand"),
) -> torch.Tensor:
  """Reward hand contact face pointing toward the wall.

  The contact spheres are on the hand's local -x face. This reward
  measures how well each hand's local -x axis aligns with the wall
  direction (+x world). Returns mean across hands, range [0, 1].

  dot(hand_local_neg_x, wall_dir) = 1 when perfectly aligned.
  """
  asset: Entity = env.scene[asset_cfg.name]
  # Wall direction: +x in world frame (toward wall).
  # hand local -x in world = negate the first column of the rotation matrix,
  # which equals -R[:, 0]. We want dot(-R[:,0], [1,0,0]) = -R[0,0].
  # Using quaternion: rotate local (-1,0,0) by hand quat.
  total = torch.zeros(env.num_envs, device=env.device)
  for bname in hand_body_names:
    ids, _ = asset.find_bodies([bname])
    body_id = ids[0]
    quat = asset.data.body_link_quat_w[:, body_id, :]  # (E, 4) wxyz

    # Rotate local (-1, 0, 0) by quat to get contact face normal in world.
    # For v=(-1,0,0): rotated_x = -(1 - 2(qy^2+qz^2)) = -1 + 2(qy^2+qz^2)
    # rotated_y = -2(qx*qy - qw*qz)
    # rotated_z = -2(qx*qz + qw*qy)
    # We want dot with (1,0,0) = just the x component of rotated vector.
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    face_x = -(1.0 - 2.0 * (y * y + z * z))  # x-component of rotated (-1,0,0)

    # face_x = 1 when local -x points toward +x (perfect), -1 when away.
    # Map to [0, 1]: (face_x + 1) / 2
    total += (face_x + 1.0) * 0.5

  return total / len(hand_body_names)


def base_lin_vel_z_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Penalize vertical velocity oscillation (promotes smooth climbing)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_w[:, 2])
