"""Custom reward functions for the wall-climbing task.

Ported from IsaacLab spider_rl_lab to mjlab API.
"""

from __future__ import annotations

import math
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

  Uses the signed perpendicular distance from the robot base to the wall plane,
  so it works correctly for tilted walls (curriculum-varied angle).
  ``wall_x`` kept for backwards-compat but no longer referenced.
  """
  del wall_x  # Unused: wall face is queried from mocap pose.
  asset: Entity = env.scene[asset_cfg.name]
  base_pos_w = asset.data.root_link_pos_w  # (E, 3)
  wall_pos_w, normal_w = _SphereGeomCache.wall_plane(env)
  wall_half = _SphereGeomCache.get(env).wall_half
  signed = ((base_pos_w - wall_pos_w) * normal_w).sum(dim=-1) - wall_half
  return torch.exp(-signed.abs() / 0.5)


def forward_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward velocity component toward the wall (along inward-normal).

  Works for tilted walls: uses wall's current outward normal instead of +x.
  """
  asset: Entity = env.scene[asset_cfg.name]
  v_w = asset.data.root_link_lin_vel_w  # (E, 3)
  _, normal_w = _SphereGeomCache.wall_plane(env)
  v_toward_wall = -(v_w * normal_w).sum(dim=-1)
  return v_toward_wall.clamp(min=0.0, max=1.0)


class _SphereGeomCache:
  """Lazily caches per-sphere geom IDs, radii, and wall geometry.

  Wall pose is queried per-step from the wall entity's mocap state so that
  curriculum-driven tilts are reflected in the derived surface/normal maths.
  """
  _instance: _SphereGeomCache | None = None
  geom_ids: torch.Tensor
  radii: torch.Tensor  # per-geom radius for surface distance
  wall_half: float
  initialized: bool = False

  @classmethod
  def get(cls, env: ManagerBasedRlEnv) -> _SphereGeomCache:
    if cls._instance is None or not cls._instance.initialized:
      cls._instance = cls()
      mj = env.sim.mj_model
      ids = []
      radii = []
      for side in ("left", "right"):
        for i in range(2):
          for j in range(4):
            gid = mj.geom(f"robot/{side}_hand_c_{i}_{j}").id
            ids.append(gid)
            radii.append(float(mj.geom_size[gid, 0]))
      cls._instance.geom_ids = torch.tensor(ids, dtype=torch.long, device=env.device)
      cls._instance.radii = torch.tensor(radii, dtype=torch.float32, device=env.device)
      wall_geom_id = mj.geom("wall/wall_geom").id
      cls._instance.wall_half = float(mj.geom_size[wall_geom_id, 0])
      cls._instance.initialized = True
    return cls._instance

  @classmethod
  def wall_plane(cls, env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env wall body position and inner-face outward normal (world frame).

    Outward normal points AWAY from the wall body, toward the robot, so a
    point ``P`` outside the wall has ``(P - wall_pos) · n > wall_half``.
    """
    wall = env.scene["wall"]
    wall_pos = wall.data.root_link_pos_w  # (E, 3)
    quat = wall.data.root_link_quat_w  # (E, 4) wxyz
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # Rotate local (-1, 0, 0) by quat to get outward normal in world.
    # For v=(-1,0,0): nx = -(1 - 2(y^2+z^2)); ny = -2(x*y + w*z); nz = -2(x*z - w*y)
    nx = -(1.0 - 2.0 * (y * y + z * z))
    ny = -2.0 * (x * y + w * z)
    nz = -2.0 * (x * z - w * y)
    normal = torch.stack([nx, ny, nz], dim=-1)
    return wall_pos, normal

  @classmethod
  def surface_distances(cls, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Per-sphere surface-to-wall signed distance. (E, 16).
    Positive=gap (sphere surface hasn't reached wall yet).
    Negative=penetration.
    Zero=touching.
    """
    c = cls.get(env)
    wall_pos, normal = cls.wall_plane(env)  # (E,3), (E,3)
    geom_pos = env.sim.data.geom_xpos[:, c.geom_ids, :]  # (E, 16, 3)
    delta = geom_pos - wall_pos.unsqueeze(1)  # (E, 16, 3)
    # Signed perpendicular distance from geom center to wall plane:
    signed = (delta * normal.unsqueeze(1)).sum(dim=-1) - c.wall_half  # (E, 16)
    # Subtract sphere radius to get gap at closest surface point.
    return signed - c.radii.unsqueeze(0)


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

  The contact spheres are on each hand's local -x face. We reward alignment
  of that face-normal with the (inward) wall direction. For a tilted wall,
  the target direction is ``-wall_outward_normal``. Returns mean over hands
  in [0, 1].
  """
  asset: Entity = env.scene[asset_cfg.name]
  _, wall_normal = _SphereGeomCache.wall_plane(env)  # (E, 3)
  target = -wall_normal  # direction from robot into wall

  total = torch.zeros(env.num_envs, device=env.device)
  for bname in hand_body_names:
    ids, _ = asset.find_bodies([bname])
    body_id = ids[0]
    quat = asset.data.body_link_quat_w[:, body_id, :]  # (E, 4) wxyz
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # Hand local -x in world:
    fx = -(1.0 - 2.0 * (y * y + z * z))
    fy = -2.0 * (x * y + w * z)
    fz = -2.0 * (x * z - w * y)
    face = torch.stack([fx, fy, fz], dim=-1)  # (E, 3)
    dot = (face * target).sum(dim=-1).clamp(-1.0, 1.0)
    total += (dot + 1.0) * 0.5
  return total / len(hand_body_names)


def base_lin_vel_z_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Penalize vertical velocity oscillation (promotes smooth climbing)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_w[:, 2])


# ---------------------------------------------------------------------------
# Gait rewards (adapted from Um et al. 2025 "RL-based Wall Climbing", Table I).
# Two-hand trot: left hand swings in φ ∈ [0, 0.5); right hand offset by 0.5.
# ---------------------------------------------------------------------------


def _phase_and_swing_masks(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """Return (swing_left, swing_right) ∈ {0,1} masks, shape (E,).

  A hand is "in swing" for the first half of its own phase cycle.
  """
  from spider_rl_mjlab.tasks.climb_wall.mdp.observations import gait_phase

  p = gait_phase(env)                       # (E,)
  p_left = p
  p_right = (p + 0.5) % 1.0
  swing_left = (p_left < 0.5).float()
  swing_right = (p_right < 0.5).float()
  return swing_left, swing_right


def _get_engaged(env: ManagerBasedRlEnv) -> torch.Tensor | None:
  """Fetch the live per-(env, hand) engaged bool tensor from MicrospineController.

  Returns None if the controller isn't attached (e.g., in tasks without it).
  """
  try:
    term = env.event_manager.get_term_cfg("microspine_adhesion")
  except Exception:
    return None
  ctrl = term.func
  engaged = getattr(ctrl, "engaged", None)
  if engaged is None:
    return None
  return engaged.float()  # (E, 2)


def _hand_body_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> list[int]:
  asset: Entity = env.scene[asset_cfg.name]
  ids = []
  for name in ("left_hand", "right_hand"):
    idx, _ = asset.find_bodies([name])
    ids.append(idx[0])
  return ids


def gait_match(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """R_g — SOFT sinusoidal swing/stance matching.

  Instead of a hard ±1 step, the desired stance signal is
      s_i(φ) = −sin(2π · φ_i)        ∈ [−1, +1]
  so that s_i ≈ +1 at the stance peak (φ=0.75) and s_i ≈ −1 at the swing
  peak (φ=0.25), and s_i ≈ 0 at the boundaries (φ=0, 0.5).

  The reward for each hand is  g_i = s_i · (2·c_i − 1)  where c_i ∈ {0, 1}
  is the engaged flag. Peak-phase mismatches cost the full ±1; near the
  swing↔stance boundaries the pressure relaxes to ≈0 so the policy is free
  to overshoot either way — i.e. **overlap emerges naturally** when the
  policy finds it beneficial (e.g. keep the next hand gripping early to
  avoid a fall). Returns 0.5·(g_L + g_R) ∈ [−1, 1].
  """
  engaged = _get_engaged(env)
  if engaged is None:
    return torch.zeros(env.num_envs, device=env.device)

  from spider_rl_mjlab.tasks.climb_wall.mdp.observations import gait_phase

  p = gait_phase(env)                      # (E,)
  two_pi = 2.0 * math.pi
  s_l = -torch.sin(two_pi * p)             # left  desired-stance signal
  s_r = -torch.sin(two_pi * ((p + 0.5) % 1.0))  # right (phase offset 0.5)

  c_l = engaged[:, 0]
  c_r = engaged[:, 1]

  g_l = s_l * (2.0 * c_l - 1.0)
  g_r = s_r * (2.0 * c_r - 1.0)
  return 0.5 * (g_l + g_r)


def swing_hand_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  lift_target: float = 0.5,
) -> torch.Tensor:
  """R_fh — Gaussian shaping to keep the swing hand ~lift_target (m) above base.

  Only the swing hand contributes; stance side masked out. Returns a value
  in (0, 0.5]. Continuous gradient everywhere — this is the main "teach
  the hand to lift" signal.
  """
  asset: Entity = env.scene[asset_cfg.name]
  base_z = asset.data.root_link_pos_w[:, 2]
  body_ids = _hand_body_ids(env, asset_cfg)
  z_l = asset.data.body_link_pos_w[:, body_ids[0], 2]
  z_r = asset.data.body_link_pos_w[:, body_ids[1], 2]

  sw_l, sw_r = _phase_and_swing_masks(env)
  dz_l = (z_l - base_z) - lift_target
  dz_r = (z_r - base_z) - lift_target
  err = sw_l * dz_l.pow(2) + sw_r * dz_r.pow(2)
  return 0.5 * torch.exp(-err)


def swing_hand_clearance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  lift_target: float = 0.5,
) -> torch.Tensor:
  """R_fc — strong penalty for a swing hand that is NOT lifted and is still
  moving (would correspond to "shuffling on the wall" rather than truly
  swinging up).

  Active only when the hand is not engaged AND below lift_target. Multiplied
  by sqrt of |v_z| so it fires harder when the hand is moving without lift.
  Returns a positive scalar (apply with negative weight in env cfg).
  """
  asset: Entity = env.scene[asset_cfg.name]
  engaged = _get_engaged(env)
  if engaged is None:
    return torch.zeros(env.num_envs, device=env.device)
  c_l, c_r = engaged[:, 0], engaged[:, 1]

  body_ids = _hand_body_ids(env, asset_cfg)
  base_z = asset.data.root_link_pos_w[:, 2]
  z_l = asset.data.body_link_pos_w[:, body_ids[0], 2] - base_z
  z_r = asset.data.body_link_pos_w[:, body_ids[1], 2] - base_z
  vz_l = asset.data.body_link_lin_vel_w[:, body_ids[0], 2]
  vz_r = asset.data.body_link_lin_vel_w[:, body_ids[1], 2]

  # deficit: how far below the lift target (0 if already at/above)
  deficit_l = (lift_target - z_l).clamp(min=0.0)
  deficit_r = (lift_target - z_r).clamp(min=0.0)

  pen_l = (1.0 - c_l) * deficit_l.pow(2) * vz_l.abs().clamp(min=1e-3).sqrt()
  pen_r = (1.0 - c_r) * deficit_r.pow(2) * vz_r.abs().clamp(min=1e-3).sqrt()
  return pen_l + pen_r


def stance_hand_anchor(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """R_fs — penalty for stance-side hand sliding (non-zero speed while in
  contact). Encourages solid grip. Apply with negative weight."""
  asset: Entity = env.scene[asset_cfg.name]
  engaged = _get_engaged(env)
  if engaged is None:
    return torch.zeros(env.num_envs, device=env.device)
  c_l, c_r = engaged[:, 0], engaged[:, 1]

  body_ids = _hand_body_ids(env, asset_cfg)
  v_l = asset.data.body_link_lin_vel_w[:, body_ids[0], :]
  v_r = asset.data.body_link_lin_vel_w[:, body_ids[1], :]
  return c_l * v_l.pow(2).sum(dim=-1) + c_r * v_r.pow(2).sum(dim=-1)


def climb_vz(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Reward upward base velocity (the actual climbing objective).

  Clipped to [0, 0.5 m/s] so the policy doesn't game it with bursts.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_w[:, 2].clamp(min=0.0, max=0.5)
