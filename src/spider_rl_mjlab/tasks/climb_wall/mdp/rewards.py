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
  """Reward hand contact face pointing toward the wall, gated by stance.

  Each hand's face-normal (local -x in world) is scored for alignment with
  the wall's inward direction. The per-hand score is then multiplied by the
  gait scheduler's desired-contact signal ``C_h^cmd`` — so a hand is only
  penalised for misalignment while it *should* be anchored to the wall.
  During its swing phase the hand is free to rotate away to reach upward.

  Returns a scalar in [0, 2] (each hand contributes up to 1 during stance).
  """
  asset: Entity = env.scene[asset_cfg.name]
  _, wall_normal = _SphereGeomCache.wall_plane(env)  # (E, 3)
  target = -wall_normal  # direction from robot into wall

  c = _desired_contact(env)  # (E, 2) stance weight per hand

  total = torch.zeros(env.num_envs, device=env.device)
  for h, bname in enumerate(hand_body_names):
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
    total += c[:, h] * (dot + 1.0) * 0.5
  return total


# ---------------------------------------------------------------------------
# Gait rewards (adapted from Margolis & Agrawal 2022 "Walk These Ways",
# Table 1, with a soft von-Mises-style contact schedule).
# Two-hand trot: left hand stance plateau p ∈ [0.5, 1); right hand offset 0.5.
# ---------------------------------------------------------------------------


_SCHEDULE_SIGMA = 0.03  # phase-units half-width of the stance↔swing transition


def _stance_bump(x: torch.Tensor, sigma: float) -> torch.Tensor:
  """Smooth periodic square wave with stance plateau on [0.5, 1).

  Uses two shifted Gaussian-CDF plateaus (primary at [0.5, 1], wrap tail at
  the end of the previous cycle) so the schedule is continuous across p=0
  and p=1. σ controls the transition half-width in phase units.
  """
  sqrt2 = math.sqrt(2.0)
  a = 0.5 * (
    torch.erf((x - 0.5) / (sigma * sqrt2))
    - torch.erf((x - 1.0) / (sigma * sqrt2))
  )
  b = 0.5 * (
    torch.erf((x + 0.5) / (sigma * sqrt2))
    - torch.erf(x / (sigma * sqrt2))
  )
  return (a + b).clamp(0.0, 1.0)


def _desired_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Smooth desired contact schedule C_h^cmd(p) ∈ [0, 1], shape (E, 2).

  Left hand stance plateau p ∈ [0.5, 1); right hand offset by 0.5.
  Transitions are Gaussian-CDF smoothed over ±~`_SCHEDULE_SIGMA` so the
  policy gets a continuous gradient across swing↔stance boundaries (matches
  Walk-These-Ways). σ=0.03 ≈ 6% of a cycle on each side.
  """
  from spider_rl_mjlab.tasks.climb_wall.mdp.observations import gait_phase

  p = gait_phase(env)  # (E,)
  c_l = _stance_bump(p, _SCHEDULE_SIGMA)
  c_r = _stance_bump((p + 0.5) % 1.0, _SCHEDULE_SIGMA)
  return torch.stack([c_l, c_r], dim=1)  # (E, 2)


def stance_phase_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, sigma: float = 0.3
) -> torch.Tensor:
  """R_cst — "Walk These Ways" stance phase tracking (velocity).

  During stance (C^cmd ≈ 1), the hand should not SLIDE along the wall:

      r = Σ_h C_h^cmd · exp(−‖v_h,tan‖² / σ²)

  ``v_h,tan`` is the component of the hand's world velocity in the wall
  plane (normal component projected out). Normal motion into/out of the
  wall is not penalised — small elastic compression against the wall is
  expected even during a solid grip. Returns a scalar in [0, 2].
  """
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = _hand_body_ids(env, asset_cfg)
  _, normal_w = _SphereGeomCache.wall_plane(env)  # (E, 3)

  v_l = asset.data.body_link_lin_vel_w[:, body_ids[0], :]  # (E, 3)
  v_r = asset.data.body_link_lin_vel_w[:, body_ids[1], :]
  v_l_tan = v_l - (v_l * normal_w).sum(-1, keepdim=True) * normal_w
  v_r_tan = v_r - (v_r * normal_w).sum(-1, keepdim=True) * normal_w
  speed_sq = torch.stack(
    [v_l_tan.pow(2).sum(-1), v_r_tan.pow(2).sum(-1)], dim=1
  )
  C = _desired_contact(env)
  kernel = torch.exp(-speed_sq / (sigma * sigma))
  return (C * kernel).sum(dim=1)


def _hand_body_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> list[int]:
  asset: Entity = env.scene[asset_cfg.name]
  ids = []
  for name in ("left_hand", "right_hand"):
    idx, _ = asset.find_bodies([name])
    ids.append(idx[0])
  return ids


def swing_hand_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  lift_amplitude: float = 0.2,
  sigma: float = 0.08,
  r_floor: float = 0.3,
) -> torch.Tensor:
  """Track each hand's distance-to-wall against a sinusoidal reference.

  Ported from "Custom Sine Waves Are Enough for Imitation Learning of
  Bipedal Gaits" (Wu et al. 2022, arXiv:2204.04157). The reference
  trajectory for each hand's perpendicular distance to the wall surface:

      d_ref_L(p) = amp · max(0, sin(2π · p))
      d_ref_R(p) = amp · max(0, sin(2π · (p + 0.5)))

  - Stance (sine ≤ 0 → clipped to 0): hand should be ON the wall (d ≈ 0).
  - Swing peak (p = 0.25 for left, 0.75 for right): d_ref = amp.
  - The sine arc captures the whole lift-peak-lower motion, not just the apex.

  This works as a stance reference because the wall has a fixed world
  position — "hand on wall surface" is a single geometric target that does
  NOT drift as the robot climbs (analogous to Cassie's foot-on-ground
  reference during stance).

  Reward: sharpened Gaussian with reward-floor normalisation.

      raw = exp(−Σ_h (d_h − d_ref_h)² / σ²)
      r   = max(0, (raw − r_floor) / (1 − r_floor))
  """
  surf_dist = _SphereGeomCache.surface_distances(env)        # (E, 16)
  d_l = surf_dist[:, :8].mean(dim=1)                         # (E,)
  d_r = surf_dist[:, 8:].mean(dim=1)

  from spider_rl_mjlab.tasks.climb_wall.mdp.observations import gait_phase

  p = gait_phase(env)                                        # (E,)
  two_pi = 2.0 * math.pi
  d_ref_l = lift_amplitude * torch.sin(two_pi * p).clamp(min=0.0)
  d_ref_r = lift_amplitude * torch.sin(two_pi * ((p + 0.5) % 1.0)).clamp(min=0.0)

  err = (d_l - d_ref_l).pow(2) + (d_r - d_ref_r).pow(2)
  raw = torch.exp(-err / (sigma * sigma))
  return ((raw - r_floor).clamp(min=0.0) / (1.0 - r_floor))


def swing_hand_reach_up(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  reach_height: float = 0.35,
  sigma: float = 0.18,
  r_floor: float = 0.3,
) -> torch.Tensor:
  """Reward each hand reaching UP (world +z) above the torso during swing.

  Tracks a sinusoidal reference for ``(hand_z - root_z)``, peaking at
  ``reach_height`` during each hand's swing phase and falling to 0 during
  stance. Combined with ``swing_hand_height`` (which rewards pulling AWAY
  from the wall), this drives the full "peel off → reach UP → re-plant higher"
  arc needed for climbing progress.

  Uses the same clamped-sine schedule as ``swing_hand_height`` so both
  rewards peak simultaneously per hand.
  """
  from spider_rl_mjlab.tasks.climb_wall.mdp.observations import gait_phase

  asset: Entity = env.scene[asset_cfg.name]
  body_ids = _hand_body_ids(env, asset_cfg)
  root_z = asset.data.root_link_pos_w[:, 2]                               # (E,)
  z_l = asset.data.body_link_pos_w[:, body_ids[0], 2] - root_z            # (E,)
  z_r = asset.data.body_link_pos_w[:, body_ids[1], 2] - root_z            # (E,)

  p = gait_phase(env)                                                    # (E,)
  two_pi = 2.0 * math.pi
  z_ref_l = reach_height * torch.sin(two_pi * p).clamp(min=0.0)
  z_ref_r = reach_height * torch.sin(two_pi * ((p + 0.5) % 1.0)).clamp(min=0.0)

  err = (z_l - z_ref_l).pow(2) + (z_r - z_ref_r).pow(2)
  raw = torch.exp(-err / (sigma * sigma))
  return ((raw - r_floor).clamp(min=0.0) / (1.0 - r_floor))


def airborne_penalty(
  env: ManagerBasedRlEnv,
  event_term_name: str = "microspine_adhesion",
) -> torch.Tensor:
  """Penalise steps where NEITHER hand is engaged — the "jump" state.

  Reads the microspine controller's per-hand latch flags. Returns 1.0 on
  every env where both hands are disengaged, else 0.0. Pair with a negative
  weight in the reward config so the policy pays for flying.
  """
  term_cfg = env.event_manager.get_term_cfg(event_term_name)
  engaged = term_cfg.func.engaged  # (E, H) bool
  both_off = (~engaged).all(dim=-1)  # (E,)
  return both_off.float()


def climb_vz(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Reward upward base velocity (climbing-progress objective).

  The wall is always vertical in world frame, so "up the wall" is world
  +z. Clipped to [0, 0.5 m/s] — above this cap the extra speed gives no
  extra reward, which removes the incentive to launch off the wall in
  ballistic bursts. Smooth, sustained climbing at ~0.5 m/s strictly
  dominates 2 m/s burst + free-fall.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_w[:, 2].clamp(min=0.0, max=0.5)


def climb_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  baseline_z: float = 2.0,
) -> torch.Tensor:
  """Reward current height above spawn (world +z, meters).

  Position-based — rewards BEING high, not moving up. Integrated over an
  episode, this naturally prefers "climb fast early and hold high" to any
  bursty/oscillating pattern: time at altitude accrues reward, whereas a
  climb-then-fall trajectory accrues less because the robot spends time
  near the bottom.

  Returns ``max(0, root_z - baseline_z)`` so the baseline (no climb) gives
  zero reward rather than a flat bias from the spawn altitude.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return (asset.data.root_link_pos_w[:, 2] - baseline_z).clamp(min=0.0)
