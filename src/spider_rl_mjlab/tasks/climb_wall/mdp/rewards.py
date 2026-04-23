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


def joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """L1 deviation of joint positions from their default (keyframe) values.

  Mirrors IsaacLab ``mdp.joint_deviation_l1``. Returns a positive sum of
  absolute errors over ``asset_cfg.joint_ids``; pair with a negative weight
  to penalise drift from the reference pose. G1 walking uses this on hip
  yaw/roll, arms, fingers, and torso to keep those joints near default.
  For climbing, it's applied to the joints that should stay parked
  (hip yaw/roll, waist) while arms — which must actively climb — are
  deliberately left out.
  """
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  current = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(current - default), dim=1)


def hand_orientation_toward_wall(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  hand_body_names: tuple[str, ...] = ("left_hand", "right_hand"),
) -> torch.Tensor:
  """Reward hand contact face pointing toward the wall (both phases).

  Each hand's face-normal (local -x in world) is scored for alignment with
  the wall's inward direction. No gait-phase gating — both stance AND
  swing hands are rewarded for staying oriented toward the wall. This
  prevents the hand from flipping around during swing.

  Returns a scalar in [0, 2] (each hand contributes up to 1).
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
  return total


def _hand_body_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> list[int]:
  asset: Entity = env.scene[asset_cfg.name]
  ids = []
  for name in ("left_hand", "right_hand"):
    idx, _ = asset.find_bodies([name])
    ids.append(idx[0])
  return ids


# ---------------------------------------------------------------------------
# Gait reward: min-grip-time, strict single-stance version. Direct port of
# IsaacLab's ``feet_air_time_positive_biped`` — the safe indicator is
# ``∑e == 1`` (exactly one hand engaged). Both-hand overlap and fully
# airborne states both give zero reward. Same (t_mode, min, tau) skeleton
# as the walking original.
# ---------------------------------------------------------------------------


def hand_grip_time_min_climb():
  """Factory: stateful reward tracking per-hand swing/grip time.

  Per-hand timers update every step based on the ``microspine_adhesion``
  latch flag:

      t_mode[h] = e[h] * t_grip[h] + (1 - e[h]) * t_swing[h]

  where ``t_grip`` resets on release and ``t_swing`` resets on grip. The
  reward is::

      r = 1[vz_cmd > eps] * min(tau, 1[∑e == 1] * min_h(t_mode[h]))

  ``1[∑e == 1]`` matches walking's strict single-stance gate: both-hand
  overlap AND both-hand airborne both give zero. Policy is only credited
  while exactly one hand is anchored — so it has no incentive to dwell
  in double-stance, and the previous reward-degenerate "hang motionless"
  solution is dead by construction (∑e == 2 gives 0).

  Closure-style (matches ``EMAObsFilter`` pattern) so each env cfg gets a
  fresh state instance and tyro doesn't try to introspect constructor args.
  The returned callable exposes a ``.reset(env_ids)`` hook that the reward
  manager drives on episode boundaries.
  """
  state: dict[str, torch.Tensor | None] = {"grip_t": None, "swing_t": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    event_term_name: str = "microspine_adhesion",
    tau: float = 0.5,
    vz_cmd: float = 0.08,
    cmd_eps: float = 0.0,
  ) -> torch.Tensor:
    term_cfg = env.event_manager.get_term_cfg(event_term_name)
    engaged = term_cfg.func.engaged  # (E, H) bool

    shape = engaged.shape
    dev = engaged.device
    if state["grip_t"] is None or tuple(state["grip_t"].shape) != tuple(shape):
      state["grip_t"] = torch.zeros(shape, dtype=torch.float32, device=dev)
      state["swing_t"] = torch.zeros(shape, dtype=torch.float32, device=dev)

    dt = env.step_dt
    e_f = engaged.float()
    # Tick engaged-hand grip timer; zero it on release.
    state["grip_t"] = e_f * (state["grip_t"] + dt)
    # Tick disengaged-hand swing timer; zero it on grip.
    state["swing_t"] = (1.0 - e_f) * (state["swing_t"] + dt)

    t_mode = e_f * state["grip_t"] + (1.0 - e_f) * state["swing_t"]  # (E, H)

    # Strict single-stance: exactly one hand engaged. Matches walking's
    # ``feet_air_time_positive_biped``. Zero reward when both hands on
    # (overlap) or both off (airborne).
    safe = (engaged.sum(dim=-1) == 1).float()  # (E,)
    move = 1.0 if vz_cmd > cmd_eps else 0.0

    min_t = t_mode.min(dim=-1).values  # (E,)
    return move * torch.clamp(safe * min_t, max=tau)

  def reset(env_ids=None) -> dict:
    if state["grip_t"] is None:
      return {}
    if env_ids is None:
      state["grip_t"].zero_()
      state["swing_t"].zero_()
    else:
      state["grip_t"][env_ids] = 0.0
      state["swing_t"][env_ids] = 0.0
    return {}

  call.reset = reset  # type: ignore[attr-defined]
  return call


def hand_slide_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  event_term_name: str = "microspine_adhesion",
) -> torch.Tensor:
  """Penalise tangential wall-plane velocity of ENGAGED hands.

  Replaces the previous clock-gated ``stance_phase_velocity``. Gate the
  slip speed by the actual ``engaged`` latch state rather than a desired-
  contact schedule, so the penalty tracks the real "grip should not
  slide" condition. Returns ``Σ_h e[h] · ‖v_h,tan‖`` (raw m/s).

  Pair with a negative weight. Normal-component motion (elastic wall
  compression, engage / release impact) is not penalised.
  """
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = _hand_body_ids(env, asset_cfg)
  _, normal_w = _SphereGeomCache.wall_plane(env)  # (E, 3)

  term_cfg = env.event_manager.get_term_cfg(event_term_name)
  engaged = term_cfg.func.engaged.float()  # (E, H)

  total = torch.zeros(env.num_envs, device=env.device)
  for h, body_id in enumerate(body_ids):
    v = asset.data.body_link_lin_vel_w[:, body_id, :]  # (E, 3)
    v_tan = v - (v * normal_w).sum(-1, keepdim=True) * normal_w
    total = total + engaged[:, h] * v_tan.norm(dim=-1)
  return total


def climb_vz_gauss(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  target: float = 0.08,
  sigma: float = 0.05,
) -> torch.Tensor:
  """Asymmetric Gaussian upward-velocity reward.

  $$ R = \\exp\\!\\left(-\\frac{(\\max(v_{\\text{target}} - v_z,\\ 0))^2}{\\sigma^2}\\right) $$

  Same exp-shape as ``track_lin_vel`` but asymmetric — no branching.
  $v_z \\ge v_{\\text{target}}$ gives $R = 1$ (overshoot is physically capped
  by arm reach). Only $v_z < v_{\\text{target}}$ decays via Gaussian.
  """
  asset: Entity = env.scene[asset_cfg.name]
  v = asset.data.root_link_lin_vel_w[:, 2]
  err = (target - v).clamp(min=0.0)
  return torch.exp(-err.pow(2) / (sigma * sigma))


def flat_orientation_gauss(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  sigma: float = 0.3,
  pitch_scale: float = 1.0,
  roll_scale: float = 1.0,
) -> torch.Tensor:
  """Gaussian-kernel base-upright tracking (target = world +z).

  ``R = exp(-(pitch_scale·gx² + roll_scale·gy²) / sigma²)``. When the base
  is upright, gx and gy are ~0 and reward is 1.

  Per-axis scales let you emphasise one failure mode — e.g.
  ``pitch_scale=0, roll_scale=1`` only penalises sideways tipping.
  """
  asset: Entity = env.scene[asset_cfg.name]
  proj_g = asset.data.projected_gravity_b  # (E, 3)
  gx = proj_g[:, 0]
  gy = proj_g[:, 1]
  err_sq = pitch_scale * gx.pow(2) + roll_scale * gy.pow(2)
  return torch.exp(-err_sq / (sigma * sigma))


def torso_orientation_gauss(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  body_name: str = "waist_roll",
  sigma: float = 0.3,
  pitch_scale: float = 1.0,
  roll_scale: float = 1.0,
) -> torch.Tensor:
  """Gaussian-kernel upright tracking for a named body (e.g. the torso).

  Same form as ``flat_orientation_gauss`` but reads the quaternion of the
  given body. Closes the waist-bend exploit where the pelvis stays level
  while the torso folds dramatically.
  """
  asset: Entity = env.scene[asset_cfg.name]
  ids, _ = asset.find_bodies([body_name])
  body_id = ids[0]
  quat = asset.data.body_link_quat_w[:, body_id, :]  # (E, 4) wxyz
  w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
  # R^T @ (0, 0, -1) — world-down in body frame. Only xy components matter.
  px = -2.0 * (x * z - w * y)  # grows with pitch (rotation around body-y)
  py = -2.0 * (w * x + y * z)  # grows with roll  (rotation around body-x)
  err_sq = pitch_scale * px.pow(2) + roll_scale * py.pow(2)
  return torch.exp(-err_sq / (sigma * sigma))
