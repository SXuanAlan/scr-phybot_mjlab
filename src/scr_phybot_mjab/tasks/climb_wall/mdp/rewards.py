"""Custom reward functions for the wall-climbing task.

Ported from IsaacLab spider_rl_lab to mjlab API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import ClimbGaitPhase

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
      for ee_name in (
        "left_hand_spine",
        "right_hand_spine",
        "left_foot_spine",
        "right_foot_spine",
      ):
        for i in range(2):
          for j in range(2):
            gid = mj.geom(f"robot/{ee_name}_c_{i}_{j}").id
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


def _hand_body_ids(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  body_names: tuple[str, ...] = ("left_hand", "right_hand"),
) -> list[int]:
  asset: Entity = env.scene[asset_cfg.name]
  ids = []
  for name in body_names:
    idx, _ = asset.find_bodies([name])
    ids.append(idx[0])
  return ids


# ---------------------------------------------------------------------------
# Gait reward: min-grip-time, dense stance-count version. Direct port of
# IsaacLab's ``feet_air_time_positive_biped`` with the stance-count gate made
# configurable. The biped default is ``∑e == 1``; C2's four spine EEs use
# ``∑e == 2``.
# ---------------------------------------------------------------------------


def hand_grip_time_min_climb():
  """Factory: stateful reward tracking per-hand swing/grip time.

  Per-hand timers update every step based on the ``microspine_grip``
  latch flag:

      t_mode[h] = e[h] * t_grip[h] + (1 - e[h]) * t_swing[h]

  where ``t_grip`` resets on release and ``t_swing`` resets on grip. The
  reward is::

      r = 1[vz_cmd > eps] * min(tau, 1[∑e == target] * min_h(t_mode[h]))

  ``target_engaged_count=1`` matches walking's strict single-stance gate.
  For four-EE climbing, set ``target_engaged_count=2`` to reward a stable
  two-anchor / two-swing gait instead.

  Closure-style (matches ``EMAObsFilter`` pattern) so each env cfg gets a
  fresh state instance and tyro doesn't try to introspect constructor args.
  The returned callable exposes a ``.reset(env_ids)`` hook that the reward
  manager drives on episode boundaries.
  """
  state: dict[str, torch.Tensor | None] = {"grip_t": None, "swing_t": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    event_term_name: str = "microspine_grip",
    tau: float = 0.5,
    vz_cmd: float = 0.08,
    cmd_eps: float = 0.0,
    target_engaged_count: int = 1,
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

    safe = (engaged.sum(dim=-1) == target_engaged_count).float()  # (E,)
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


def ee_capped_disengage_time(
  env: ManagerBasedRlEnv,
  *,
  event_term_name: str = "microspine_grip",
  command_name: str | None = None,
  vz_cmd: float = 0.08,
  command_threshold: float = 0.01,
  cap_time: float = 0.3,
  max_disengage_time: float = 0.5,
  target_engaged_count: int = 2,
) -> torch.Tensor:
  """Dense swing shaping for climbing: reward bounded disengage time.

  For each EE, ``current_disengage_time`` grows only while the EE is not
  latched and resets to 0 on engagement. The reward is enabled only when the
  commanded climb velocity is active, exactly ``target_engaged_count`` EEs are
  engaged, and no EE has been disengaged longer than ``max_disengage_time``.
  """
  term_cfg = env.event_manager.get_term_cfg(event_term_name)
  physics = term_cfg.func
  engaged: torch.Tensor = physics.engaged
  if not hasattr(physics, "current_disengage_time"):
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

  disengage_t: torch.Tensor = physics.current_disengage_time
  disengaged_t = disengage_t * (~engaged).float()

  count_gate = (engaged.sum(dim=-1) == target_engaged_count).float()
  max_gate = (disengage_t.max(dim=-1).values < max_disengage_time).float()

  if command_name is not None:
    cmd_z = env.command_manager.get_command(command_name)[:, 2]
    cmd_gate = (cmd_z.abs() > command_threshold).float()
  else:
    cmd_gate = 1.0 if abs(vz_cmd) > command_threshold else 0.0

  raw = disengaged_t.clamp(max=cap_time).sum(dim=-1)
  return cmd_gate * count_gate * max_gate * raw


def hand_slide_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  event_term_name: str = "microspine_grip",
  hand_body_names: tuple[str, ...] = ("left_hand", "right_hand"),
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
  body_ids = _hand_body_ids(env, asset_cfg, hand_body_names)
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

  Same exp-shape as the climb velocity tracking terms but asymmetric.
  $v_z \\ge v_{\\text{target}}$ gives $R = 1$ (overshoot is physically capped
  by arm reach). Only $v_z < v_{\\text{target}}$ decays via Gaussian.
  """
  asset: Entity = env.scene[asset_cfg.name]
  v = asset.data.root_link_lin_vel_w[:, 2]
  err = (target - v).clamp(min=0.0)
  return torch.exp(-err.pow(2) / (sigma * sigma))


def climb_linear_velocity_gauss(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  target: tuple[float, float, float] = (0.0, 0.0, 0.08),
  command_name: str | None = None,
  sigma: float = 0.05,
) -> torch.Tensor:
  """Gaussian reward for tracking the commanded world-frame climb velocity.

  Unlike ``climb_vz_gauss``, this tracks the full vector. If ``command_name``
  is provided, the target comes from the command manager so command changes
  are reflected in both observation and reward.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if command_name is not None:
    target_w = env.command_manager.get_command(command_name)
  else:
    target_w = torch.tensor(
      target, dtype=torch.float32, device=env.device
    ).unsqueeze(0)
  err = asset.data.root_link_lin_vel_w - target_w
  return torch.exp(-torch.sum(err.pow(2), dim=-1) / (sigma * sigma))


def climb_linear_velocity_xy_gauss(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  target: tuple[float, float] = (0.0, 0.0),
  command_name: str | None = None,
  sigma: float = 0.05,
) -> torch.Tensor:
  """Gaussian reward for tracking commanded world-frame x/y velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  if command_name is not None:
    target_xy = env.command_manager.get_command(command_name)[:, :2]
  else:
    target_xy = torch.tensor(
      target, dtype=torch.float32, device=env.device
    ).unsqueeze(0)
  err = asset.data.root_link_lin_vel_w[:, :2] - target_xy
  return torch.exp(-torch.sum(err.pow(2), dim=-1) / (sigma * sigma))


def climb_linear_velocity_z_gauss(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  target: float = 0.08,
  command_name: str | None = None,
  sigma: float = 0.05,
) -> torch.Tensor:
  """Gaussian reward for tracking commanded world-frame z velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  if command_name is not None:
    target_z = env.command_manager.get_command(command_name)[:, 2]
  else:
    target_z = torch.tensor(target, dtype=torch.float32, device=env.device)
  err = asset.data.root_link_lin_vel_w[:, 2] - target_z
  return torch.exp(-err.pow(2) / (sigma * sigma))


def climb_height_progress():
  """Factory: reward new height after the reset pose has physically settled.

  The term tracks the best base height reached in each episode and rewards
  only increases of that high-water mark. A policy that thrusts the hips up
  and falls back down can receive this reward once for the first peak, but not
  repeatedly for the same oscillation. During ``settle_time_s`` the high-water
  mark follows the base without reward, preventing the unavoidable reset sag
  from creating a several-centimeter reward dead zone.
  """
  state: dict[str, torch.Tensor | None] = {"best_z": None, "fresh": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    target_vz: float = 0.08,
    command_threshold: float = 0.01,
    max_normalized_progress: float = 1.5,
    settle_time_s: float = 0.5,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]

    if state["best_z"] is None or state["best_z"].shape != z.shape:
      state["best_z"] = z.clone()
      state["fresh"] = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
      return torch.zeros_like(z)

    fresh = state["fresh"]
    prev_best = state["best_z"]
    assert fresh is not None and prev_best is not None
    settling = env.episode_length_buf.float() * env.step_dt <= settle_time_s
    initialize = fresh | settling
    prev_best = torch.where(initialize, z, prev_best)
    new_best = torch.maximum(prev_best, z)
    progress_m = new_best - prev_best
    state["best_z"] = torch.where(initialize, z, new_best)
    fresh.zero_()

    if command_name is not None:
      cmd_z = env.command_manager.get_command(command_name)[:, 2].abs()
    else:
      cmd_z = torch.full_like(z, abs(target_vz))

    active = cmd_z > command_threshold
    target_step = (cmd_z * env.step_dt).clamp_min(1.0e-6)
    progress = (progress_m / target_step).clamp(max=max_normalized_progress)
    progress = torch.where(initialize, torch.zeros_like(progress), progress)
    return torch.where(active, progress, torch.zeros_like(progress))

  def reset(env_ids=None) -> dict:
    if state["fresh"] is None:
      return {}
    if env_ids is None:
      state["fresh"].fill_(True)
    else:
      state["fresh"][env_ids] = True
    return {}

  call.reset = reset  # type: ignore[attr-defined]
  return call


def signed_climb_height_progress():
  """Factory: dense signed base-height change after initial settling.

  Unlike a positive-only velocity reward, a closed up/down cycle has roughly
  zero undiscounted return. Its weight should stay below the high-water term:
  it supplies a recovery gradient below the previous maximum, while genuinely
  new height remains the primary objective.
  """
  state: dict[str, torch.Tensor | None] = {"prev_z": None, "fresh": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    target_vz: float = 0.08,
    command_threshold: float = 0.01,
    max_normalized_progress: float = 1.0,
    settle_time_s: float = 0.5,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]

    if state["prev_z"] is None or state["prev_z"].shape != z.shape:
      state["prev_z"] = z.clone()
      state["fresh"] = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
      return torch.zeros_like(z)

    prev_z = state["prev_z"]
    fresh = state["fresh"]
    assert prev_z is not None and fresh is not None
    dz = z - prev_z
    state["prev_z"] = z.clone()

    if command_name is not None:
      cmd_z = env.command_manager.get_command(command_name)[:, 2]
    else:
      cmd_z = torch.full_like(z, target_vz)
    active = cmd_z.abs() > command_threshold
    target_step = (cmd_z.abs() * env.step_dt).clamp_min(1.0e-6)
    progress = (dz / target_step).clamp(
      min=-max_normalized_progress,
      max=max_normalized_progress,
    )

    settling = env.episode_length_buf.float() * env.step_dt <= settle_time_s
    initialize = fresh | settling
    fresh.zero_()
    return torch.where(active & ~initialize, progress, torch.zeros_like(progress))

  def reset(env_ids=None) -> dict:
    if state["fresh"] is None:
      return {}
    ids = slice(None) if env_ids is None else env_ids
    state["fresh"][ids] = True
    return {}

  call.reset = reset  # type: ignore[attr-defined]
  return call


def climb_velocity_z_error_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str | None = None,
  target_vz: float = 0.08,
  velocity_scale: float = 0.1,
  max_error: float = 4.0,
) -> torch.Tensor:
  """Normalized vertical command error with zero reward at perfect tracking."""
  asset: Entity = env.scene[asset_cfg.name]
  if command_name is not None:
    target = env.command_manager.get_command(command_name)[:, 2]
  else:
    target = torch.full_like(asset.data.root_link_lin_vel_w[:, 2], target_vz)
  error = (asset.data.root_link_lin_vel_w[:, 2] - target) / velocity_scale
  return error.square().clamp(max=max_error)


def climb_lateral_velocity_y_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  velocity_scale: float = 0.1,
  max_error: float = 4.0,
) -> torch.Tensor:
  """Penalize wall-tangent lateral drift without suppressing wall-normal motion."""
  asset: Entity = env.scene[asset_cfg.name]
  error = asset.data.root_link_lin_vel_w[:, 1] / velocity_scale
  return error.square().clamp(max=max_error)


def wall_distance_band_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  target_distance: float = 0.255,
  tolerance: float = 0.08,
  distance_scale: float = 0.1,
  max_error: float = 4.0,
  wall_half_thickness: float = 0.5,
) -> torch.Tensor:
  """Penalize only base-to-wall distances outside a permissive dead band."""
  asset: Entity = env.scene[asset_cfg.name]
  wall_pos, outward_normal = _SphereGeomCache.wall_plane(env)
  base_offset = asset.data.root_link_pos_w - wall_pos
  distance = (
    (base_offset * outward_normal).sum(dim=-1) - wall_half_thickness
  )
  error = (distance - target_distance).abs().sub(tolerance).clamp_min(0.0)
  return (error / distance_scale).square().clamp(max=max_error)


def bounded_joint_acc_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  acceleration_scale: float = 100.0,
  max_cost: float = 10.0,
) -> torch.Tensor:
  """Mean squared joint acceleration with a finite influence bound.

  Constraint release and failed episodes can produce brief acceleration
  impulses several orders of magnitude above normal motion. An unbounded
  squared cost lets one such sample dominate an entire PPO batch. Normalize
  before averaging across joints and cap only that regularizer's influence;
  the termination and grip-safety terms still represent the failure itself.
  """
  if acceleration_scale <= 0.0:
    raise ValueError("acceleration_scale must be positive")
  if max_cost <= 0.0:
    raise ValueError("max_cost must be positive")

  asset: Entity = env.scene[asset_cfg.name]
  joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
  cost = torch.mean(torch.square(joint_acc / acceleration_scale), dim=1)
  cost = torch.nan_to_num(cost, nan=max_cost, posinf=max_cost, neginf=max_cost)
  return torch.clamp(cost, max=max_cost)


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


# ---------------------------------------------------------------------------
# Anymal-style quadruped rewards adapted for wall climbing (engage signal
# replaces contact signal; wall normal replaces world z).
# ---------------------------------------------------------------------------


def lin_vel_into_wall_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize base linear velocity along the wall normal.

  Kept for climb_test and legacy configs. Main climb-wall C2 now uses full
  vector velocity tracking instead of this extra wall-normal velocity term.
  """
  asset: Entity = env.scene[asset_cfg.name]
  _, wall_normal = _SphereGeomCache.wall_plane(env)  # (E, 3)
  v_w = asset.data.root_link_lin_vel_w  # (E, 3)
  v_along_n = (v_w * wall_normal).sum(dim=-1)  # (E,)
  return v_along_n.pow(2)


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize base angular velocity around body x and y axes.

  Direct port of Anymal's ``ang_vel_xy_l2``. Discourages the policy from
  flailing roll/pitch — yaw rate (z) is intentionally not penalised here.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=-1)


def ee_air_time():
  """Factory for engage-signal version of Anymal's ``feet_air_time``.

  Per-EE bookkeeping:
    * ``current_disengage_t`` — seconds the EE has been continuously disengaged.
    * On the step an EE transitions disengaged→engaged (``first_engage``):
      reward += ``(current_disengage_t - threshold)``.

  Closure-style (matches ``hand_grip_time_min_climb``) so each env cfg gets
  fresh state and tyro doesn't introspect a class. Knobs are passed via
  ``RewardTermCfg.params``; the returned callable exposes ``.reset(env_ids)``
  for the reward manager to drive on episode boundaries.

  Reward is gated to zero when the climbing command is below ``cmd_eps``.
  Since this task uses a fixed-target ``vz_cmd``, the gate is a constant
  evaluated per call.
  """
  state: dict[str, torch.Tensor | None] = {
    "prev_engaged": None,
    "current_disengage_t": None,
  }

  def call(
    env: ManagerBasedRlEnv,
    *,
    threshold: float = 0.3,
    vz_cmd: float = 0.08,
    cmd_eps: float = 0.0,
    event_term_name: str = "microspine_grip",
  ) -> torch.Tensor:
    term_cfg = env.event_manager.get_term_cfg(event_term_name)
    engaged: torch.Tensor = term_cfg.func.engaged  # (E, H) bool

    if state["prev_engaged"] is None or state["prev_engaged"].shape != engaged.shape:
      # Init to the current engaged state (not zeros) so step 1 doesn't
      # spuriously fire first-engage events for every EE that started
      # the episode latched (microspine reset seeds engaged=True).
      state["prev_engaged"] = engaged.clone()
      state["current_disengage_t"] = torch.zeros(
        engaged.shape, dtype=torch.float32, device=engaged.device
      )

    prev = state["prev_engaged"]
    first_engage = (~prev) & engaged  # (E, H) — newly engaged this step

    # Snapshot the disengage spell that just ended.
    last_disengage_t = torch.where(
      first_engage,
      state["current_disengage_t"],
      torch.zeros_like(state["current_disengage_t"]),
    )

    # Anymal: reward = sum_h (last_air_time - threshold) * first_contact[h]
    reward = torch.sum(
      (last_disengage_t - threshold) * first_engage.float(),
      dim=1,
    )

    # Gate by command activity (fixed-target task — constant per call).
    if abs(vz_cmd) <= cmd_eps:
      reward = torch.zeros_like(reward)

    # Tick disengage timer for the NEXT step: zero on engage, +dt on disengage.
    dt = env.step_dt
    state["current_disengage_t"] = torch.where(
      engaged,
      torch.zeros_like(state["current_disengage_t"]),
      state["current_disengage_t"] + dt,
    )
    state["prev_engaged"] = engaged.clone()

    return reward

  def reset(env_ids=None) -> dict:
    if state["prev_engaged"] is None:
      return {}
    # Reset prev_engaged to True (matches microspine reset which seeds
    # engaged=True). This way the first post-reset step doesn't fire a
    # spurious first-engage event.
    if env_ids is None:
      state["prev_engaged"].fill_(True)
      state["current_disengage_t"].zero_()
    else:
      state["prev_engaged"][env_ids] = True
      state["current_disengage_t"][env_ids] = 0.0
    return {}

  call.reset = reset  # type: ignore[attr-defined]
  return call


# Alias so the climb-wall env_cfg can use a name that matches the
# Anymal-style "ee_*" convention. ``hand_slide_penalty`` already accepts a
# ``hand_body_names`` list, so passing the 4 spine names produces the
# generalised "engaged-EE tangential velocity" penalty.
ee_slide_penalty = hand_slide_penalty


def _ee_face_alignment_scores(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  ee_body_names: tuple[str, ...],
  face_local_axis: tuple[float, float, float],
) -> torch.Tensor:
  """Return per-EE wall-facing alignment scores in ``[0, 1]``."""
  asset: Entity = env.scene[asset_cfg.name]
  _, wall_normal = _SphereGeomCache.wall_plane(env)  # (E, 3) outer normal
  target = -wall_normal  # wall inward direction (robot → wall)

  face_local = torch.tensor(face_local_axis, dtype=torch.float32, device=env.device)
  vx, vy, vz = face_local[0], face_local[1], face_local[2]

  per_ee: list[torch.Tensor] = []
  for bname in ee_body_names:
    ids, _ = asset.find_bodies([bname])
    body_id = ids[0]
    quat = asset.data.body_link_quat_w[:, body_id, :]  # (E, 4) wxyz
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # Rotate face_local by quat (q · v · q^-1).
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    fx = vx + w * tx + (y * tz - z * ty)
    fy = vy + w * ty + (z * tx - x * tz)
    fz = vz + w * tz + (x * ty - y * tx)
    face = torch.stack([fx, fy, fz], dim=-1)  # (E, 3)
    dot = (face * target).sum(dim=-1).clamp(-1.0, 1.0)
    per_ee.append((dot + 1.0) * 0.5)
  return torch.stack(per_ee, dim=-1)


def ee_face_parallel_to_wall(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  ee_body_names: tuple[str, ...] = (
    "left_hand_spine",
    "right_hand_spine",
    "left_foot_spine",
    "right_foot_spine",
  ),
  face_local_axis: tuple[float, float, float] = (0.0, -1.0, 0.0),
) -> torch.Tensor:
  """Legacy positive alignment score: minimum across all configured EEs."""
  scores = _ee_face_alignment_scores(
    env, asset_cfg, ee_body_names, face_local_axis
  )
  return scores.min(dim=-1).values


def ee_face_parallel_error(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  ee_body_names: tuple[str, ...],
  event_term_name: str = "microspine_grip",
  command_name: str | None = "microspine_affordance",
  face_local_axis: tuple[float, float, float] = (0.0, -1.0, 0.0),
) -> torch.Tensor:
  """Mean face-alignment error for engaged or wall-contacting EEs only.

  A remote swing limb is free to reorient. Once it touches the wall it joins
  the mask, providing approach alignment without paying a constant positive
  reward for an already-static pose.
  """
  scores = _ee_face_alignment_scores(
    env, asset_cfg, ee_body_names, face_local_axis
  )
  physics = env.event_manager.get_term_cfg(event_term_name).func
  mask = physics.engaged.clone()
  if command_name is not None:
    affordance = env.command_manager.get_term(command_name)
    mask |= affordance.contact_count > 0
  mask_f = mask.float()
  return ((1.0 - scores) * mask_f).sum(dim=-1) / mask_f.sum(
    dim=-1
  ).clamp_min(1.0)


def microspine_support_loss(
  env: ManagerBasedRlEnv,
  event_term_name: str = "microspine_grip",
  minimum_supports: int = 2,
) -> torch.Tensor:
  """Penalty kernel for having fewer than the minimum safe anchors."""
  engaged = env.event_manager.get_term_cfg(event_term_name).func.engaged
  shortfall = minimum_supports - engaged.sum(dim=-1)
  return shortfall.clamp_min(0).float()


def unsafe_microspine_release_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
  minimum_supports: int = 2,
  settle_time_s: float = 0.5,
  reverse_release_cost: float = 0.0,
  tension_release_cost: float = 0.25,
  lateral_release_cost: float = 0.25,
  ultimate_release_cost: float = 1.0,
  support_loss_cost: float = 1.0,
) -> torch.Tensor:
  """Per-event cost that distinguishes controlled and failed disengagement.

  Reason 1 (reverse shear) is the passive spine's intended unhooking path and
  is free while sufficient support remains. Pull-off, lateral, and ultimate
  failures carry increasing costs. Dividing by ``step_dt`` makes the configured
  weight an actual one-shot event cost after RewardManager's dt scaling.
  """
  affordance = env.command_manager.get_term(command_name)
  released = affordance.forced_release_this_step
  reason = affordance.release_reason
  cost = torch.zeros_like(reason, dtype=torch.float32)
  reason_costs = (
    (1, reverse_release_cost),
    (2, tension_release_cost),
    (3, lateral_release_cost),
    (4, ultimate_release_cost),
  )
  for reason_id, value in reason_costs:
    cost += (released & (reason == reason_id)).float() * value

  engaged_count = affordance.engaged.sum(dim=-1)
  unsafe_support = released.any(dim=-1) & (engaged_count < minimum_supports)
  event_cost = cost.sum(dim=-1) + unsafe_support.float() * support_loss_cost
  settling = env.episode_length_buf.float() * env.step_dt <= settle_time_s
  return torch.where(
    settling,
    torch.zeros_like(event_cost),
    event_cost / env.step_dt,
  )


def confirmed_anchor_advance():
  """Factory: reward stable re-engagement at a genuinely different height.

  A rising edge becomes eligible only after it survives the full grace and
  debounce window. Reset-time contacts merely initialize per-EE baselines.
  Updating the baseline on both upward and downward confirmed re-grips makes
  a down/up cycle approximately zero-sum, while same-height engage spam earns
  nothing. The optional frontier bonus advances only when the lowest of all
  four confirmed anchors rises, marking completion of a full limb cycle.
  """
  state: dict[str, torch.Tensor | None] = {
    "last_anchor_z": None,
    "has_baseline": None,
    "prev_engaged": None,
    "swing_time": None,
    "pending": None,
    "pending_swing_time": None,
    "fresh": None,
  }

  def call(
    env: ManagerBasedRlEnv,
    *,
    event_term_name: str = "microspine_grip",
    settle_time_s: float = 0.5,
    confirmation_steps: int = 16,
    minimum_swing_time: float = 0.1,
    step_height: float = 0.08,
    max_normalized_advance: float = 1.0,
    frontier_bonus_scale: float = 0.5,
  ) -> torch.Tensor:
    physics = env.event_manager.get_term_cfg(event_term_name).func
    engaged: torch.Tensor = physics.engaged
    anchor_z: torch.Tensor = physics.anchor_pos_w[..., 2]
    engage_age: torch.Tensor = physics.engage_age
    shape = engaged.shape

    if state["last_anchor_z"] is None or state["last_anchor_z"].shape != shape:
      state["last_anchor_z"] = anchor_z.clone()
      state["has_baseline"] = engaged.clone()
      state["prev_engaged"] = engaged.clone()
      state["swing_time"] = torch.zeros(shape, device=engaged.device)
      state["pending"] = torch.zeros(shape, dtype=torch.bool, device=engaged.device)
      state["pending_swing_time"] = torch.zeros(shape, device=engaged.device)
      state["fresh"] = torch.zeros(
        shape[0], dtype=torch.bool, device=engaged.device
      )
      return torch.zeros(shape[0], device=engaged.device)

    last_anchor_z = state["last_anchor_z"]
    has_baseline = state["has_baseline"]
    prev_engaged = state["prev_engaged"]
    swing_time = state["swing_time"]
    pending = state["pending"]
    pending_swing_time = state["pending_swing_time"]
    fresh = state["fresh"]
    assert all(
      item is not None
      for item in (
        last_anchor_z,
        has_baseline,
        prev_engaged,
        swing_time,
        pending,
        pending_swing_time,
        fresh,
      )
    )

    settling = env.episode_length_buf.float() * env.step_dt <= settle_time_s
    initialize = fresh | settling
    initialize_ee = initialize.unsqueeze(-1)

    # Settling contacts establish baselines but can never produce reward.
    baseline_now = initialize_ee & engaged
    last_anchor_z = torch.where(baseline_now, anchor_z, last_anchor_z)
    has_baseline = has_baseline | baseline_now
    pending = pending & ~initialize_ee

    rising = engaged & ~prev_engaged & ~initialize_ee
    pending_swing_time = torch.where(rising, swing_time, pending_swing_time)
    pending = (pending | rising) & engaged
    swing_time = torch.where(
      engaged,
      torch.zeros_like(swing_time),
      swing_time + env.step_dt,
    )

    confirmed = pending & engaged & (engage_age >= confirmation_steps)
    eligible = (
      confirmed
      & has_baseline
      & (pending_swing_time >= minimum_swing_time)
    )

    old_all_baselined = has_baseline.all(dim=-1)
    old_frontier = last_anchor_z.min(dim=-1).values
    delta = ((anchor_z - last_anchor_z) / step_height).clamp(
      min=-max_normalized_advance,
      max=max_normalized_advance,
    )
    per_ee_score = torch.where(eligible, delta, torch.zeros_like(delta)).sum(
      dim=-1
    )

    last_anchor_z = torch.where(confirmed, anchor_z, last_anchor_z)
    has_baseline = has_baseline | confirmed
    new_frontier = last_anchor_z.min(dim=-1).values
    frontier_score = ((new_frontier - old_frontier) / step_height).clamp(
      min=-max_normalized_advance,
      max=max_normalized_advance,
    )
    frontier_valid = old_all_baselined & eligible.any(dim=-1)
    frontier_score = torch.where(
      frontier_valid, frontier_score, torch.zeros_like(frontier_score)
    )

    pending = pending & ~confirmed
    prev_engaged = engaged.clone()
    fresh.zero_()
    state["last_anchor_z"] = last_anchor_z
    state["has_baseline"] = has_baseline
    state["prev_engaged"] = prev_engaged
    state["swing_time"] = swing_time
    state["pending"] = pending
    state["pending_swing_time"] = pending_swing_time

    score = per_ee_score + frontier_bonus_scale * frontier_score
    return torch.where(initialize, torch.zeros_like(score), score / env.step_dt)

  def reset(env_ids=None) -> dict:
    fresh = state["fresh"]
    if fresh is None:
      return {}
    ids = slice(None) if env_ids is None else env_ids
    fresh[ids] = True
    state["last_anchor_z"][ids] = 0.0
    state["has_baseline"][ids] = False
    state["prev_engaged"][ids] = False
    state["pending"][ids] = False
    state["swing_time"][ids] = 0.0
    state["pending_swing_time"][ids] = 0.0
    return {}

  call.reset = reset  # type: ignore[attr-defined]
  return call


def microspine_forced_release_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Count of EEs released by the limit surface this step, shape (E,).

  Penalizing forced releases teaches the policy to keep anchor loads inside
  the hold envelope (lean into shear, avoid pull-off/lateral peeling) instead
  of treating grip as unconditional.
  """
  affordance = env.command_manager.get_term(command_name)
  return affordance.forced_release_this_step.float().sum(dim=1)


def commanded_anchor_advance(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_gait",
  max_normalized_advance: float = 1.5,
) -> torch.Tensor:
  """One-shot signed height change of the commanded, confirmed re-grip."""
  gait = env.command_manager.get_term(command_name)
  advance = gait.anchor_advance_this_step.clamp(
    min=-max_normalized_advance,
    max=max_normalized_advance,
  )
  return advance / env.step_dt


def productive_gait_cycle(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_gait",
) -> torch.Tensor:
  """One-shot reward after all four commanded re-grips advance upward."""
  gait = env.command_manager.get_term(command_name)
  return gait.productive_cycle_this_step.float() / env.step_dt


def unscheduled_microspine_release(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_gait",
) -> torch.Tensor:
  """One-shot cost for releases on any limb other than the commanded limb."""
  gait = env.command_manager.get_term(command_name)
  return gait.unscheduled_release_count / env.step_dt


def climb_gait_phase_timeout(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_gait",
) -> torch.Tensor:
  """One-shot cost when a contact-driven gait phase exceeds its deadline."""
  gait = env.command_manager.get_term(command_name)
  return gait.phase_timeout_this_step.float() / env.step_dt


def climb_gait_support_shortfall(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_gait",
) -> torch.Tensor:
  """Adaptive support cost: three-point crawl after all anchors are acquired."""
  gait = env.command_manager.get_term(command_name)
  return gait.support_shortfall


def commanded_ee_target_error_l2(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_gait",
  max_cost: float = 4.0,
) -> torch.Tensor:
  """Guide a released EE toward its commanded wall target during swing/seat."""
  gait = env.command_manager.get_term(command_name)
  delta = gait.command[:, -3:]
  active = (gait.phase == int(ClimbGaitPhase.SWING)) | (
    gait.phase == int(ClimbGaitPhase.SEAT)
  )
  cost = delta.square().mean(dim=1).clamp(max=max_cost)
  return torch.where(active, cost, torch.zeros_like(cost))


def climb_seat_press_shaping(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_gait",
  affordance_command_name: str = "microspine_affordance",
  force_scale: float = 12.0,
) -> torch.Tensor:
  """Partial credit for pressing the commanded pad into the wall during SEAT.

  Re-engagement was a sparse all-or-nothing jump: policies that released a
  pad almost never latched it back (engage 0.03 Hz vs release 0.83 Hz,
  2026-07-15) because nothing paid between "reached the target" and "fully
  latched". This pays for the missing middle step -- compression on the
  commanded pad -- and saturates at an engage-grade press (``force_scale``)
  so grinding harder into the wall earns nothing extra.
  """
  gait = env.command_manager.get_term(command_name)
  affordance = env.command_manager.get_term(affordance_command_name)
  press = affordance.pad_press_force.gather(
    1, gait.target_ee_index.unsqueeze(1)
  ).squeeze(1)
  in_seat = gait.phase == int(ClimbGaitPhase.SEAT)
  return in_seat.float() * (press / force_scale).clamp(max=1.0)


class climb_swing_progress_shaping:
  """Pay per-step approach of the released pad toward its commanded target.

  The 2026-07-17 rollout frames showed hands never leave their initial
  anchors: nothing rewards the risky middle of a regrip (weight shift plus
  reach), only its endpoints. This term pays the path itself: each step of
  SWING that shrinks the target distance earns the shrinkage (in step-height
  units, capped per step). Baselines re-arm without paying on phase or
  target switches and on fresh episodes, so discontinuities cannot mint
  fake progress.
  """

  def __init__(self, cfg, env):
    del cfg, env
    self._prev_dist: torch.Tensor | None = None
    self._prev_key: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_gait",
    progress_cap: float = 0.5,
  ) -> torch.Tensor:
    gait = env.command_manager.get_term(command_name)
    dist = gait.command[:, -3:].norm(dim=1)
    key = gait.target_ee_index * len(ClimbGaitPhase) + gait.phase
    if self._prev_dist is None or self._prev_dist.shape != dist.shape:
      self._prev_dist = dist.clone()
      self._prev_key = key.clone()
      return torch.zeros_like(dist)
    valid = (
      (gait.phase == int(ClimbGaitPhase.SWING))
      & (key == self._prev_key)
      & (env.episode_length_buf > 1)
    )
    # Signed, potential-based: retreat is charged what approach earns, so
    # waving the pad back and forth nets zero (the one-way clamp version was
    # farmed within 80 iterations on 2026-07-17).
    progress = (self._prev_dist - dist).clamp(min=-progress_cap, max=progress_cap)
    reward = torch.where(valid, progress, torch.zeros_like(progress))
    self._prev_dist = dist.clone()
    self._prev_key = key.clone()
    return reward
