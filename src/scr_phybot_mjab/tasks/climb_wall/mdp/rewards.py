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

  Per-hand timers update every step based on the ``microspine_adhesion``
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
    event_term_name: str = "microspine_adhesion",
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
  event_term_name: str = "microspine_adhesion",
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
  event_term_name: str = "microspine_adhesion",
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
  """Factory: reward net upward progress, not oscillatory vertical velocity.

  The term tracks the best base height reached in each episode and rewards
  only increases of that high-water mark. A policy that thrusts the hips up
  and falls back down can receive this reward once for the first peak, but not
  repeatedly for the same oscillation.
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
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]

    if state["best_z"] is None or state["best_z"].shape != z.shape:
      state["best_z"] = z.clone()
      state["fresh"] = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
      return torch.zeros_like(z)

    fresh = state["fresh"]
    assert fresh is not None
    if fresh.any():
      state["best_z"][fresh] = z[fresh]
      fresh.zero_()

    prev_best = state["best_z"]
    assert prev_best is not None
    new_best = torch.maximum(prev_best, z)
    progress_m = new_best - prev_best
    state["best_z"] = new_best

    if command_name is not None:
      cmd_z = env.command_manager.get_command(command_name)[:, 2].abs()
    else:
      cmd_z = torch.full_like(z, abs(target_vz))

    active = cmd_z > command_threshold
    target_step = (cmd_z * env.step_dt).clamp_min(1.0e-6)
    progress = (progress_m / target_step).clamp(max=max_normalized_progress)
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
    event_term_name: str = "microspine_adhesion",
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
  """Reward the WORST-aligned EE's contact face staying parallel to the wall.

  For each EE body, ``face_local_axis`` is rotated by the body's world quat
  and dotted with the wall inward normal. Per-EE score = ``(dot + 1) / 2``
  in [0, 1] (1 = face perfectly aligned, 0 = opposite). Returns the MIN
  across EEs — strictest "weakest link" measure: if any single spine
  rotates off-axis the reward drops, forcing ALL spines flush.

  For c2 spines, the contact face (sphere-bearing -y face of the collision
  box) maps to world wall-inward in default pose, so the default
  ``face_local_axis = (0, -1, 0)`` gives +1 per EE when the spine is flush
  against the wall.
  """
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
  scores = torch.stack(per_ee, dim=-1)  # (E, num_ee)
  return scores.min(dim=-1).values  # (E,)
