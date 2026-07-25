"""Reward functions for the wall-climbing task.

DeepMimic-style split (2026-07-21 redesign):

  * ``motion_*`` tracking terms (exp kernels against the IK cycle clip) teach
    the regrip MOTION. None of them references a global position — joint
    space, root-frame EE positions, and root orientation only.
  * Height/velocity progress terms remain the task DRIVE: climbing is paid
    for by actual base rise, not by reproducing the clip's world trajectory.
  * Microspine engage/safety terms remain the PHYSICS contract: minimum
    support, controlled releases, no sliding of engaged pads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .motion import ClimbMotionCommand


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


# ---------------------------------------------------------------------------
# DeepMimic-style motion tracking. All terms read the ``climb_motion`` command
# (mdp.motion.ClimbMotionCommand) and use the BeyondMimic/mjlab exp kernel
# ``exp(-error / std**2)``. No term references a global position: the drive to
# actually gain height stays with the progress rewards below.
# ---------------------------------------------------------------------------


def _motion(env: ManagerBasedRlEnv, command_name: str) -> "ClimbMotionCommand":
  return env.command_manager.get_term(command_name)


def motion_joint_pos_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.2,
) -> torch.Tensor:
  """exp kernel on the mean squared clip-joint angle error (rad)."""
  motion = _motion(env, command_name)
  error = (motion.ref_joint_pos - motion.robot_joint_pos).square().mean(dim=-1)
  return torch.exp(-error / std**2)


def motion_joint_vel_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 1.5,
) -> torch.Tensor:
  """exp kernel on the mean squared clip-joint velocity error (rad/s)."""
  motion = _motion(env, command_name)
  error = (motion.ref_joint_vel - motion.robot_joint_vel).square().mean(dim=-1)
  return torch.exp(-error / std**2)


def motion_ee_pos_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.1,
) -> torch.Tensor:
  """exp kernel on root-frame end-effector position error (m).

  Root-relative on both sides, so this shapes the limb geometry of the cycle
  (reach, swing arcs) without ever pinning the base to the clip's world
  trajectory.
  """
  motion = _motion(env, command_name)
  error = (
    (motion.ref_ee_pos_r - motion.robot_ee_pos_r).square().sum(dim=-1).mean(dim=-1)
  )
  return torch.exp(-error / std**2)


def motion_root_ori_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.3,
) -> torch.Tensor:
  """exp kernel on the root orientation geodesic error (rad).

  Orientation only — the reference base tilts and rights itself through the
  hand swings, and that posture is part of the motion. Position stays free.
  """
  from mjlab.utils.lab_api.math import quat_error_magnitude

  motion = _motion(env, command_name)
  error = quat_error_magnitude(
    motion.ref_root_quat_w, motion.robot_root_quat_w
  ).square()
  return torch.exp(-error / std**2)


def motion_engage_mismatch(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  grace_time_s: float | None = None,
) -> torch.Tensor:
  """Per-EE bill for a latch state disagreeing with the clip's schedule.

  Pair with a negative weight. This is the DeepMimic-style replacement for
  the old commanded-gait machinery: the clip says which pads are anchored at
  each phase; the policy must physically release/press to match it (the
  affordance still owns the latch).

  With ``grace_time_s`` the bill scales linearly with the time to the
  NEAREST schedule edge, saturating at full price ``grace_time_s`` away from
  it. A latch transition can never be instantaneous (0.06 s engage debounce,
  0.20 s re-engage cooldown), so full price right at a scheduled flip bills
  the physically impossible. The graded bill is a piecewise-linear function
  of the same countdown channels the policy observes in ``motion_command``,
  so the schedule -> action mapping is learnable from the chart itself —
  and transfers to a different clip — rather than memorized against the
  phase clock.
  """
  motion = _motion(env, command_name)
  mismatch = (motion.ref_engaged != motion.robot_engaged).float()
  if grace_time_s is not None and grace_time_s > 0.0:
    edge_seconds = motion.ref_edge_distance * motion.cycle_time_s
    mismatch = mismatch * (edge_seconds / grace_time_s).clamp(max=1.0)
  return mismatch.sum(dim=-1)


def motion_scheduled_press_shaping(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  press_norm_n: float = 12.0,
  regrip_lead_time_s: float = 0.3,
) -> torch.Tensor:
  """Seat-phase step credit: pay pad press force inside the regrip window.

  Pair with a small positive weight. Seven runs in a row parked a limb pair
  because a regrip's first motions (reach, press, hold through debounce)
  earn nothing until the latch closes — undirected exploration never strings
  them together. This term pays the press itself, but ONLY per EE inside its
  scheduled seating window: the reference says the pad should be anchored
  (``ref_engaged``) or is about to be (regrip countdown within
  ``regrip_lead_time_s`` — the same observed channel), AND the robot has not
  latched yet. Payment stops the instant the latch closes: an engaged
  microspine anchor carries tension, pressing is only the doorway. Pressing
  outside the window pays zero, and the per-EE ceiling (press_norm_n
  saturation x the configured weight) must stay below the mismatch bill so
  press-and-never-latch is strictly net-negative.
  """
  motion = _motion(env, command_name)
  affordance = motion._affordance()
  lead = regrip_lead_time_s / motion.cycle_time_s
  regrip_soon = motion.ref_edge_countdowns[..., 1] <= lead
  window = (motion.ref_engaged | regrip_soon) & ~motion.robot_engaged
  press = (affordance.pad_press_force / press_norm_n).clamp(max=1.0)
  return (press * window.float()).sum(dim=-1)


def motion_ref_height_tracking():
  """Factory: exp kernel on (robot climb - reference climb) since settling.

  The climbing DRIVE as an imitation term. Every free-standing height income
  ("any rise pays, no questions asked") funded a reference-ignoring gait —
  scramble, arms-only, legs-only — and each takeoff bought climb income with
  falling schedule adherence. Here the money follows the ghost instead: the
  reference's cumulative climb since this episode settled (``ref_root_pos_w``
  accrues one step height per loop) is the target, and the payout decays as
  the robot falls BEHIND or runs AHEAD of it symmetrically. RSI places the
  robot exactly on the profile, so holding a proper stance and moving with
  the ghost pays full from the first settled step — the honest path finally
  has dense income. Bounded in (0, 1]: the per-step caps and budget
  machinery of the free-standing terms have nothing left to guard.
  """
  state: dict[str, torch.Tensor | None] = {
    "fresh": None,
    "z_base": None,
    "ref_base": None,
  }

  def call(
    env: ManagerBasedRlEnv,
    *,
    asset_cfg: SceneEntityCfg,
    motion_command_name: str = "climb_motion",
    std: float = 0.05,
    settle_time_s: float = 0.5,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]
    motion = env.command_manager.get_term(motion_command_name)
    ref_z = motion.ref_root_pos_w[:, 2]

    if state["fresh"] is None or state["fresh"].shape != z.shape[:1]:
      state["fresh"] = torch.zeros(
        z.shape[0], dtype=torch.bool, device=z.device
      )
      state["z_base"] = z.clone()
      state["ref_base"] = ref_z.clone()
      return torch.ones_like(z)

    fresh = state["fresh"]
    z_base = state["z_base"]
    ref_base = state["ref_base"]
    assert z_base is not None and ref_base is not None
    settling = env.episode_length_buf.float() * env.step_dt <= settle_time_s
    initialize = fresh | settling
    z_base = torch.where(initialize, z, z_base)
    ref_base = torch.where(initialize, ref_z, ref_base)
    state["z_base"] = z_base
    state["ref_base"] = ref_base
    fresh.zero_()

    dh = (z - z_base) - (ref_z - ref_base)
    return torch.exp(-(dh**2) / std**2)

  def reset(env_ids=None) -> dict:
    if state["fresh"] is None:
      return {}
    ids = slice(None) if env_ids is None else env_ids
    state["fresh"][ids] = True
    return {}

  call.reset = reset  # type: ignore[attr-defined]
  return call


# ---------------------------------------------------------------------------
# Task drive: height and velocity. These pay for actual climbing, independent
# of how closely the motion is imitated.
# ---------------------------------------------------------------------------


def climb_height_progress():
  """Factory: reward new height after the reset pose has physically settled.

  The term tracks the best base height reached in each episode and rewards
  only increases of that high-water mark. A policy that thrusts the hips up
  and falls back down can receive this reward once for the first peak, but not
  repeatedly for the same oscillation. During ``settle_time_s`` the high-water
  mark follows the base without reward, preventing the unavoidable reset sag
  from creating a several-centimeter reward dead zone.
  """
  state: dict[str, torch.Tensor | None] = {
    "best_z": None,
    "fresh": None,
    "budget": None,
  }

  def call(
    env: ManagerBasedRlEnv,
    *,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    target_vz: float = 0.08,
    command_threshold: float = 0.01,
    max_normalized_progress: float = 1.5,
    settle_time_s: float = 0.5,
    rate_budget_scale: float | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]

    if state["best_z"] is None or state["best_z"].shape != z.shape:
      state["best_z"] = z.clone()
      state["fresh"] = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
      state["budget"] = torch.zeros_like(z)
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
    progress = torch.where(
      active & ~initialize, progress, torch.zeros_like(progress)
    )
    if rate_budget_scale is not None:
      # Cumulative income cap: total normalized progress paid per episode may
      # not exceed rate_budget_scale x the commanded climb distance over the
      # same (active, settled) time. The reference's within-cycle burst spends
      # headroom banked during its non-climbing phases, so honest cycles pay
      # in full while sustained overspeed scrambling is rate-limited — the
      # per-step cap above cannot distinguish the two.
      budget = state["budget"]
      assert budget is not None
      budget = budget + (active & ~initialize).float() * rate_budget_scale
      paid = torch.minimum(progress, budget)
      state["budget"] = budget - paid
      progress = paid
    return progress

  def reset(env_ids=None) -> dict:
    if state["fresh"] is None:
      return {}
    if env_ids is None:
      state["fresh"].fill_(True)
      if state["budget"] is not None:
        state["budget"].zero_()
    else:
      state["fresh"][env_ids] = True
      if state["budget"] is not None:
        state["budget"][env_ids] = 0.0
    return {}

  call.reset = reset  # type: ignore[attr-defined]
  return call


def signed_climb_height_progress():
  """Factory: dense signed base-height change after initial settling.

  Unlike a positive-only velocity reward, a closed up/down cycle has roughly
  zero undiscounted return. Its weight should stay below the high-water term:
  it supplies a recovery gradient below the previous maximum, while genuinely
  new height remains the primary objective.

  Two mutually exclusive upward-income caps: ``reference_budget_command``
  bounds cumulative net income by the motion reference's own climb profile
  (the cap IS the reference's value at this instant), while
  ``rate_budget_scale`` is the legacy flat-rate budget. Descent is never
  budgeted in either mode.
  """
  state: dict[str, torch.Tensor | None] = {
    "prev_z": None,
    "fresh": None,
    "budget": None,
    "ref_base": None,
    "net": None,
  }

  def call(
    env: ManagerBasedRlEnv,
    *,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    target_vz: float = 0.08,
    command_threshold: float = 0.01,
    max_normalized_progress: float = 1.0,
    settle_time_s: float = 0.5,
    rate_budget_scale: float | None = None,
    reference_budget_command: str | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]

    if state["prev_z"] is None or state["prev_z"].shape != z.shape:
      state["prev_z"] = z.clone()
      state["fresh"] = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
      state["budget"] = torch.zeros_like(z)
      state["ref_base"] = torch.zeros_like(z)
      state["net"] = torch.zeros_like(z)
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
    out = torch.where(active & ~initialize, progress, torch.zeros_like(progress))
    if reference_budget_command is not None:
      # Upper cap = the reference's own value: cumulative net income may not
      # exceed the height the motion reference itself has gained since this
      # episode settled (``ref_root_pos_w`` accrues one step height per loop,
      # so the cap keeps climbing with the schedule). Climbing ahead of the
      # schedule earns nothing until the reference catches up; descent bills
      # in full and restores the same headroom, so a sag-and-recover loop
      # nets zero instead of bleeding.
      motion = env.command_manager.get_term(reference_budget_command)
      ref_z = motion.ref_root_pos_w[:, 2]
      ref_base = state["ref_base"]
      net = state["net"]
      assert ref_base is not None and net is not None
      ref_base = torch.where(initialize, ref_z, ref_base)
      net = torch.where(initialize, torch.zeros_like(net), net)
      state["ref_base"] = ref_base
      allowance = ((ref_z - ref_base) / target_step - net).clamp_min(0.0)
      paid = torch.minimum(out.clamp_min(0.0), allowance)
      out = paid + out.clamp_max(0.0)
      state["net"] = net + out
    elif rate_budget_scale is not None:
      # Same cumulative cap as climb_height_progress, applied to the upward
      # side only: descent keeps its full recovery gradient.
      budget = state["budget"]
      assert budget is not None
      budget = budget + (active & ~initialize).float() * rate_budget_scale
      paid = torch.minimum(out.clamp_min(0.0), budget)
      state["budget"] = budget - paid
      out = paid + out.clamp_max(0.0)
    return out

  def reset(env_ids=None) -> dict:
    if state["fresh"] is None:
      return {}
    ids = slice(None) if env_ids is None else env_ids
    state["fresh"][ids] = True
    if state["budget"] is not None:
      state["budget"][ids] = 0.0
    if state["net"] is not None:
      state["net"][ids] = 0.0
    return {}

  call.reset = reset  # type: ignore[attr-defined]
  return call


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


# ---------------------------------------------------------------------------
# Microspine physics contract: minimum support, controlled release, no slide,
# wall-facing pads. These are unchanged by the DeepMimic redesign — tracking
# teaches the motion, these guarantee the physical rules while doing it.
# ---------------------------------------------------------------------------


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


def hand_slide_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  event_term_name: str = "microspine_grip",
  hand_body_names: tuple[str, ...] = ("left_hand", "right_hand"),
) -> torch.Tensor:
  """Penalise tangential wall-plane velocity of ENGAGED end-effectors.

  Gate the slip speed by the actual ``engaged`` latch state rather than a
  desired-contact schedule, so the penalty tracks the real "grip should not
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


# ``hand_slide_penalty`` accepts any EE body list, so passing the 4 spine
# names produces the generalised "engaged-EE tangential velocity" penalty.
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


# ---------------------------------------------------------------------------
# Regularization.
# ---------------------------------------------------------------------------


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
