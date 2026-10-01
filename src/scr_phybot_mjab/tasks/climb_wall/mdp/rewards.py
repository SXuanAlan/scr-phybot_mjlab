"""Reward functions for the wall-climbing task.

Empirical tasks select 16 terms: BeyondMimic body tracking, climbing progress
and landing, contact schedule/support/release shaping, regularization and a
failure penalty. No explicit survival reward is used. Legacy joint/EE and
auxiliary terms remain available to the Wang transition configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .motion import ClimbMotionCommand


class _SphereGeomCache:
  """Contact-proxy geometry cache (legacy class name), for boxes and spheres.

  Wall pose is queried per-step from the wall entity's mocap state so that
  curriculum-driven tilts are reflected in the derived surface/normal maths.
  """

  _instance: _SphereGeomCache | None = None
  geom_ids: torch.Tensor
  sizes: torch.Tensor
  is_box: torch.Tensor
  wall_half: float
  initialized: bool = False

  @classmethod
  def get(cls, env: ManagerBasedRlEnv) -> _SphereGeomCache:
    if cls._instance is None or getattr(cls._instance, "model", None) is not env.sim.mj_model:
      cls._instance = cls()
      mj = env.sim.mj_model
      ids = []
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
      cls._instance.geom_ids = torch.tensor(ids, dtype=torch.long, device=env.device)
      cls._instance.model = mj
      cls._instance.sizes = torch.tensor(mj.geom_size[ids], dtype=torch.float32, device=env.device)
      cls._instance.is_box = torch.tensor(mj.geom_type[ids] == mujoco.mjtGeom.mjGEOM_BOX, device=env.device)
      if not all(t in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE) for t in mj.geom_type[ids]):
        raise ValueError("Pad distance supports only box and sphere contact proxies")
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
    """Per-proxy closest-surface-to-wall signed distance. (E, 16).
    Positive=gap (contact surface hasn't reached wall yet).
    Negative=penetration.
    Zero=touching.
    """
    c = cls.get(env)
    wall_pos, normal = cls.wall_plane(env)  # (E,3), (E,3)
    geom_pos = env.sim.data.geom_xpos[:, c.geom_ids, :]  # (E, 16, 3)
    delta = geom_pos - wall_pos.unsqueeze(1)  # (E, 16, 3)
    # Signed perpendicular distance from geom center to wall plane:
    signed = (delta * normal.unsqueeze(1)).sum(dim=-1) - c.wall_half  # (E, 16)
    # Box support along the wall normal depends on its full orientation;
    # treating size[0] as a radius is incorrect, especially at tilted poses.
    rotations = env.sim.data.geom_xmat[:, c.geom_ids].reshape(-1, len(c.geom_ids), 3, 3)
    local_normal = torch.einsum("egij,ei->egj", rotations, normal)
    extent = (local_normal.abs() * c.sizes.unsqueeze(0)).sum(-1)
    extent = torch.where(c.is_box.unsqueeze(0), extent, c.sizes[:, 0].unsqueeze(0))
    return signed - extent


# ---------------------------------------------------------------------------
# Tracking terms read the climb_motion command and use exp(-error/std**2).
# Body poses align in XY/yaw; body velocities stay in the world frame.
# ---------------------------------------------------------------------------


def _motion(env: ManagerBasedRlEnv, command_name: str) -> "ClimbMotionCommand":
  return env.command_manager.get_term(command_name)


def _graced_mismatch(
  motion: "ClimbMotionCommand",
  grace_time_s: float | None,
  grace_fraction: float | None,
) -> torch.Tensor:
  """Per-EE mismatch scaled by the edge-distance grace ramp, (E, H).

  ``grace_fraction`` prices the grace as a fraction of the CURRENT scheduled
  run's length (per EE, per frame): a weld-through bill is then a constant
  fraction of the violated window no matter how the clip is timed. The fixed
  ``grace_time_s`` alternative silently discounted welding when the clip was
  compressed — at 0.25 s over a 0.6 s foot window only the middle 0.1 s
  billed full price (motionref16's weld economics). Fraction takes
  precedence when both are given.
  """
  mismatch = (motion.ref_engaged != motion.robot_engaged).float()
  if grace_fraction is not None and grace_fraction > 0.0:
    edge_seconds = motion.ref_edge_distance * motion.cycle_time_s
    grace_seconds = (motion.ref_run_time * motion.cycle_time_s * grace_fraction).clamp(min=1.0e-6)
    return mismatch * (edge_seconds / grace_seconds).clamp(max=1.0)
  if grace_time_s is not None and grace_time_s > 0.0:
    edge_seconds = motion.ref_edge_distance * motion.cycle_time_s
    return mismatch * (edge_seconds / grace_time_s).clamp(max=1.0)
  return mismatch


def _engage_match_gate(
  motion: "ClimbMotionCommand",
  floor: float,
  grace_time_s: float,
  grace_fraction: float | None = None,
) -> torch.Tensor:
  """Schedule-adherence income gate in [floor, 1], per env.

  The mirror of ``motion_engage_mismatch``: each EE's mismatch is priced by
  the same edge-distance ramp the bill uses (near a scheduled flip it is
  forgiven — instant compliance is physically impossible), and the gate is
  1 - mean(billed mismatch), squeezed above ``floor``. Pose-tracking income
  multiplied by this gate flows in full only while the latch pattern follows
  the chart; a limb-welding statue keeps at most ``floor`` + its accidental
  match share. The floor keeps early pose learning funded (RSI births start
  fully matched, so the gradient toward keeping the match exists from step
  one).
  """
  mismatch = _graced_mismatch(motion, grace_time_s, grace_fraction)
  # R4: pending reseat EEs are exempt here too — a frozen-clock recovery
  # must not throttle the whole robot's tracking income through the gate.
  mismatch = mismatch * (~motion._affordance().dr_release_pending).float()
  match = 1.0 - mismatch.mean(dim=-1)
  return floor + (1.0 - floor) * match


def motion_body_position_error_exp(
  env,
  command_name="climb_motion",
  std=0.1,
  engage_gate_floor=None,
  engage_gate_grace_s=0.25,
  engage_gate_grace_fraction=None,
):
  """BeyondMimic mean body-origin position error after XY/yaw alignment."""
  motion = _motion(env, command_name)
  reference, _ = motion.aligned_body_reference()
  error = (reference - motion.robot_body_pos_w).square().sum(-1).mean(-1)
  reward = torch.exp(-error / std**2)
  if engage_gate_floor is not None:
    reward *= _engage_match_gate(motion, engage_gate_floor, engage_gate_grace_s, engage_gate_grace_fraction)
  return reward


def motion_body_orientation_error_exp(
  env,
  command_name="climb_motion",
  std=0.3,
  engage_gate_floor=None,
  engage_gate_grace_s=0.25,
  engage_gate_grace_fraction=None,
):
  """BeyondMimic mean squared geodesic body orientation error."""
  from mjlab.utils.lab_api.math import quat_error_magnitude

  motion = _motion(env, command_name)
  _, reference = motion.aligned_body_reference()
  error = quat_error_magnitude(reference, motion.robot_body_quat_w).square().mean(-1)
  reward = torch.exp(-error / std**2)
  if engage_gate_floor is not None:
    reward *= _engage_match_gate(motion, engage_gate_floor, engage_gate_grace_s, engage_gate_grace_fraction)
  return reward


def motion_body_linear_velocity_error_exp(env, command_name="climb_motion", std=1.0):
  motion = _motion(env, command_name)
  error = (motion.ref_body_lin_vel_w - motion.robot_body_lin_vel_w).square().sum(-1).mean(-1)
  return torch.exp(-error / std**2)


def motion_body_angular_velocity_error_exp(env, command_name="climb_motion", std=3.14):
  motion = _motion(env, command_name)
  error = (motion.ref_body_ang_vel_w - motion.robot_body_ang_vel_w).square().sum(-1).mean(-1)
  return torch.exp(-error / std**2)


def undesired_body_contacts(env, sensor_name="climb_undesired_contact", threshold=1.0):
  """Count unwanted body contacts exceeding threshold during a policy step.

  The sensor excludes expected pad/toe bodies; a peak over substeps retains
  brief impacts. maxforce reduction prevents opposing contacts cancelling.
  """
  data = env.scene.sensors[sensor_name].data
  forces = data.force_history
  if forces is None:
    forces = data.force.unsqueeze(2)
  return (forces.norm(dim=-1).amax(-1) > threshold).float().sum(-1)


def motion_root_ori_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.3,
  engage_gate_floor: float | None = None,
  engage_gate_grace_s: float = 0.25,
  engage_gate_grace_fraction: float | None = None,
) -> torch.Tensor:
  """exp kernel on the root orientation geodesic error (rad).

  Orientation only — the reference base tilts and rights itself through the
  hand swings, and that posture is part of the motion. Position stays free.
  ``engage_gate_floor``: see ``_engage_match_gate``.
  """
  from mjlab.utils.lab_api.math import quat_error_magnitude

  motion = _motion(env, command_name)
  error = quat_error_magnitude(motion.ref_root_quat_w, motion.robot_root_quat_w).square()
  reward = torch.exp(-error / std**2)
  if engage_gate_floor is not None:
    reward = reward * _engage_match_gate(motion, engage_gate_floor, engage_gate_grace_s, engage_gate_grace_fraction)
  return reward


def motion_engage_mismatch(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  grace_time_s: float | None = None,
  grace_fraction: float | None = None,
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

  ``grace_fraction`` prices the grace as a fraction of the violated window's
  own length instead of a fixed number of seconds (see ``_graced_mismatch``)
  — the weld-through bill then survives clip retiming unchanged.
  """
  motion = _motion(env, command_name)
  bill = _graced_mismatch(motion, grace_time_s, grace_fraction)
  # R4: EEs lost to a dr-forced release are exempt while their reseat hold
  # is armed — the frozen clock exists so recovery is not billed as truancy.
  pending = motion._affordance().dr_release_pending
  return (bill * (~pending).float()).sum(dim=-1)


def motion_anchor_landing():
  """Factory: pay each scheduled latch by landing accuracy on the anchor ladder.

  The shape rewards are deliberately root-relative and the engage gate only
  checks latch TIMING, so a policy can execute the whole regrip cycle in
  place — releasing and re-latching each pad at its old spot — and still
  collect nearly every imitation term (the measured iter-1400 basin: match
  0.98, net climb ~20% of the ghost). This term closes that hole with the
  one channel none of them read: WHERE the pad lands. On an engage rising
  edge inside that EE's scheduled window it pays
  ``exp(-(|anchor - ref_anchor|/std)^2)`` once — the reference anchor ladder
  climbs one step height per cycle, so landing at the old spot (30 mm short)
  earns ~0.37 at std 0.03 while the scheduled rung pays 1.0.

  Anti-farming: one payment per (EE, schedule window), cleared only when the
  reference window closes — release-and-relatch inside a window collects
  nothing extra (and still pays the mismatch/unsafe-release bills).
  Off-schedule latches never pay.
  """
  state: dict[str, torch.Tensor | None] = {"paid": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    std: float = 0.03,
  ) -> torch.Tensor:
    motion = _motion(env, command_name)
    affordance = motion._affordance()
    executor = affordance._get_executor()
    ref_on = motion.ref_engaged
    paid = state["paid"]
    if paid is None or paid.shape != ref_on.shape:
      paid = torch.zeros_like(ref_on)
    # A window already open at spawn (RSI seeds the latch mid-window) is
    # marked PAID: its honest play is "stay latched" with no landing event,
    # so leaving it eligible would pay a release-and-relatch churn (+0.8
    # gross vs ~0.26 of mismatch bill at these weights). Eligibility starts
    # at each EE's first fresh window.
    fresh = (env.episode_length_buf <= 1).unsqueeze(-1)
    paid = torch.where(fresh, affordance._engaged & ref_on, paid)
    paid = paid & ref_on  # window closed -> eligibility restored
    landed = affordance.engaged_this_step & ref_on & ~paid
    # The executor captures the weld anchor one step after the latch
    # decision, so judge the CURRENT pad center — the exact point the
    # capture will pin next step (anchor projection is off in this task).
    error = (executor.pad_center_world() - motion.ref_anchor_pos_w).norm(dim=-1)
    pay = torch.exp(-((error / std) ** 2))
    state["paid"] = paid | landed
    return (pay * landed.float()).sum(dim=-1)

  return call


def motion_swing_approach(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.05,
) -> torch.Tensor:
  """Dense shaping toward the upcoming landing rung during scheduled swings.

  The anchor-landing bonus pays once, at the latch, judged on the anchor
  ladder — but reaching a rung one step height UP requires the whole
  preceding swing to aim there, and none of those intermediate motions earn
  anything on their own (measured motionref16: landing income stuck at 3-5%
  of its ceiling because exploration never sampled a high landing). This
  pays the approach itself: while an EE is scheduled OFF (its swing window)
  and actually released, an exp kernel on the pad-center distance to
  ``ref_anchor_pos_w`` — which during a swing IS the upcoming landing
  target (see ClimbMotionCommand.ref_anchor_pos_w). Latching ends the pay
  (the landing bonus takes over); a released pad during a scheduled ON
  window earns nothing here — that mismatch is billed, not rewarded.
  """
  motion = _motion(env, command_name)
  affordance = motion._affordance()
  executor = affordance._get_executor()
  active = (~motion.ref_engaged) & (~motion.robot_engaged)
  error = (executor.pad_center_world() - motion.ref_anchor_pos_w).norm(dim=-1)
  pay = torch.exp(-((error / std) ** 2))
  return (pay * active.float()).sum(dim=-1)


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
    motion = env.command_manager.get_term(motion_command_name)
    # Along-wall climb coordinates (world z on a vertical wall, bit-identical
    # there). BOTH sides project onto the same per-env wall axis — rotating
    # only one of them would silently misprice every tilted rung.
    up = motion.wall_up_w
    z = (asset.data.root_link_pos_w * up).sum(dim=-1)
    ref_z = (motion.ref_root_pos_w * up).sum(dim=-1)

    if state["fresh"] is None or state["fresh"].shape != z.shape[:1]:
      state["fresh"] = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
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


# ---------------------------------------------------------------------------
# Microspine physics contract: minimum support, controlled release, no slide,
# wall-facing pads. These are unchanged by the DeepMimic redesign — tracking
# teaches the motion, these guarantee the physical rules while doing it.
# ---------------------------------------------------------------------------


def microspine_support_loss(
  env: ManagerBasedRlEnv,
  event_term_name: str = "microspine_grip",
  minimum_supports: int = 2,
) -> torch.Tensor:
  """Penalty kernel for having fewer than the minimum safe anchors."""
  engaged = env.event_manager.get_term_cfg(event_term_name).func.engaged
  shortfall = minimum_supports - engaged.sum(dim=-1)
  return shortfall.clamp_min(0).float()


def microspine_release_load_reward():
  """Factory: reward scheduled unloading, penalize tangentially loaded release.

  Uses the empirical judge's PRE-release unit loads, including twist-induced
  shear/lateral loads that can cancel in the pad resultant. Normal force is
  not an unloading target. The physical release rules remain with the judge.
  Each scheduled release window (lead-in plus OFF interval) can pay once;
  repeated releases still incur the load cost. External DR releases pay/cost
  nothing. Divide by dt so the configured weight is a per-event amount.
  """
  state: dict[str, torch.Tensor | None] = {"paid": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "microspine_affordance",
    motion_command_name: str = "climb_motion",
    shear_scale_n: float = 2.0,
    lateral_scale_n: float = 2.0,
    shear_cost: float = 0.25,
    lateral_cost: float = 0.1,
    max_load_cost: float = 4.0,
    minimum_supports: int = 3,
    release_lead_time_s: float = 0.25,
    settle_time_s: float = 0.5,
  ) -> torch.Tensor:
    """Scales are N per active unit; the most loaded unit sets the score."""
    if min(shear_scale_n, lateral_scale_n) <= 0 or max_load_cost <= 0:
      raise ValueError("Release-load scales and maximum cost must be positive")
    if min(shear_cost, lateral_cost, release_lead_time_s) < 0:
      raise ValueError("Release-load costs and lead time must be nonnegative")
    affordance = env.command_manager.get_term(command_name)
    motion = _motion(env, motion_command_name)
    # Populated before the latch falling edge; do not read post-release
    # contacts, whose force has already vanished. Axes are [shear,lateral,normal].
    loads = affordance.unit_load_tln
    finite = torch.isfinite(loads).all(dim=-1).all(dim=-1)
    shear_sq = (loads[..., 0] / shear_scale_n).square()
    lateral_sq = (loads[..., 1] / lateral_scale_n).square()
    quality = torch.exp(-(shear_sq + lateral_sq).amax(dim=-1))
    load_cost = (shear_cost * shear_sq + lateral_cost * lateral_sq).amax(dim=-1)
    load_cost = torch.where(finite, load_cost.clamp(max=max_load_cost), max_load_cost)

    reason = affordance.release_reason
    released = affordance.forced_release_this_step & (reason > 0) & (reason != 6)
    release_soon = motion.ref_edge_countdowns[..., 0] * motion.cycle_time_s <= release_lead_time_s
    window = ~motion.ref_engaged | release_soon
    paid = state["paid"]
    if paid is None or paid.shape != window.shape or paid.device != window.device:
      paid = torch.zeros_like(window)
    # An RSI/reset born inside an existing swing cannot earn a fresh release
    # bonus by relatching there. Partial resets only touch their own env rows.
    fresh = (env.episode_length_buf <= 1).unsqueeze(-1)
    paid = torch.where(fresh, window, paid) & window
    support_ok = affordance.engaged.sum(dim=-1) >= minimum_supports
    recovering = affordance.dr_release_pending.any(dim=-1)
    # Unknown/early empirical failures are not successful controlled releases.
    controlled_reason = (reason >= 1) & (reason <= 5)
    pay = released & controlled_reason & finite & window & ~paid & support_ok.unsqueeze(-1) & ~recovering.unsqueeze(-1)
    # Consume even an unsuccessful first release: relatch/release retries in
    # the same window cannot turn a bad release into extra positive income.
    state["paid"] = paid | (released & window)
    score = (torch.where(pay, quality, 0.0) - torch.where(released, load_cost, 0.0)).sum(dim=-1)
    settling = env.episode_length_buf.float() * env.step_dt <= settle_time_s
    return torch.where(settling, 0.0, score / env.step_dt)

  return call


def unsafe_microspine_release_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
  minimum_supports: int = 2,
  settle_time_s: float = 0.5,
  reverse_release_cost: float = 0.0,
  tension_release_cost: float = 0.25,
  lateral_release_cost: float = 0.25,
  ultimate_release_cost: float = 1.0,
  tilt_release_cost: float = 0.5,
  support_loss_cost: float = 1.0,
  motion_command_name: str | None = None,
  scheduled_reverse_cost: float = 0.3,
  scheduled_reverse_grace_s: float = 0.25,
) -> torch.Tensor:
  """Per-event cost that distinguishes controlled and failed disengagement.

  Reason 1 (reverse shear) is the passive spine's intended unhooking path and
  has no base reason cost while sufficient support remains. The separate
  release-load reward can still penalize its tangential load. Pull-off, lateral, and ultimate
  failures carry increasing costs. Dividing by ``step_dt`` makes the configured
  weight an actual one-shot event cost after RewardManager's dt scaling.

  With ``motion_command_name`` set, a reverse-shear unhook of an EE the clip
  schedules ENGAGED bills ``scheduled_reverse_cost`` — scaled down to zero
  within ``scheduled_reverse_grace_s`` of the nearest schedule edge, so the
  on-time unhook at a scheduled release stays free. This prices the
  latch-jiggle-unhook churn (motionref16b: 43-45 free reverse releases per
  30 s on the abandoned right foot) without touching legitimate releases.
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
    (5, tilt_release_cost),  # tilt-peel: rotated the patch off its grip
    (7, ultimate_release_cost),  # empirical measured-domain rejection
    (8, tension_release_cost),  # empirical early-loading failure
  )
  for reason_id, value in reason_costs:
    cost += (released & (reason == reason_id)).float() * value

  if motion_command_name is not None and scheduled_reverse_cost > 0.0:
    motion = _motion(env, motion_command_name)
    off_schedule = released & (reason == 1) & motion.ref_engaged
    edge_seconds = motion.ref_edge_distance * motion.cycle_time_s
    ramp = (
      (edge_seconds / max(scheduled_reverse_grace_s, 1.0e-6)).clamp(max=1.0)
      if scheduled_reverse_grace_s > 0.0
      else torch.ones_like(edge_seconds)
    )
    cost += off_schedule.float() * ramp * scheduled_reverse_cost

  engaged_count = affordance.engaged.sum(dim=-1)
  unsafe_support = released.any(dim=-1) & (engaged_count < minimum_supports)
  event_cost = cost.sum(dim=-1) + unsafe_support.float() * support_loss_cost
  settling = env.episode_length_buf.float() * env.step_dt <= settle_time_s
  return torch.where(
    settling,
    torch.zeros_like(event_cost),
    event_cost / env.step_dt,
  )


# ---------------------------------------------------------------------------
# Regularization.
# ---------------------------------------------------------------------------
