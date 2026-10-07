"""Climb tracking rewards: BeyondMimic body tracking adapted to climbing.

Pose kernels compare XY/yaw-aligned body poses with the reference and are
gated by schedule adherence; body velocities stay in the world frame. The
reference-height ghost pays absolute climbing progress, regrips get landing
and approach shaping, and the schedule mismatch bill prices latch states
that disagree with the clip. All read the climb_motion command and use
exp(-error / std**2) kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .motion import ClimbMotionCommand

__all__ = (
  "engage_match_gate",
  "motion_anchor_landing",
  "motion_body_angular_velocity_error_exp",
  "motion_body_linear_velocity_error_exp",
  "motion_body_orientation_error_exp",
  "motion_body_position_error_exp",
  "motion_engage_mismatch",
  "motion_ref_height_tracking",
  "motion_root_ori_error_exp",
  "motion_swing_approach",
)


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


def engage_match_gate(
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
    reward *= engage_match_gate(motion, engage_gate_floor, engage_gate_grace_s, engage_gate_grace_fraction)
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
    reward *= engage_match_gate(motion, engage_gate_floor, engage_gate_grace_s, engage_gate_grace_fraction)
  return reward


def motion_body_linear_velocity_error_exp(env, command_name="climb_motion", std=1.0):
  motion = _motion(env, command_name)
  error = (motion.ref_body_lin_vel_w - motion.robot_body_lin_vel_w).square().sum(-1).mean(-1)
  return torch.exp(-error / std**2)


def motion_body_angular_velocity_error_exp(env, command_name="climb_motion", std=3.14):
  motion = _motion(env, command_name)
  error = (motion.ref_body_ang_vel_w - motion.robot_body_ang_vel_w).square().sum(-1).mean(-1)
  return torch.exp(-error / std**2)


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
  ``engage_gate_floor``: see ``engage_match_gate``.
  """
  from mjlab.utils.lab_api.math import quat_error_magnitude

  motion = _motion(env, command_name)
  error = quat_error_magnitude(motion.ref_root_quat_w, motion.robot_root_quat_w).square()
  reward = torch.exp(-error / std**2)
  if engage_gate_floor is not None:
    reward = reward * engage_match_gate(motion, engage_gate_floor, engage_gate_grace_s, engage_gate_grace_fraction)
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
  return _graced_mismatch(motion, grace_time_s, grace_fraction).sum(dim=-1)


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
