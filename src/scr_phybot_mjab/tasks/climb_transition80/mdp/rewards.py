"""Transition80 rewards: BeyondMimic's tracking kernels + one engage-progress term.

Ports of mjlab ``tasks/tracking/mdp/rewards.py`` against the
``TransitionMotionCommand`` properties, plus ``engage_progress`` — the single
reward that sees the contact-rich engage chain. Everything that used to be
a wage, bonus, tax, tow, fence or rent in the transition ledger is absent
on purpose: the terminations own legality, these terms own quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

from scr_phybot_mjab.tasks.climb_wall.mdp.rewards_tracking import engage_match_gate

from .observations import unit_preload_fraction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .motion import TransitionMotionCommand

__all__ = (
  "engage_progress",
  "scheduled_anchor_rip_penalty",
  "scheduled_rip_events",
  "scheduled_hold_margin_wage",
  "motion_pad_center_error_exp",
  "motion_anchor_orientation_error_exp",
  "motion_anchor_position_error_exp",
  "motion_body_angular_velocity_error_exp",
  "motion_body_linear_velocity_error_exp",
  "motion_body_orientation_error_exp",
  "motion_body_position_error_exp",
)


def _tracking(env: ManagerBasedRlEnv, command_name: str) -> "TransitionMotionCommand":
  return env.command_manager.get_term(command_name)


def _gated(
  reward: torch.Tensor,
  cmd: "TransitionMotionCommand",
  floor: float | None,
  grace_s: float,
  grace_fraction: float | None,
) -> torch.Tensor:
  """Climb's schedule-adherence discount (v1e, user call 2026-08-21):
  pose income is multiplied by ``engage_match_gate`` in [floor, 1] — full
  pay only while every pad's latch state matches the chart, ``floor`` of it
  when it does not, ramped near scheduled edges (instant compliance is
  impossible). Without it a pad hovering unwelded inside its window collected
  full tracking pay for the whole contact-law grace and the weld's only
  margin lay beyond the 3 s horizon (model_1100 probe: in-window pads within
  10 cm of the anchor only 14-25 % of the time, zero down-shear). Multiplicative
  and bounded below — the ledger stays positive. None disables the gate."""
  if floor is None:
    return reward
  return reward * engage_match_gate(cmd, floor, grace_s, grace_fraction)


def motion_anchor_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.3,
  engage_gate_floor: float | None = None,
  engage_gate_grace_s: float = 0.25,
  engage_gate_grace_fraction: float | None = None,
) -> torch.Tensor:
  """exp(-|p_root - p̂_root|² / std²), world frame."""
  cmd = _tracking(env, command_name)
  error = (cmd.anchor_pos_w - cmd.robot_anchor_pos_w).square().sum(dim=-1)
  reward = torch.exp(-error / std**2)
  return _gated(reward, cmd, engage_gate_floor, engage_gate_grace_s, engage_gate_grace_fraction)


def motion_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.4,
  engage_gate_floor: float | None = None,
  engage_gate_grace_s: float = 0.25,
  engage_gate_grace_fraction: float | None = None,
) -> torch.Tensor:
  """exp(-θ(q_root, q̂_root)² / std²), geodesic angle."""
  cmd = _tracking(env, command_name)
  error = quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w)
  reward = torch.exp(-error.square() / std**2)
  return _gated(reward, cmd, engage_gate_floor, engage_gate_grace_s, engage_gate_grace_fraction)


def motion_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.3,
  engage_gate_floor: float | None = None,
  engage_gate_grace_s: float = 0.25,
  engage_gate_grace_fraction: float | None = None,
) -> torch.Tensor:
  """exp(-mean_B |p_b - p̂_b|² / std²) over the tracked bodies."""
  cmd = _tracking(env, command_name)
  error = (cmd.body_pos_w - cmd.robot_body_pos_w).square().sum(dim=-1).mean(dim=-1)
  return torch.exp(-error / std**2)


def motion_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.4,
  engage_gate_floor: float | None = None,
  engage_gate_grace_s: float = 0.25,
  engage_gate_grace_fraction: float | None = None,
) -> torch.Tensor:
  """exp(-mean_B θ_b² / std²) over the tracked bodies."""
  cmd = _tracking(env, command_name)
  ref = cmd.body_quat_w
  rob = cmd.robot_body_quat_w
  error = quat_error_magnitude(ref.reshape(-1, 4), rob.reshape(-1, 4)).reshape(ref.shape[0], ref.shape[1])
  return torch.exp(-error.square().mean(dim=-1) / std**2)


def motion_pad_center_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.2,
) -> torch.Tensor:
  """exp(-mean_pads |c - ĉ|² / std²) on the pad FACE centers (world).

  Deliberately UNGATED: the face reaching the reference face (= the anchor
  while scheduled) is the path to schedule compliance, so it is the one
  pose income that must not be discounted while unwelded (v1g)."""
  cmd = _tracking(env, command_name)
  error = (cmd.ref_pad_center_w - cmd.robot_pad_center_w).square().sum(dim=-1).mean(dim=-1)
  return torch.exp(-error / std**2)


def motion_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 1.0,
) -> torch.Tensor:
  """exp(-mean_B |v_b - v̂_b|² / std²), world frame."""
  cmd = _tracking(env, command_name)
  error = (cmd.body_lin_vel_w - cmd.robot_body_lin_vel_w).square().sum(dim=-1).mean(dim=-1)
  return torch.exp(-error / std**2)


def motion_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 3.14,
) -> torch.Tensor:
  """exp(-mean_B |ω_b - ω̂_b|² / std²), world frame."""
  cmd = _tracking(env, command_name)
  error = (cmd.body_ang_vel_w - cmd.robot_body_ang_vel_w).square().sum(dim=-1).mean(dim=-1)
  return torch.exp(-error / std**2)


def engage_progress(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  affordance_command_name: str = "microspine_affordance",
  edge_lead_time_s: float = 2.0,
  reach_radius_m: float = 0.25,
  touch_radius_m: float = 0.15,
  anchor_radius_m: float = 0.15,
  reach_share: float = 0.3,
  touch_level: float = 0.5,
  load_share: float = 0.3,
  weld_level: float = 1.0,
  ee_indices: tuple[int, ...] | None = None,
  approach_share: float = 0.0,
  approach_radius_m: float = 0.10,
  press_share: float = 0.0,
  press_ref_n: float = 20.0,
) -> torch.Tensor:
  """Staged engage-progress potential Φ, mean over pads, in [0, 1].

  Per pad, inside its scheduled window (rising edge - ``edge_lead_time_s``
  ... release edge)::

      not touching                 Φ = reach_share·(1 - clip(d / reach_radius))
                                     + approach_share·(1 - clip(gap / approach_radius))
      touching, not engaged, near  Φ = touch_level + load_share·preload
                                     + press_share·clip(press / press_ref)
      touching, not engaged, far   Φ = reach (wrong place)
      engaged within anchor_radius Φ = weld_level
      engaged elsewhere            Φ = reach (wrong place; the contact law kills)

  and outside the window ``Φ = 1[not engaged]``. d is the pad-center to
  scheduled-anchor distance (the contact law's quantity), gap the wall-normal
  clearance of the pad's closest sphere, press the pad normal force and
  preload the engage gate's progress (:func:`unit_preload_fraction`). Keep
  ``reach_share + approach_share < touch_level`` and
  ``touch_level + load_share + press_share <= weld_level`` so Φ rises
  monotonically along the physical chain.
  """
  motion = _tracking(env, command_name)
  affordance = env.command_manager.get_term(affordance_command_name)
  executor = affordance._get_executor()
  dist = (executor.pad_center_world() - motion.ref_anchor_pos_w).norm(dim=-1)
  engaged = affordance.engaged
  touch = affordance.contact_count > 0

  lead = edge_lead_time_s / motion.cycle_time_s
  regrip_soon = motion.ref_edge_countdowns[..., 1] <= lead
  window = motion.ref_engaged | regrip_soon

  reach = reach_share * (1.0 - (dist / reach_radius_m).clamp(max=1.0))
  if approach_share > 0.0:
    gaps = affordance.proxy_wall_gaps().view(dist.shape[0], dist.shape[1], -1)
    gap = gaps.min(dim=-1).values.clamp(min=0.0)  # closest sphere surface -> wall
    reach = reach + approach_share * (1.0 - (gap / approach_radius_m).clamp(max=1.0))
  load = touch_level + load_share * unit_preload_fraction(affordance)
  if press_share > 0.0:
    press = getattr(affordance, "pad_press_force", None)
    if press is not None:
      load = load + press_share * (press.clamp(min=0.0) / max(press_ref_n, 1.0e-6)).clamp(max=1.0)
  weld = torch.full_like(dist, weld_level)
  phi_window = torch.where(
    engaged,
    torch.where(dist <= anchor_radius_m, weld, reach),
    torch.where(touch & (dist <= touch_radius_m), load, reach),
  )
  phi_off = (~engaged).float()
  phi = torch.where(window, phi_window, phi_off)
  if ee_indices is not None:
    phi = phi[:, list(ee_indices)]
  return phi.mean(dim=-1)


def scheduled_hold_margin_wage(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  affordance_command_name: str = "microspine_affordance",
  full_margin: float = 0.3,
) -> torch.Tensor:
  """Dwell wage for SAFELY loaded scheduled welds (v3, 2026-08-22).

  Per pad that the chart says engaged AND that is welded, pay
  ``clip(hold_margin / full_margin, 0, 1)`` — the plant's worst-unit hold
  margin (1 = unloaded, 0 = at capacity, < 0 = violating); averaged over the
  scheduled-engaged pads so the count of welds does not change the scale.
  A positive term gated on being welded: never a reason not to weld, only a
  preference for sharing the load (feet under the hips, soft landings)
  before the rip law has anything to say. Weight stays well under the
  tracking income so it can never become rent (08-13 lesson)."""
  cmd = _tracking(env, command_name)
  aff = env.command_manager.get_term(affordance_command_name)
  scheduled = cmd.ref_engaged
  gate = (scheduled & aff.engaged).to(aff.hold_margin.dtype)
  pay = (aff.hold_margin / max(float(full_margin), 1.0e-6)).clamp(0.0, 1.0)
  n_sched = scheduled.sum(dim=-1).clamp_min(1).to(pay.dtype)
  return (gate * pay).sum(dim=-1) / n_sched


def scheduled_rip_events(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  affordance_command_name: str = "microspine_affordance",
  release_lead_time_s: float = 0.5,
  seed_settle_exempt_s: float = 0.2,
) -> torch.Tensor:
  """Pads ripped this step, (E, H): a forced release while the chart says hold.

  Every hold-judge release counts, whatever its reason. Exempt: releases
  within ``release_lead_time_s`` of the scheduled falling edge (letting go
  slightly early is the choreography), and seeded anchors lost within
  ``seed_settle_exempt_s`` of birth, when the kinematic birth drops the body
  weight onto welds the policy has not loaded yet.
  """
  motion = env.command_manager.get_term(command_name)
  aff = env.command_manager.get_term(affordance_command_name)
  released = aff.forced_release_this_step
  lead = release_lead_time_s / motion.cycle_time_s
  release_soon = motion.ref_edge_countdowns[..., 0] <= lead
  rip = released & motion.ref_engaged & ~release_soon
  if seed_settle_exempt_s > 0.0:
    settle_steps = max(1, int(round(seed_settle_exempt_s / env.step_dt)))
    settling = (env.episode_length_buf <= settle_steps).unsqueeze(-1)
    was_seeded = getattr(aff, "released_was_seeded", None)
    if was_seeded is not None:
      rip = rip & ~(was_seeded & settling)
  return rip


def scheduled_anchor_rip_penalty():
  """Factory: per-step bill while a RIPPED scheduled anchor stays un-welded.

  v5 (2026-08-23, user decision): climb's rule — a rip is a penalty, not a
  death. The flag is raised on the rip event exactly as the retired
  ``anchor_rip`` termination defined it (a forced limit-surface release while
  the chart says hold, outside the release lead, not an R4 reseat, and not a
  SEEDED anchor inside the birth-settle exemption) and cleared when the pad
  is welded again or the chart releases it. Value = ``weight_per_pad`` x the
  number of flagged pads (the cfg weight is -1); ``weight_per_pad`` climbs
  on a curriculum. Kept under the tracking income per pad so a rip is never
  worth dying over (08-13 lesson: a bill above the income makes the early
  death the optimum). The contact law (LadderMan grace) stays as the
  backstop for a pad that is never re-seated.
  """
  state: dict[str, torch.Tensor | None] = {"flag": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    affordance_command_name: str = "microspine_affordance",
    weight_per_pad: float = 1.0,
    release_lead_time_s: float = 0.5,
    seed_settle_exempt_s: float = 0.2,
  ) -> torch.Tensor:
    motion = env.command_manager.get_term(command_name)
    aff = env.command_manager.get_term(affordance_command_name)
    rip = scheduled_rip_events(
      env,
      command_name=command_name,
      affordance_command_name=affordance_command_name,
      release_lead_time_s=release_lead_time_s,
      seed_settle_exempt_s=seed_settle_exempt_s,
    )
    flag = state["flag"]
    if flag is None or flag.shape != rip.shape:
      flag = torch.zeros_like(rip)
    fresh = (env.episode_length_buf <= 1).unsqueeze(-1)
    flag = (flag & ~fresh) | rip
    flag = flag & motion.ref_engaged & ~aff.engaged
    state["flag"] = flag
    call.last_rips = rip  # type: ignore[attr-defined]  # metrics read the event
    call.last_flags = flag  # type: ignore[attr-defined]
    return float(weight_per_pad) * flag.sum(dim=-1).to(torch.float32)

  call.last_rips = None  # type: ignore[attr-defined]
  call.last_flags = None  # type: ignore[attr-defined]
  return call
