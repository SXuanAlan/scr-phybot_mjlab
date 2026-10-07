"""Climb contact rewards: microspine support, release loads and hold failures.

Two, three and four anchors are all acceptable; only too little support is
penalized. Releases are priced by the unit loads the hold judge captured,
and hold failures carry one-shot costs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from scr_phybot_mjab.microspine.release_reason import ReleaseReason

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "microspine_release_load_reward",
  "microspine_support_loss",
  "undesired_body_contacts",
  "unsafe_microspine_release_penalty",
)


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
    motion = env.command_manager.get_term(motion_command_name)
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
    released = affordance.forced_release_this_step & (reason > 0)
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
    # Only a normal-load release can be a controlled one; domain and early
    # failures are not.
    controlled_reason = reason == ReleaseReason.NORMAL
    pay = released & controlled_reason & finite & window & ~paid & support_ok.unsqueeze(-1)
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
  tension_release_cost: float = 0.25,
  ultimate_release_cost: float = 1.0,
  support_loss_cost: float = 1.0,
) -> torch.Tensor:
  """One-shot cost of each hold failure, plus a surcharge for losing support.

  A unit that exceeds its normal-load draw or fails early in loading costs
  ``tension_release_cost``; a load outside the measured shear/lateral domain
  costs ``ultimate_release_cost``. A release that leaves fewer than
  ``minimum_supports`` latches adds ``support_loss_cost``. Dividing by
  ``step_dt`` makes each configured cost a one-shot event cost after the
  reward manager's dt scaling.
  """
  affordance = env.command_manager.get_term(command_name)
  released = affordance.forced_release_this_step
  reason = affordance.release_reason
  cost = torch.zeros_like(reason, dtype=torch.float32)
  reason_costs = (
    (ReleaseReason.NORMAL, tension_release_cost),
    (ReleaseReason.EMPIRICAL_DOMAIN, ultimate_release_cost),
    (ReleaseReason.EMPIRICAL_EARLY, tension_release_cost),
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
