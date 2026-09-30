"""Transition80 metrics: handoff readiness + the engage cone's lanes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "scheduled_anchor_rips",
  "scheduled_anchor_ripped_pads",
  "scheduled_hold_margin",
  "engage_constraint_delta",
  "engage_constraint_violation",
  "mount_handoff_ready",
)


def mount_handoff_ready(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  affordance_command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """1.0 while the terminal-hold clock sits on its last playable frame (the
  all-four handoff stance) with all four pads actually latched, (E,)."""
  motion = env.command_manager.get_term(command_name)
  affordance = env.command_manager.get_term(affordance_command_name)
  return (motion.at_terminal_hold & affordance.engaged.all(dim=1)).float()


def _constraint_term(env: ManagerBasedRlEnv, termination_name: str):
  try:
    return env.termination_manager.get_term_cfg(termination_name).func
  except (ValueError, AttributeError, KeyError):
    return None


def engage_constraint_violation(
  env: ManagerBasedRlEnv,
  termination_name: str = "engage_constraint",
  key: str = "tilt",
) -> torch.Tensor:
  """(E,) 1.0 where ANY pad violates constraint ``key`` this step (reads the
  violation tensors the cone termination caches on itself)."""
  term = _constraint_term(env, termination_name)
  violations = getattr(term, "last_violations", None) if term else None
  if not violations or key not in violations:
    return torch.zeros(env.num_envs, device=env.device)
  return (violations[key] > 0.0).any(dim=-1).float()


def engage_constraint_delta(
  env: ManagerBasedRlEnv,
  termination_name: str = "engage_constraint",
) -> torch.Tensor:
  """(E,) the CaT termination probability δ the cone computed this step."""
  term = _constraint_term(env, termination_name)
  delta = getattr(term, "last_delta", None) if term else None
  if delta is None:
    return torch.zeros(env.num_envs, device=env.device)
  return delta


def scheduled_hold_margin(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  affordance_command_name: str = "microspine_affordance",
  ee_index: int | None = None,
) -> torch.Tensor:
  """Mean worst-unit hold margin over welded scheduled pads (or one pad), 1.0
  when none is welded — the wage's own quantity, for the digest."""
  cmd = env.command_manager.get_term(command_name)
  aff = env.command_manager.get_term(affordance_command_name)
  gate = cmd.ref_engaged & aff.engaged
  margin = aff.hold_margin
  if ee_index is not None:
    gate = gate[:, ee_index : ee_index + 1]
    margin = margin[:, ee_index : ee_index + 1]
  g = gate.to(margin.dtype)
  return (margin * g).sum(dim=-1) / g.sum(dim=-1).clamp_min(1.0) + (g.sum(dim=-1) == 0).to(margin.dtype)


def _rip_term(env: ManagerBasedRlEnv, reward_name: str):
  try:
    return env.reward_manager.get_term_cfg(reward_name).func
  except (ValueError, AttributeError, KeyError):
    return None


def scheduled_anchor_rips(env: ManagerBasedRlEnv, reward_name: str = "anchor_rip_penalty") -> torch.Tensor:
  """(E,) number of scheduled-anchor RIP events this step (the v5 rip bill's
  event lane; its episode mean x steps = rips per episode)."""
  term = _rip_term(env, reward_name)
  rips = getattr(term, "last_rips", None) if term else None
  if rips is None:
    return torch.zeros(env.num_envs, device=env.device)
  return rips.sum(dim=-1).float()


def scheduled_anchor_ripped_pads(env: ManagerBasedRlEnv, reward_name: str = "anchor_rip_penalty") -> torch.Tensor:
  """(E,) pads currently flagged ripped-and-not-re-welded (the bill's base)."""
  term = _rip_term(env, reward_name)
  flags = getattr(term, "last_flags", None) if term else None
  if flags is None:
    return torch.zeros(env.num_envs, device=env.device)
  return flags.sum(dim=-1).float()
