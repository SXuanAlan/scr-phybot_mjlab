"""Episode metrics for the wall-climbing task.

Each function returns a per-env (E,) sample; the MetricsManager accumulates
it every step and logs the episode mean as ``Episode_Metrics/<key>``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def microspine_engagement_occupancy(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
  ee_index: int | None = None,
) -> torch.Tensor:
  """Current latch occupancy, optionally for one EE, shape ``(E,)``."""
  affordance = env.command_manager.get_term(command_name)
  if ee_index is not None:
    return affordance.engaged[:, ee_index].float()
  return affordance.engaged.float().mean(dim=1)


# Compatibility alias for old configs. The value is occupancy, not an event rate.
microspine_engage_rate = microspine_engagement_occupancy


def microspine_forced_release_rate(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Limit-surface release event rate in events/s, shape ``(E,)``."""
  affordance = env.command_manager.get_term(command_name)
  return affordance.forced_release_this_step.float().sum(dim=1) / env.step_dt


def microspine_engage_events(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Debounced OFF -> ON latch rate in events/s, excluding reset anchors."""
  affordance = env.command_manager.get_term(command_name)
  return affordance.engaged_this_step.float().sum(dim=1) / env.step_dt


def microspine_reseat_hold_fraction(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Fraction of steps in the R4 reseat hold (phase clock frozen), (E,)."""
  affordance = env.command_manager.get_term(command_name)
  return affordance.reseat_hold.float()


def microspine_reseat_pending_count(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Mean count of EEs awaiting a dr-forced reseat, (E,)."""
  affordance = env.command_manager.get_term(command_name)
  return affordance.dr_release_pending.float().sum(dim=1)


def microspine_hold_margin_engaged(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Mean hold margin over engaged EEs (0 when none engaged), shape (E,)."""
  affordance = env.command_manager.get_term(command_name)
  engaged = affordance.engaged.float()
  margin_sum = (affordance.hold_margin * engaged).sum(dim=1)
  return margin_sum / engaged.sum(dim=1).clamp(min=1.0)


def action_saturation_fraction(
  env: ManagerBasedRlEnv,
  clip: float = 1.0,
  tolerance: float = 1.0e-6,
) -> torch.Tensor:
  """Fraction of applied policy actions sitting on the action clip boundary."""
  return (env.action_manager.action.abs() >= clip - tolerance).float().mean(dim=1)


def joint_target_clamp_fraction(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
) -> torch.Tensor:
  """Fraction of joint targets caught by the final soft-limit guard."""
  term = env.action_manager.get_term(action_name)
  return term.target_clamped.float().mean(dim=1)


def joint_soft_limit_violation(
  env: ManagerBasedRlEnv,
  asset_name: str = "robot",
) -> torch.Tensor:
  """Fraction of actual joints outside their soft position limits."""
  asset = env.scene[asset_name]
  limits = asset.data.soft_joint_pos_limits
  below = asset.data.joint_pos < limits[..., 0]
  above = asset.data.joint_pos > limits[..., 1]
  return (below | above).float().mean(dim=1)


def motion_height_gained(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
) -> torch.Tensor:
  """Base rise since RSI placement (m); the curriculum's success quantity."""
  motion = env.command_manager.get_term(command_name)
  return motion.height_gained
