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


def gait_productive_regrips(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_gait",
) -> torch.Tensor:
  """Productive commanded re-grip rate in events/s."""
  gait = env.command_manager.get_term(command_name)
  return gait.productive_regrip_this_step.float() / env.step_dt


def gait_productive_cycles(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_gait",
) -> torch.Tensor:
  """Productive four-limb cycle rate in events/s."""
  gait = env.command_manager.get_term(command_name)
  return gait.productive_cycle_this_step.float() / env.step_dt


def gait_phase_occupancy(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_gait",
  phase: int = 0,
) -> torch.Tensor:
  """Fractional occupancy of one gait phase (per-env value is 0 or 1)."""
  gait = env.command_manager.get_term(command_name)
  return (gait.phase == phase).float()


def gait_target_release_rate(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_gait",
) -> torch.Tensor:
  """Commanded target-limb release rate in events/s."""
  gait = env.command_manager.get_term(command_name)
  return gait.target_release_this_step.float() / env.step_dt


def gait_target_contact_rate(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_gait",
) -> torch.Tensor:
  """First wall-contact transition of the commanded swing limb in events/s."""
  gait = env.command_manager.get_term(command_name)
  return gait.target_contact_this_step.float() / env.step_dt


def gait_regrip_rate(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_gait",
) -> torch.Tensor:
  """All confirmed commanded re-grips in events/s, productive or not."""
  gait = env.command_manager.get_term(command_name)
  return gait.regrip_this_step.float() / env.step_dt
