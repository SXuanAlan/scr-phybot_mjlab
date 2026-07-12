"""Episode metrics for the wall-climbing task.

Each function returns a per-env (E,) sample; the MetricsManager accumulates
it every step and logs the episode mean as ``Episode_Metrics/<key>``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def microspine_engage_rate(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Fraction of EEs currently engaged, shape (E,)."""
  affordance = env.command_manager.get_term(command_name)
  return affordance.engaged.float().mean(dim=1)


def microspine_forced_release_rate(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """EEs released by the limit surface this step, shape (E,)."""
  affordance = env.command_manager.get_term(command_name)
  return affordance.forced_release_this_step.float().sum(dim=1)


def microspine_hold_margin_engaged(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Mean hold margin over engaged EEs (0 when none engaged), shape (E,)."""
  affordance = env.command_manager.get_term(command_name)
  engaged = affordance.engaged.float()
  margin_sum = (affordance.hold_margin * engaged).sum(dim=1)
  return margin_sum / engaged.sum(dim=1).clamp(min=1.0)
