"""Episode metrics for the walk->climb transition task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def mount_handoff_ready(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  affordance_command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Handoff readiness, shape (E,).

  1.0 while the terminal-hold clock sits on its last playable frame (the
  all-four handoff stance) with all four pads actually latched — the state
  the climb policy takes over from. Averaged over an episode this reads as
  the fraction of time spent handoff-ready.
  """
  motion = env.command_manager.get_term(command_name)
  affordance = env.command_manager.get_term(affordance_command_name)
  at_end = motion.time_steps >= (motion.clip.cycle_frames - 1)
  return (at_end & affordance.engaged.all(dim=1)).float()
