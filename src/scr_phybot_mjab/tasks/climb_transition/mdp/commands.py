"""Transition-side command variants.

``ScheduleAwareClimbVelocityCommand`` serves the unified full-chain task:
the climb-rate obs lane must read an honest zero while the reference is
still in the mount prologue (a mount has no climb rate) and the CYCLE
teacher's sampled rate once the clock rolls past the handoff — feeding the
cycle segment a hard zero would put the frozen cycle teacher's KL answers
off-distribution (it trained reading sampled rates on that lane).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from scr_phybot_mjab.tasks.climb_wall.mdp.commands import (
  ClimbVelocityCommand,
  ClimbVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "ScheduleAwareClimbVelocityCommand",
  "ScheduleAwareClimbVelocityCommandCfg",
)


class ScheduleAwareClimbVelocityCommand(ClimbVelocityCommand):
  """Climb-rate command gated by the unified clock's segment."""

  cfg: ScheduleAwareClimbVelocityCommandCfg

  @property
  def command(self) -> torch.Tensor:
    motion = self._env.command_manager.get_term(self.cfg.motion_command_name)
    out = self.vel_command_w.clone()
    out[:, 2] = out[:, 2] * motion.in_cycle_region.float()
    return out


@dataclass(kw_only=True)
class ScheduleAwareClimbVelocityCommandCfg(ClimbVelocityCommandCfg):
  """Config for :class:`ScheduleAwareClimbVelocityCommand`."""

  motion_command_name: str = "climb_motion"

  def build(self, env: ManagerBasedRlEnv) -> ScheduleAwareClimbVelocityCommand:
    return ScheduleAwareClimbVelocityCommand(self, env)
