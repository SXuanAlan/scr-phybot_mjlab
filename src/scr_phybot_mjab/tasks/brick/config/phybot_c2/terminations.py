"""Brick terminations: time out, failure and success."""

from __future__ import annotations

from mjlab.envs import mdp as base_mdp
from mjlab.managers.termination_manager import TerminationTermCfg

from scr_phybot_mjab.tasks.brick import mdp


def brick_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
    "failure": TerminationTermCfg(func=mdp.failed),
    "success": TerminationTermCfg(func=mdp.finished),
  }
