"""Training curricula: the command range for walk, foot-locking weights for stand and amp.

Stages count environment steps; a PPO iteration is 24 of them.
"""

from __future__ import annotations

from mjlab.managers.curriculum_manager import CurriculumTermCfg

from scr_phybot_mjab.tasks.walk import mdp

from .commands import TWIST_COMMAND
from .variants import WalkVariant

_STEPS_PER_ITER = 24


def walk_curriculum(variant: WalkVariant) -> dict[str, CurriculumTermCfg]:
  if variant == "walk":
    return {
      "command_vel": CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
          "command_name": TWIST_COMMAND,
          "velocity_stages": [
            {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
            {"step": 5000 * _STEPS_PER_ITER, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
            {"step": 10000 * _STEPS_PER_ITER, "lin_vel_x": (-2.0, 3.0)},
          ],
        },
      ),
    }
  if variant == "stand":
    return {
      "stand_feet_lock_weight": _reward_weight("stand_feet_lock", {0: -0.25, 1000: -0.50, 2500: -0.75}),
    }
  return {
    "stand_feet_stationary_weight": _reward_weight(
      "stand_feet_stationary", {0: -0.5, 2000: -1.5, 6000: -3.0, 9000: -4.0}
    ),
    "stand_still_weight": _reward_weight("stand_still", {0: -0.05, 4000: -0.15, 8000: -0.25}),
  }


def _reward_weight(reward_name: str, weight_by_iter: dict[int, float]) -> CurriculumTermCfg:
  return CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": reward_name,
      "weight_stages": [
        {"step": iteration * _STEPS_PER_ITER, "weight": weight} for iteration, weight in weight_by_iter.items()
      ],
    },
  )
