"""The wall-angle curriculum: the wall tilts about its base line under real gravity."""

from __future__ import annotations

from mjlab.managers.curriculum_manager import CurriculumTermCfg

from scr_phybot_mjab.tasks.climb_wall import mdp

STEPS_PER_ITER = 96
"""Policy steps per PPO iteration; curricula convert iterations with it.

One rollout must hold a full release -> cooldown -> re-engage -> hold
sequence (~0.7 s). Equals the phybot_c2 runner's num_steps_per_env
(test_climb_task_config cross-checks).
"""


def climb_curriculum(angle_stages=mdp.WALL_ANGLE_STAGES) -> dict[str, CurriculumTermCfg]:
  """The wall-angle ladder, realized by tilting the wall about its base line
  under real gravity (``mdp.wall_angle``). Play pins it to the final stage."""
  return {
    "wall_angle": CurriculumTermCfg(
      func=mdp.wall_angle,
      params={
        "angle_stages": [dict(stage) for stage in angle_stages],
        "steps_per_iter": STEPS_PER_ITER,
        "angle_jitter_deg": 5.0,
      },
    ),
  }
