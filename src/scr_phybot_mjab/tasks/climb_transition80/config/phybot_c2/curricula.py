"""Transition curricula: the wall pinned at 80 degrees, the anti-rake ladder and the threshold anneals.

The pad gimbal travel follows its schedule through startup and reset events,
since it writes per-world joint ranges.
"""

from __future__ import annotations

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from scr_phybot_mjab.tasks.climb_transition80 import mdp
from scr_phybot_mjab.tasks.climb_wall import mdp as climb_mdp
from scr_phybot_mjab.tasks.climb_wall.climb_wall_env_cfg import MICROSPINE_AFFORDANCE_COMMAND, STEPS_PER_ITER

from .schedules import (
  ANCHOR_POS_STAGES,
  ANCHOR_RADIUS_STAGES,
  ANTIRAKE_LADDER,
  CONTACT_GRACE_STAGES,
  FORCE_STAGES,
  GIMBAL_TRAVEL_STAGES,
  GRIND_STAGES,
  PAD_POS_STAGES,
  RIP_PENALTY_STAGES,
  SLAM_STAGES,
  SOFT_P_STAGES,
  TILT_STAGES,
)

GIMBAL_TRAVEL_EVENTS = (("pad_gimbal_travel_init", "startup"), ("pad_gimbal_travel", "reset"))


def _anneal(termination_name: str, param_name: str, stages: list[dict]) -> CurriculumTermCfg:
  return CurriculumTermCfg(
    func=mdp.termination_param_schedule,
    params={
      "termination_name": termination_name,
      "param_name": param_name,
      "stages": [dict(s) for s in stages],
      "steps_per_iter": STEPS_PER_ITER,
    },
  )


def transition80_curriculum() -> dict[str, CurriculumTermCfg]:
  return {
    # The mount clip's native angle, without jitter.
    "wall_angle": CurriculumTermCfg(
      func=climb_mdp.wall_angle,
      params={
        "angle_stages": [{"iter": 0, "angle_deg": 80.0}],
        "steps_per_iter": STEPS_PER_ITER,
        "angle_jitter_deg": 0.0,
      },
    ),
    "engage_antirake_ladder": CurriculumTermCfg(
      func=mdp.microspine_param_schedule,
      params={
        "param_name": "engage_max_antirake_vel",
        "stages": [dict(s) for s in ANTIRAKE_LADDER],
        "command_name": MICROSPINE_AFFORDANCE_COMMAND,
        "steps_per_iter": STEPS_PER_ITER,
      },
    ),
    "contact_law_grace": _anneal("contact_loss", "grace_s", CONTACT_GRACE_STAGES),
    "contact_law_radius": _anneal("contact_loss", "anchor_radius_m", ANCHOR_RADIUS_STAGES),
    "rip_bill": CurriculumTermCfg(
      func=mdp.reward_param_schedule,
      params={
        "reward_name": "anchor_rip_penalty",
        "param_name": "weight_per_pad",
        "stages": [dict(s) for s in RIP_PENALTY_STAGES],
        "steps_per_iter": STEPS_PER_ITER,
        "interpolate": True,
      },
    ),
    # In lockstep with the contact law's radius.
    "engage_progress_radius": CurriculumTermCfg(
      func=mdp.reward_param_schedule,
      params={
        "reward_name": "engage_progress",
        "param_name": "anchor_radius_m",
        "stages": [dict(s) for s in ANCHOR_RADIUS_STAGES],
        "steps_per_iter": STEPS_PER_ITER,
      },
    ),
    "anchor_pos_anneal": _anneal("anchor_pos", "threshold", ANCHOR_POS_STAGES),
    "body_pos_anneal": _anneal("body_pos", "threshold", PAD_POS_STAGES),
    "constraint_tilt": _anneal("engage_constraint", "tilt_limit_deg", TILT_STAGES),
    "constraint_grind": _anneal("engage_constraint", "grind_limit_mps", GRIND_STAGES),
    "constraint_slam": _anneal("engage_constraint", "slam_limit_mps", SLAM_STAGES),
    "constraint_force": _anneal("engage_constraint", "force_limit_n", FORCE_STAGES),
    "constraint_p_tilt": _anneal("engage_constraint", "p_tilt", SOFT_P_STAGES),
    "constraint_p_vel": _anneal("engage_constraint", "p_vel", SOFT_P_STAGES),
    "constraint_p_force": _anneal("engage_constraint", "p_force", SOFT_P_STAGES),
  }


def gimbal_travel_events() -> dict[str, EventTermCfg]:
  """Write the scheduled gimbal stop travel into every world's joint ranges, at startup and on reset."""
  stages = [dict(s) for s in GIMBAL_TRAVEL_STAGES]
  return {
    name: EventTermCfg(
      mode=mode,
      func=mdp.pad_gimbal_travel_schedule,
      params={
        "stages": stages,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_gimbal_[xyz]",)),
        "steps_per_iter": STEPS_PER_ITER,
      },
    )
    for name, mode in GIMBAL_TRAVEL_EVENTS
  }
