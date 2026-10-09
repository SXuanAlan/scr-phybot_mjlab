"""The transition termination table: tracking deviation, the contact law, the wall fence and the engage cone."""

from __future__ import annotations

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from scr_phybot_mjab.tasks.climb_transition80 import mdp
from scr_phybot_mjab.tasks.climb_wall import mdp as climb_mdp
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.commands import MICROSPINE_AFFORDANCE_COMMAND, MOTION_COMMAND

from .schedules import (
  ANCHOR_POS_STAGES,
  ANCHOR_RADIUS_STAGES,
  CONTACT_GRACE_STAGES,
  FORCE_STAGES,
  GRIND_STAGES,
  PAD_POS_STAGES,
  SEED_SETTLE_EXEMPT_S,
  SLAM_STAGES,
  SOFT_P_STAGES,
  TILT_STAGES,
  stage0,
)

FENCE_GEOMS = (
  "neck_pitch_collision",
  "torso_collision",
  "base_link_collision",
  "left_hip_roll_collision",
  "right_hip_roll_collision",
  "left_hip_yaw_collision",
  "right_hip_yaw_collision",
  "left_knee_collision",
  "right_knee_collision",
)
"""Links that never legitimately reach the wall: their collision surfaces touching it end the episode."""


def transition80_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_height": TerminationTermCfg(
      func=climb_mdp.root_below_wall_altitude,
      params={"minimum_altitude": 0.3, "motion_command_name": MOTION_COMMAND},
    ),
    "bad_state": TerminationTermCfg(
      func=climb_mdp.bad_state,
      params={"asset_cfg": SceneEntityCfg("robot"), "max_velocity": 20.0},
    ),
    # Tracking deviation, persisting 10 steps (0.2 s).
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_termination(),
      params={"command_name": MOTION_COMMAND, "threshold": stage0(ANCHOR_POS_STAGES), "persist_steps": 10},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori_termination(),
      params={"command_name": MOTION_COMMAND, "threshold": 0.8, "persist_steps": 10},
    ),
    # The pad face centers; the key keeps the logged lanes comparable across runs.
    "body_pos": TerminationTermCfg(
      func=mdp.bad_pad_center_termination(),
      params={"command_name": MOTION_COMMAND, "threshold": stage0(PAD_POS_STAGES), "persist_steps": 10},
    ),
    "contact_loss": TerminationTermCfg(
      func=mdp.scheduled_contact_loss_termination(),
      params={
        "command_name": MOTION_COMMAND,
        "affordance_command_name": MICROSPINE_AFFORDANCE_COMMAND,
        "grace_s": stage0(CONTACT_GRACE_STAGES),
        "anchor_radius_m": stage0(ANCHOR_RADIUS_STAGES),
      },
    ),
    "wall_contact": TerminationTermCfg(
      func=mdp.illegal_wall_contact_termination(),
      params={"command_name": MOTION_COMMAND, "geom_names": FENCE_GEOMS, "threshold_m": 0.0, "gate": "none"},
    ),
    # The engage cone as Constraints-as-Terminations: a sustained violation
    # ends the episode with the stage's probability.
    "engage_constraint": TerminationTermCfg(
      func=mdp.engage_constraint_termination(),
      params={
        "command_name": MOTION_COMMAND,
        "affordance_command_name": MICROSPINE_AFFORDANCE_COMMAND,
        "mode": "cat",
        "tilt_limit_deg": stage0(TILT_STAGES),
        "grind_limit_mps": stage0(GRIND_STAGES),
        "slam_limit_mps": stage0(SLAM_STAGES),
        "force_limit_n": stage0(FORCE_STAGES),
        "touch_force_n": 3.0,
        "edge_lead_time_s": 2.0,
        "p_tilt": stage0(SOFT_P_STAGES),
        "p_vel": stage0(SOFT_P_STAGES),
        "p_force": stage0(SOFT_P_STAGES),
        "p_unscheduled": 1.0,
        "ee_indices": (0, 1, 2, 3),
        # 0.1 s, the weld engine's own debounce: noise spikes do not count,
        # behavior does.
        "persist_steps": 5,
        "seed_settle_exempt_s": SEED_SETTLE_EXEMPT_S,
      },
    ),
  }
