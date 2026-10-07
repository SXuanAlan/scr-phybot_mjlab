"""Climbing commands: the commanded climb rate, the grip affordance and the reference motion."""

from __future__ import annotations

from pathlib import Path

from mjlab.managers.command_manager import CommandTermCfg

from scr_phybot_mjab.microspine.config import MicrospineAffordanceCfg, MicrospineCfg
from scr_phybot_mjab.tasks.climb_wall import mdp
from scr_phybot_mjab.tasks.climb_wall.climb_wall_env_cfg import (
  MICROSPINE_AFFORDANCE_COMMAND,
  MICROSPINE_CONTACT_SENSOR,
  MICROSPINE_GRIP_EVENT,
  MOTION_COMMAND,
  VELOCITY_COMMAND,
)

TRACKING_BODY_NAMES = (
  "torso",
  "torso_upper",
  "neck_pitch",
  "left_shoulder_roll",
  "left_elbow_pitch",
  "left_hand_spine",
  "right_shoulder_roll",
  "right_elbow_pitch",
  "right_hand_spine",
  "left_hip_yaw",
  "left_knee",
  "left_ankle_roll",
  "left_foot_spine",
  "right_hip_yaw",
  "right_knee",
  "right_ankle_roll",
  "right_foot_spine",
)
"""Bodies the BeyondMimic tracking terms compare against the reference."""


def affordance_command(microspine_cfg: MicrospineCfg) -> MicrospineAffordanceCfg:
  """The grip affordance: the per-pad latch state machine."""
  return MicrospineAffordanceCfg(
    microspine_cfg=microspine_cfg,
    contact_sensor_name=MICROSPINE_CONTACT_SENSOR,
    physics_event_name=MICROSPINE_GRIP_EVENT,
    skip_zero_time_update=True,
  )


def climb_commands(microspine_cfg: MicrospineCfg, motion_file: Path) -> dict[str, CommandTermCfg]:
  # Command exactly the average climb rate one reference cycle delivers, so
  # the height-progress normalization and the commanded-velocity observation
  # agree with the motion being imitated.
  clip_vz = mdp.clip_average_climb_rate(motion_file)
  return {
    VELOCITY_COMMAND: mdp.ClimbVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 8.0),
      ranges=mdp.ClimbVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.0),
        lin_vel_y=(0.0, 0.0),
        lin_vel_z=(clip_vz, clip_vz),
      ),
    ),
    MICROSPINE_AFFORDANCE_COMMAND: affordance_command(microspine_cfg),
    # After the affordance on purpose: its reset (RSI) seeds the per-env
    # latch state and must run after the affordance's own reset (the command
    # asserts this order).
    MOTION_COMMAND: mdp.ClimbMotionCommandCfg(
      entity_name="robot",
      motion_file=str(motion_file),
      affordance_command_name=MICROSPINE_AFFORDANCE_COMMAND,
      ee_body_names=microspine_cfg.ee_body_names,
      # Failure-adaptive RSI over the cycle phase. A freshly reset env holds
      # its reference frame for one step, and its RSI-seeded latches get
      # their anchors immediately.
      sampling_mode="adaptive",
      synchronize_reset_phase=True,
      initialize_seeded_anchors=True,
      # Along-wall spawn height below the reference (on top of the vertical
      # height compensation); at -0.30 the toes stay ~0.2 m above the base
      # line. Decouples absolute altitude from cycle phase.
      height_offset_range=(-0.30, 0.0),
      tracking_body_names=TRACKING_BODY_NAMES,
    ),
  }
