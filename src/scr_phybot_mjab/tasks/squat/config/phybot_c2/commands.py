"""The squat reference command."""

from __future__ import annotations

from mjlab.managers.command_manager import CommandTermCfg

from scr_phybot_mjab.assets.motions import SQUAT_MOTION
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_POLICY_JOINT_NAMES
from scr_phybot_mjab.tasks.squat.mdp import SquatMotionCommandCfg

from .motion import FOOT_SITE_NAMES

MOTION_COMMAND = "motion"
"""The command's manager key; the observations, rewards and validation scripts read it."""


def squat_commands() -> dict[str, CommandTermCfg]:
  """The clip from its first frame, with small reset perturbations of the standing state.

  A squat policy is selected while the real robot is already standing, not
  teleported to an exact simulator keyframe, so the reset scatters the
  residual settling error of the previous policy. It has no x/y/yaw pose
  offsets: the clip anchors the feet in the world frame, and moving the
  whole robot would command it to slide back to the environment origin.
  """
  return {
    MOTION_COMMAND: SquatMotionCommandCfg(
      entity_name="robot",
      # Never resample mid-episode; every episode starts at motion frame zero.
      resampling_time_range=(1.0e9, 1.0e9),
      motion_files=(str(SQUAT_MOTION),),
      joint_names=tuple(PHYBOT_C2_POLICY_JOINT_NAMES),
      foot_site_names=FOOT_SITE_NAMES,
      sampling_mode="start",
      pose_range={
        "z": (-0.005, 0.005),
        "roll": (-0.025, 0.025),
        "pitch": (-0.025, 0.025),
      },
      velocity_range={
        "x": (-0.075, 0.075),
        "y": (-0.075, 0.075),
        "z": (-0.025, 0.025),
        "roll": (-0.05, 0.05),
        "pitch": (-0.05, 0.05),
        "yaw": (-0.05, 0.05),
      },
      joint_position_range=(-0.025, 0.025),
      joint_velocity_range=(-0.05, 0.05),
    )
  }
