"""The velocity command, and the AMP variant's reset into the AMP clips."""

from __future__ import annotations

import math

from mjlab.managers.command_manager import CommandTermCfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import DEFAULT_KEYFRAME, PHYBOT_C2_POLICY_JOINT_NAMES
from scr_phybot_mjab.tasks.walk.mdp import (
  AmpReferenceResetCommandCfg,
  PresetVelocityCommandCfg,
  UniformVelocityCommandCfg,
)

from .amp_rl_cfg import AmpDatasetCfg
from .variants import WalkVariant

TWIST_COMMAND = "twist"
"""The velocity command's manager key; observations, rewards, curricula and scripts read it."""


def walk_commands(variant: WalkVariant) -> dict[str, CommandTermCfg]:
  if variant == "amp":
    return {TWIST_COMMAND: _preset_twist(), "amp_reset": _amp_reset()}
  if variant == "stand":
    # Every env holds the zero command: the task is purely to stand.
    ranges = UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=None
    )
  else:
    ranges = UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi)
    )
  return {
    TWIST_COMMAND: UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=1.0 if variant == "stand" else 0.25,
      rel_heading_envs=0.3,
      heading_command=variant != "stand",
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=ranges,
      viz=UniformVelocityCommandCfg.VizCfg(z_offset=0.8),
    )
  }


def _preset_twist() -> PresetVelocityCommandCfg:
  """Gait presets in proportion: stand, slow and normal forward walking, and turns."""
  return PresetVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(4.0, 8.0),
    rel_standing_envs=0.0,
    rel_heading_envs=0.0,
    heading_command=False,
    debug_vis=True,
    ranges=PresetVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 0.6),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(-0.35, 0.35),
      heading=None,
    ),
    presets=(
      PresetVelocityCommandCfg.Preset("stand", 0.0, 0.0, 0.0, 45.0),
      PresetVelocityCommandCfg.Preset("slow_forward", 0.35, 0.0, 0.0, 25.0),
      PresetVelocityCommandCfg.Preset("walk_forward", 0.60, 0.0, 0.0, 20.0),
      PresetVelocityCommandCfg.Preset("turn_left", 0.20, 0.0, 0.35, 5.0),
      PresetVelocityCommandCfg.Preset("turn_right", 0.20, 0.0, -0.35, 5.0),
    ),
    slow_speed=0.35,
    walk_speed=0.60,
    turn_forward_speed=0.20,
    turn_speed=0.35,
  )


def _amp_reset() -> AmpReferenceResetCommandCfg:
  """Reference state initialization from the discriminator's clips.

  Resets go into valid reference poses (standing, mid-stride, turning) in
  place of random resets. AMP's reward is distribution-based, so this aids
  coverage and is not required.
  """
  dataset = AmpDatasetCfg()
  return AmpReferenceResetCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),  # On reset only, never mid-episode.
    amp_data_path=dataset.amp_data_path,
    datasets=dataset.datasets,
    joint_names=tuple(PHYBOT_C2_POLICY_JOINT_NAMES),
    slow_down_factor=dataset.slow_down_factor,
    velocity_representation=dataset.velocity_representation,
    reset_height=DEFAULT_KEYFRAME.pos[2],
  )
