"""Brick commands: the grip affordance and the measured-phase task command."""

from __future__ import annotations

from mjlab.managers.command_manager import CommandTermCfg

from scr_phybot_mjab.assets.brick import BrickCfg
from scr_phybot_mjab.microspine.config import MicrospineAffordanceCfg, MicrospineCfg
from scr_phybot_mjab.tasks.brick.command import BrickTaskCommandCfg


def brick_commands(
  mc: MicrospineCfg, brick: BrickCfg, stage: str, randomize: bool, unit_layout: str
) -> dict[str, CommandTermCfg]:
  return {
    "microspine_affordance": MicrospineAffordanceCfg(
      microspine_cfg=mc, skip_zero_time_update=True, contact_sensor_name="microspine_brick_contact"
    ),
    # The brick pose arrives as a noisy, delayed estimate, with occasional
    # dropouts under randomization.
    "brick_task": BrickTaskCommandCfg(
      brick=brick,
      training_stage=stage,
      randomize=randomize,
      unit_layout=unit_layout,
      pose_position_noise_m=0.003,
      pose_rotation_noise_rad=0.01,
      max_delay_steps=2,
      dropout_probability=0.01 if randomize else 0.0,
      reset_xy_noise_m=0.01 if randomize else 0.0,
      reset_yaw_noise_rad=0.03 if randomize else 0.0,
    ),
  }
