"""The PHYBot C2 entity of the transition task: the real robot's PD table and a standing start."""

from __future__ import annotations

import dataclasses

from mjlab.entity import EntityCfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_ACTUATOR_ANKLE_PITCH,
  PHYBOT_C2_ACTUATOR_ANKLE_ROLL,
  PHYBOT_C2_ACTUATOR_ELBOW_PITCH,
  PHYBOT_C2_ACTUATOR_HIP_PITCH,
  PHYBOT_C2_ACTUATOR_HIP_ROLL,
  PHYBOT_C2_ACTUATOR_HIP_YAW,
  PHYBOT_C2_ACTUATOR_KNEE,
  PHYBOT_C2_ACTUATOR_NECK,
  PHYBOT_C2_ACTUATOR_SHOULDER_PITCH,
  PHYBOT_C2_ACTUATOR_SHOULDER_ROLL,
  PHYBOT_C2_ACTUATOR_SHOULDER_YAW,
  PHYBOT_C2_ACTUATOR_WAIST_YAW,
  PHYBOT_C2_STANDING_JOINT_POS,
)

# The motor firmware's (stiffness, damping) table, the one the walk-AMP and
# squat tasks deploy with. These arm gains hold a hang with 5 mm of sag;
# climb's 100/10 arms turn exploration noise into ~1 m/s pad grinds and
# tens-of-newton slams that the engage cone then ends.
LEG_PD = (100.0, 10.0)
ANKLE_PD = (50.0, 5.0)
WAIST_PD = (20.0, 10.0)
ARM_PD = (50.0, 5.0)


def transition80_actuators() -> tuple:
  """The firmware table in asset order.

  Copies, not edits: the actuator cfgs are module singletons that walk and
  squat share.
  """

  def pd(actuator, gains):
    return dataclasses.replace(actuator, stiffness=gains[0], damping=gains[1])

  return (
    pd(PHYBOT_C2_ACTUATOR_HIP_PITCH, LEG_PD),
    pd(PHYBOT_C2_ACTUATOR_HIP_ROLL, LEG_PD),
    pd(PHYBOT_C2_ACTUATOR_HIP_YAW, LEG_PD),
    pd(PHYBOT_C2_ACTUATOR_KNEE, LEG_PD),
    pd(PHYBOT_C2_ACTUATOR_ANKLE_PITCH, ANKLE_PD),
    pd(PHYBOT_C2_ACTUATOR_ANKLE_ROLL, ANKLE_PD),
    pd(PHYBOT_C2_ACTUATOR_WAIST_YAW, WAIST_PD),
    pd(PHYBOT_C2_ACTUATOR_SHOULDER_PITCH, ARM_PD),
    pd(PHYBOT_C2_ACTUATOR_SHOULDER_ROLL, ARM_PD),
    pd(PHYBOT_C2_ACTUATOR_SHOULDER_YAW, ARM_PD),
    pd(PHYBOT_C2_ACTUATOR_ELBOW_PITCH, ARM_PD),
    PHYBOT_C2_ACTUATOR_NECK,  # not a policy joint; the asset's 20/2
  )


def standing_init_state() -> EntityCfg.InitialStateCfg:
  """Standing 0.19 m from the wall's base line; the RSI overwrites it on every reset."""
  return EntityCfg.InitialStateCfg(
    pos=(1.31, 0.0, 0.726),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos=dict(PHYBOT_C2_STANDING_JOINT_POS),
    joint_vel={".*": 0.0},
  )
