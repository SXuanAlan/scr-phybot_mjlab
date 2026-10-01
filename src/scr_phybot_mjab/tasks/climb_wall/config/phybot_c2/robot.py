"""The PHYBot C2 entity for climbing: bracing toes, the climbing stance and the PD table."""

from __future__ import annotations

import dataclasses

import mujoco
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
  get_phybot_c2_robot_cfg,
  get_spec,
)
from scr_phybot_mjab.microspine.spec import ToeContact, add_toe_geoms

from .stance import climb_initial_state


def climb_robot_cfg(toe_contacts: tuple[ToeContact, ...]) -> EntityCfg:
  robot_cfg = get_phybot_c2_robot_cfg()

  def _robot_spec_with_toes() -> mujoco.MjSpec:
    # The toe geoms live on the robot entity, since sensors resolve geom
    # names against it; their wall pairs are scene-level.
    spec = get_spec()
    add_toe_geoms(spec, toe_contacts, robot_prefix="")
    return spec

  robot_cfg.spec_fn = _robot_spec_with_toes
  # A task-level init state: robot_cfg's keyframe is a singleton other C2
  # tasks share.
  initial = climb_initial_state()
  robot_cfg.init_state = EntityCfg.InitialStateCfg(
    pos=tuple(initial["root_position"]),
    rot=tuple(initial["root_quaternion_wxyz"]),
    joint_pos=initial["joint_positions"],
    joint_vel={".*": 0.0},
  )
  robot_cfg.articulation = dataclasses.replace(
    robot_cfg.articulation,
    # Reserve 1% of the hard range at each end: the reference cycle works
    # 0.06 rad from some stops and tracking must never fight the target
    # clamp; the physical stop remains the final guard.
    soft_joint_pos_limit_factor=0.98,
    actuators=climb_actuators(),
  )
  return robot_cfg


def climb_actuators() -> tuple:
  """Legs, waist and the whole arm chain at 100/10; ankles 50/5; neck at its default.

  kd = 10 lifts the hanging arm chain's damping ratio from 0.35 to 0.5.
  Copies, not edits: the actuator cfgs are module singletons that walk and
  squat share.
  """

  def pd100(actuator):
    return dataclasses.replace(actuator, stiffness=100.0, damping=10.0)

  return (
    pd100(PHYBOT_C2_ACTUATOR_HIP_PITCH),
    pd100(PHYBOT_C2_ACTUATOR_HIP_ROLL),
    pd100(PHYBOT_C2_ACTUATOR_HIP_YAW),
    pd100(PHYBOT_C2_ACTUATOR_KNEE),
    PHYBOT_C2_ACTUATOR_ANKLE_PITCH,
    PHYBOT_C2_ACTUATOR_ANKLE_ROLL,
    pd100(PHYBOT_C2_ACTUATOR_WAIST_YAW),
    pd100(PHYBOT_C2_ACTUATOR_SHOULDER_PITCH),
    pd100(PHYBOT_C2_ACTUATOR_SHOULDER_ROLL),
    pd100(PHYBOT_C2_ACTUATOR_SHOULDER_YAW),
    pd100(PHYBOT_C2_ACTUATOR_ELBOW_PITCH),
    PHYBOT_C2_ACTUATOR_NECK,
  )
