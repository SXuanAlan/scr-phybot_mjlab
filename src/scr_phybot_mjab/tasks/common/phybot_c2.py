"""PHYBot C2 sensor names, joint selections, actuator settings and observations shared by the tasks."""

from __future__ import annotations

import copy

from mjlab.entity import EntityCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_FOOT_SITES,
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_REAL_PD_DAMPING_BY_GROUP,
)
from scr_phybot_mjab.tasks.common.observations import projected_gravity_from_framequat_sensor

# Builtin base-link sensors of the scene's "robot" entity.
BASE_QUAT_SENSOR = "robot/baselink-quat"
BASE_LIN_VEL_SENSOR = "robot/baselink-velocity"
BASE_ANG_VEL_SENSOR = "robot/baselink-gyro"

TORSO_BODY = "torso"
FEET_ROOT_BODIES = r"^(left_ankle_roll|right_ankle_roll)$"
"""The two foot links; each sole hangs in its subtree."""
FOOT_SITE_NAMES = tuple(site for _, site in PHYBOT_C2_FOOT_SITES)
"""The sole sites, left then right."""

# Contact sensors of the standing tasks (walk, stand and squat).
FEET_GROUND_CONTACT_SENSOR = "feet_ground_contact"
SELF_COLLISION_SENSOR = "self_collision"
ILLEGAL_GROUND_CONTACT_SENSOR = "illegal_ground_contact"

# Bodies that must never touch the floor (official PhyBot Lab terminates on
# knee/shoulder/elbow/base/waist contact). Only bodies with active collision
# geoms are listed: the waist_yaw and shoulder_pitch collisions are commented
# out in the MJCF and cannot generate contacts.
_ILLEGAL_GROUND_BODIES = (
  r"^(left_knee|right_knee"
  r"|left_shoulder_roll|right_shoulder_roll"
  r"|left_shoulder_yaw|right_shoulder_yaw"
  r"|left_elbow_pitch|right_elbow_pitch"
  r"|torso|torso_upper)$"
)


def policy_joint_cfg() -> SceneEntityCfg:
  """The policy's actuated joints in action order."""
  return SceneEntityCfg("robot", joint_names=PHYBOT_C2_POLICY_JOINT_NAMES, preserve_order=True)


def real_pd_robot(robot_cfg: EntityCfg) -> EntityCfg:
  """A copy of ``robot_cfg`` with the deployed PD table's damping (hip/knee kd 5 -> 10).

  A deep copy: the actuator cfgs are module singletons shared with every
  other task.
  """
  robot_cfg = copy.deepcopy(robot_cfg)
  damping_by_target_expr = {(f".*_{group}",): damping for group, damping in PHYBOT_C2_REAL_PD_DAMPING_BY_GROUP.items()}

  updated = set()
  for actuator in robot_cfg.articulation.actuators:
    target_expr = tuple(actuator.target_names_expr)
    if target_expr in damping_by_target_expr:
      actuator.damping = damping_by_target_expr[target_expr]
      updated.add(target_expr)

  missing = set(damping_by_target_expr) - updated
  if missing:
    raise ValueError(f"Missing PHYBot C2 real-PD actuator damping targets: {sorted(missing)}")

  return robot_cfg


def ground_contact_sensors() -> tuple[ContactSensorCfg, ...]:
  """The standing tasks' contact sensors: feet on the floor, self-collision and forbidden floor contact."""
  return (
    ContactSensorCfg(
      name=FEET_GROUND_CONTACT_SENSOR,
      primary=ContactMatch(mode="subtree", pattern=FEET_ROOT_BODIES, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    ),
    # The self-collision cost counts the substeps of its force history.
    ContactSensorCfg(
      name=SELF_COLLISION_SENSOR,
      primary=ContactMatch(mode="subtree", pattern=TORSO_BODY, entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern=TORSO_BODY, entity="robot"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      history_length=4,
    ),
    ContactSensorCfg(
      name=ILLEGAL_GROUND_CONTACT_SENSOR,
      primary=ContactMatch(mode="body", pattern=_ILLEGAL_GROUND_BODIES, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      history_length=4,
    ),
  )


def standing_actor_terms(command_name: str) -> dict[str, ObservationTermCfg]:
  """The actor of the walk, stand and squat policies: noisy proprioception and the task command.

  It has no base linear velocity: the real robot has no estimate of it.
  Term order and noise are the deployed observation contract.
  """
  return {
    "base_ang_vel": ObservationTermCfg(
      func=base_mdp.builtin_sensor,
      params={"sensor_name": BASE_ANG_VEL_SENSOR},
      noise=Unoise(n_min=-0.15, n_max=0.15),
    ),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity_from_framequat_sensor,
      params={"sensor_name": BASE_QUAT_SENSOR},
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel,
      params={"asset_cfg": policy_joint_cfg()},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel,
      params={"asset_cfg": policy_joint_cfg()},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "actions": ObservationTermCfg(func=base_mdp.last_action),
    "command": ObservationTermCfg(func=base_mdp.generated_commands, params={"command_name": command_name}),
  }
