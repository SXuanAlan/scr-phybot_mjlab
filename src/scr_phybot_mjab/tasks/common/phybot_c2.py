"""PHYBot C2 sensor names, joint selections and actuator settings shared by the tasks."""

from __future__ import annotations

import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_REAL_PD_DAMPING_BY_GROUP,
)

# Builtin base-link sensors of the scene's "robot" entity.
BASE_QUAT_SENSOR = "robot/baselink-quat"
BASE_LIN_VEL_SENSOR = "robot/baselink-velocity"
BASE_ANG_VEL_SENSOR = "robot/baselink-gyro"


def policy_joint_cfg() -> SceneEntityCfg:
  """The policy's actuated joints in action order."""
  return SceneEntityCfg("robot", joint_names=PHYBOT_C2_POLICY_JOINT_NAMES, preserve_order=True)


def apply_real_pd_damping(cfg: ManagerBasedRlEnvCfg) -> None:
  """Set actuator damping to the deployed PD table (hip/knee kd 5 -> 10).

  The robot cfg is deep-copied first: actuator cfgs are module singletons
  shared with every other task.
  """
  robot_cfg = copy.deepcopy(cfg.scene.entities["robot"])
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

  cfg.scene.entities["robot"] = robot_cfg
