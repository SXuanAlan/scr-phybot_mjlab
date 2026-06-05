"""Forearm + hand robot constants and configuration.

Minimal model: right arm (4 active DoFs) + hand for microspine adhesion testing.
Arm is fixed-base at the shoulder_pitch body.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg


##
# MJCF and assets.
##

FOREARM_HAND_DIR: Path = Path(__file__).resolve().parent
FOREARM_HAND_XML: Path = FOREARM_HAND_DIR / "xml" / "forearm_hand.xml"
assert FOREARM_HAND_XML.exists(), f"MJCF not found: {FOREARM_HAND_XML}"

FOREARM_HAND_MESHES: Path = FOREARM_HAND_DIR / "meshes"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, FOREARM_HAND_MESHES, meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(FOREARM_HAND_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

FOREARM_HAND_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("right_shoulder_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=200.0,
)

FOREARM_HAND_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("right_shoulder_roll",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=135.0,
)

FOREARM_HAND_ACTUATOR_ELBOW_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("right_elbow_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=89.0,
)

FOREARM_HAND_ACTUATOR_ELBOW_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("right_elbow_yaw",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=58.0,
)

##
# Keyframe config.
##

FOREARM_HAND_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "base_x": 0.0,
    "right_shoulder_pitch": 0.0,
    "right_shoulder_roll": 0.0,
    "right_elbow_pitch": 0.0,
    "right_elbow_yaw": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FOREARM_HAND_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision.*", "right_hand_c_.*"),
  condim=6,
  friction=(30.0, 0.01, 0.01),
  solref=(0.02, 1.5),
  solimp=(0.9, 0.99, 0.001, 0.5, 2),
  priority={"right_hand_c_.*": 1},
)

##
# Final config.
##

FOREARM_HAND_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    FOREARM_HAND_ACTUATOR_SHOULDER_PITCH,
    FOREARM_HAND_ACTUATOR_SHOULDER_ROLL,
    FOREARM_HAND_ACTUATOR_ELBOW_PITCH,
    FOREARM_HAND_ACTUATOR_ELBOW_YAW,
  ),
  # 1.0 = no soft clamp. right_elbow_pitch range [-2.36, 0] makes the default
  # pose (all joints at 0) sit at the edge of the range; with factor < 1.0
  # mjlab would push the keyframe into the interior and break the geometry.
  soft_joint_pos_limit_factor=1.0,
)


def get_forearm_hand_robot_cfg() -> EntityCfg:
  """Get a fresh forearm+hand robot configuration instance."""
  return EntityCfg(
    init_state=FOREARM_HAND_KEYFRAME,
    collisions=(FOREARM_HAND_COLLISION,),
    spec_fn=get_spec,
    articulation=FOREARM_HAND_ARTICULATION,
  )


FOREARM_HAND_ACTION_SCALE = 0.25


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_forearm_hand_robot_cfg())
  viewer.launch(robot.spec.compile())
