"""PHYBot C2 robot constants and configuration.

23-DOF humanoid (2 legs x 6, waist yaw, 2 arms x 4, neck yaw/pitch).
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

PHYBOT_C2_XML: Path = Path(__file__).resolve().parent / "xml" / "phybot_c2.xml"
assert PHYBOT_C2_XML.exists(), f"MJCF not found: {PHYBOT_C2_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PHYBOT_C2_XML.parent.parent / "new_meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PHYBOT_C2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

PHYBOT_C2_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=200.0,
)

PHYBOT_C2_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=80.0,
)

PHYBOT_C2_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=58.0,
)

PHYBOT_C2_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=80.0,
)

PHYBOT_C2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=58.0,
)

PHYBOT_C2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=58.0,
)

PHYBOT_C2_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=58.0,
)

PHYBOT_C2_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=58.0,
)

PHYBOT_C2_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_roll",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=58.0,
)

PHYBOT_C2_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_yaw",),
  stiffness=5.0,
  damping=5.0,
  effort_limit=58.0,
)

PHYBOT_C2_ACTUATOR_ELBOW_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=58.0,
)

PHYBOT_C2_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
  target_names_expr=("neck_yaw", "neck_pitch"),
  stiffness=20.0,
  damping=2.0,
  effort_limit=30.0,
)

##
# Keyframe config.
##

DEFAULT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.70),
  joint_pos={
    "left_hip_pitch": -0.24,
    "left_hip_roll": 0.026,
    "left_hip_yaw": 0.012,
    "left_knee": 0.51,
    "left_ankle_pitch": -0.28,
    "left_ankle_roll": 0.0,
    "right_hip_pitch": -0.24,
    "right_hip_roll": 0.026,
    "right_hip_yaw": 0.012,
    "right_knee": 0.51,
    "right_ankle_pitch": -0.28,
    "right_ankle_roll": 0.0,
    "waist_yaw": 0.0,
    "neck_yaw": 0.0,
    "neck_pitch": 0.0,
    "left_shoulder_pitch": 0.0,
    "left_shoulder_roll": 1.0,
    "left_shoulder_yaw": -0.1,
    "left_elbow_pitch": 0.0,
    "right_shoulder_pitch": 0.0,
    "right_shoulder_roll": -1.0,
    "right_shoulder_yaw": 0.0,
    "right_elbow_pitch": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

PHYBOT_C2_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  condim={
    r"^(left|right)_ankle_roll_collision$": 6,
    ".*": 3,
  },
  friction={
    r"^(left|right)_ankle_roll_collision$": (30.0, 0.01, 0.01),
    ".*": (1.0, 0.1, 0.1),
  },
  priority={
    r"^(left|right)_ankle_roll_collision$": 1,
  },
)

##
# Final config.
##

PHYBOT_C2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    PHYBOT_C2_ACTUATOR_HIP_PITCH,
    PHYBOT_C2_ACTUATOR_HIP_ROLL,
    PHYBOT_C2_ACTUATOR_HIP_YAW,
    PHYBOT_C2_ACTUATOR_KNEE,
    PHYBOT_C2_ACTUATOR_ANKLE_PITCH,
    PHYBOT_C2_ACTUATOR_ANKLE_ROLL,
    PHYBOT_C2_ACTUATOR_WAIST_YAW,
    PHYBOT_C2_ACTUATOR_SHOULDER_PITCH,
    PHYBOT_C2_ACTUATOR_SHOULDER_ROLL,
    PHYBOT_C2_ACTUATOR_SHOULDER_YAW,
    PHYBOT_C2_ACTUATOR_ELBOW_PITCH,
    PHYBOT_C2_ACTUATOR_NECK,
  ),
  soft_joint_pos_limit_factor=1.0,
)


def get_phybot_c2_robot_cfg() -> EntityCfg:
  """Get a fresh PHYBot C2 robot configuration instance."""
  return EntityCfg(
    init_state=DEFAULT_KEYFRAME,
    collisions=(PHYBOT_C2_COLLISION,),
    spec_fn=get_spec,
    articulation=PHYBOT_C2_ARTICULATION,
  )


PHYBOT_C2_ACTION_SCALE = 0.25


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_phybot_c2_robot_cfg())
  viewer.launch(robot.spec.compile())
