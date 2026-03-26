"""PHYBot V8 robot constants and configuration.

Follows the pattern from phybot_c2_constants.py, adapted for the
PHYBot V8 26-DOF humanoid (24 actuated joints: +waist_roll, +elbow_yaw x2
compared to PHYBot C2).
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

PHYBOT_V8_XML: Path = Path(__file__).resolve().parent / "phybot_v8_26.xml"
assert PHYBOT_V8_XML.exists(), f"MJCF not found: {PHYBOT_V8_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PHYBOT_V8_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PHYBOT_V8_XML))
  spec.assets = get_assets(spec.meshdir)

  # Add MuJoCo adhesion actuators for microspine end-effectors.
  from spider_rl_mjlab.assets.microspine.actuators import add_adhesion_actuators
  from spider_rl_mjlab.assets.microspine.config import MicrospineCfg

  # PHYBot V8 arm EEs are the hand bodies at the tip of each arm.
  cfg = MicrospineCfg(
    ee_body_names=(
      "left_ankle_roll",
      "right_ankle_roll",
      "left_hand",
      "right_hand",
    ),
  )
  add_adhesion_actuators(spec, cfg)
  return spec


##
# Actuator config.
##

PHYBOT_V8_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=500.0,
)

PHYBOT_V8_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=200.0,
)

PHYBOT_V8_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=200.0,
)

PHYBOT_V8_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=500.0,
)

PHYBOT_V8_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=200.0,
)

PHYBOT_V8_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=58.0,
)

PHYBOT_V8_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=230.0,
)

PHYBOT_V8_ACTUATOR_WAIST_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_roll",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=200.0,
)

PHYBOT_V8_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=200.0,
)

PHYBOT_V8_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_roll",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=200.0,
)

PHYBOT_V8_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_yaw",),
  stiffness=5.0,
  damping=5.0,
  effort_limit=58.0,
)

PHYBOT_V8_ACTUATOR_ELBOW_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=58.0,
)

PHYBOT_V8_ACTUATOR_ELBOW_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_yaw",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=58.0,
)

##
# Keyframe config.
##

CLIMB_WALL_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.1),
  joint_pos={
    "left_hip_pitch": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_yaw": 0.0,
    "left_knee": 0.0,
    "left_ankle_pitch": 0.0,
    "left_ankle_roll": 0.0,
    "right_hip_pitch": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_knee": 0.0,
    "right_ankle_pitch": 0.0,
    "right_ankle_roll": 0.0,
    "waist_yaw": 0.0,
    "waist_roll": 0.0,
    "left_shoulder_pitch": 0.0,
    "left_shoulder_roll": 0.0,
    "left_shoulder_yaw": 0.0,
    "left_elbow_pitch": 0.0,
    "left_elbow_yaw": 0.0,
    "right_shoulder_pitch": 0.0,
    "right_shoulder_roll": 0.0,
    "right_shoulder_yaw": 0.0,
    "right_elbow_pitch": 0.0,
    "right_elbow_yaw": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

PHYBOT_V8_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  condim=3,
  friction=(1.0, 0.1, 0.1),
)

##
# Final config.
##

PHYBOT_V8_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    PHYBOT_V8_ACTUATOR_HIP_PITCH,
    PHYBOT_V8_ACTUATOR_HIP_ROLL,
    PHYBOT_V8_ACTUATOR_HIP_YAW,
    PHYBOT_V8_ACTUATOR_KNEE,
    PHYBOT_V8_ACTUATOR_ANKLE_PITCH,
    PHYBOT_V8_ACTUATOR_ANKLE_ROLL,
    PHYBOT_V8_ACTUATOR_WAIST_YAW,
    PHYBOT_V8_ACTUATOR_WAIST_ROLL,
    PHYBOT_V8_ACTUATOR_SHOULDER_PITCH,
    PHYBOT_V8_ACTUATOR_SHOULDER_ROLL,
    PHYBOT_V8_ACTUATOR_SHOULDER_YAW,
    PHYBOT_V8_ACTUATOR_ELBOW_PITCH,
    PHYBOT_V8_ACTUATOR_ELBOW_YAW,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_phybot_v8_robot_cfg() -> EntityCfg:
  """Get a fresh PHYBot V8 robot configuration instance."""
  return EntityCfg(
    init_state=CLIMB_WALL_KEYFRAME,
    collisions=(PHYBOT_V8_COLLISION,),
    spec_fn=get_spec,
    articulation=PHYBOT_V8_ARTICULATION,
  )


PHYBOT_V8_ACTION_SCALE = 0.25


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_phybot_v8_robot_cfg())
  viewer.launch(robot.spec.compile())
