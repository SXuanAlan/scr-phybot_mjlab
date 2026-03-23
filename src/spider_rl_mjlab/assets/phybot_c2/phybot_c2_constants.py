"""PHYBot C2 robot constants and configuration.

Follows the pattern from mjlab's G1 robot definition:
  mjlab/asset_zoo/robots/unitree_g1/g1_constants.py
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

PHYBOT_C2_XML: Path = Path(__file__).resolve().parent / "xmls" / "phybot_c2.xml"
assert PHYBOT_C2_XML.exists(), f"MJCF not found: {PHYBOT_C2_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PHYBOT_C2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PHYBOT_C2_XML))
  spec.assets = get_assets(spec.meshdir)

  # Add MuJoCo adhesion actuators for microspine end-effectors.
  from spider_rl_mjlab.microspine.actuators import add_adhesion_actuators
  from spider_rl_mjlab.microspine.config import MicrospineCfg

  add_adhesion_actuators(spec, MicrospineCfg())
  return spec


##
# Actuator config.
#
# Gains from the original IsaacLab PHYBOT_C2_CFG (ImplicitActuatorCfg).
# mjlab uses BuiltinPositionActuatorCfg which creates MuJoCo position actuators.
##

# BuiltinPositionActuatorCfg requires scalar stiffness/damping/effort_limit,
# so we create one config per unique gain group.

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

PHYBOT_C2_ACTUATOR_FEET = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch", ".*_ankle_roll"),
  stiffness=50.0,
  damping=5.0,
  effort_limit=58.0,
)

PHYBOT_C2_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
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

PHYBOT_C2_ACTUATOR_LEFT_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=("left_elbow_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=9.0,
)

PHYBOT_C2_ACTUATOR_RIGHT_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=("right_elbow_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=58.0,
)

##
# Keyframe config.
##

CLIMB_WALL_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.66),
  joint_pos={
    "left_hip_pitch": -0.242,
    "left_hip_roll": 0.029,
    "left_hip_yaw": 0.007,
    "left_knee": 0.49,
    "left_ankle_pitch": -0.295,
    "left_ankle_roll": 0.0,
    "right_hip_pitch": -0.242,
    "right_hip_roll": -0.029,
    "right_hip_yaw": -0.007,
    "right_knee": 0.49,
    "right_ankle_pitch": -0.295,
    "right_ankle_roll": 0.0,
    "waist_yaw": 0.0,
    "left_shoulder_pitch": 0.0,
    "left_shoulder_roll": 0.0,
    "left_shoulder_yaw": 0.0,
    "left_elbow_pitch": -0.17,
    "right_shoulder_pitch": 0.0,
    "right_shoulder_roll": 0.0,
    "right_shoulder_yaw": 0.0,
    "right_elbow_pitch": -0.17,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

PHYBOT_C2_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  condim=3,
  friction=(1.0, 0.1, 0.1),
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
    PHYBOT_C2_ACTUATOR_FEET,
    PHYBOT_C2_ACTUATOR_WAIST,
    PHYBOT_C2_ACTUATOR_SHOULDER_PITCH,
    PHYBOT_C2_ACTUATOR_SHOULDER_ROLL,
    PHYBOT_C2_ACTUATOR_SHOULDER_YAW,
    PHYBOT_C2_ACTUATOR_LEFT_ELBOW,
    PHYBOT_C2_ACTUATOR_RIGHT_ELBOW,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_phybot_c2_robot_cfg() -> EntityCfg:
  """Get a fresh PHYBot C2 robot configuration instance."""
  return EntityCfg(
    init_state=CLIMB_WALL_KEYFRAME,
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
