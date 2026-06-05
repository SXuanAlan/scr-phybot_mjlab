"""PHYBot C2 robot constants and configuration.

23-DOF humanoid (2 legs x 6, waist yaw, 2 arms x 4, neck yaw/pitch).
The 4 hand/foot spines are rigid attachments (URDF fixed joints / no MuJoCo joint).
"""

from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils import spec_config as spec_cfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

PHYBOT_C2_XML: Path = Path(__file__).resolve().parent / "xml" / "phybot_c2.xml"
assert PHYBOT_C2_XML.exists(), f"MJCF not found: {PHYBOT_C2_XML}"

SPINE_BODY_NAMES = (
  "left_foot_spine",
  "right_foot_spine",
  "left_hand_spine",
  "right_hand_spine",
)


def _parse_xml_range(value: str) -> tuple[float, float]:
  lower, upper = (float(item) for item in value.split())
  return lower, upper


def _read_xml_motor_ctrl_ranges() -> dict[str, tuple[float, float]]:
  root = ET.parse(PHYBOT_C2_XML).getroot()
  ranges = {
    motor.attrib["name"]: _parse_xml_range(motor.attrib["ctrlrange"])
    for motor in root.findall("./actuator/motor")
    if "name" in motor.attrib and "ctrlrange" in motor.attrib
  }
  if ranges:
    return ranges

  return {
    joint.attrib["name"]: _parse_xml_range(joint.attrib["actuatorfrcrange"])
    for joint in root.findall(".//joint")
    if "name" in joint.attrib and "actuatorfrcrange" in joint.attrib
  }


def _range_effort_limit(ctrlrange: tuple[float, float]) -> float:
  return max(abs(ctrlrange[0]), abs(ctrlrange[1]))


def _ctrl_group_limit(*motor_names: str) -> float:
  limits = [_range_effort_limit(PHYBOT_C2_MOTOR_CTRL_RANGES[name]) for name in motor_names]
  if any(limit != limits[0] for limit in limits):
    raise ValueError(f"Mismatched ctrlrange limits for {motor_names}: {limits}")
  return limits[0]


PHYBOT_C2_MOTOR_CTRL_RANGES = _read_xml_motor_ctrl_ranges()

PHYBOT_C2_CTRL_LIMITS = {
  "hip_pitch": _ctrl_group_limit("left_hip_pitch", "right_hip_pitch"),
  "hip_roll": _ctrl_group_limit("left_hip_roll", "right_hip_roll"),
  "hip_yaw": _ctrl_group_limit("left_hip_yaw", "right_hip_yaw"),
  "knee": _ctrl_group_limit("left_knee", "right_knee"),
  "ankle_pitch": _ctrl_group_limit("left_ankle_pitch", "right_ankle_pitch"),
  "ankle_roll": _ctrl_group_limit("left_ankle_roll", "right_ankle_roll"),
  "waist_yaw": _ctrl_group_limit("waist_yaw"),
  "shoulder_pitch": _ctrl_group_limit("left_shoulder_pitch", "right_shoulder_pitch"),
  "shoulder_roll": _ctrl_group_limit("left_shoulder_roll", "right_shoulder_roll"),
  "shoulder_yaw": _ctrl_group_limit("left_shoulder_yaw", "right_shoulder_yaw"),
  "elbow_pitch": _ctrl_group_limit("left_elbow_pitch", "right_elbow_pitch"),
  "neck": _ctrl_group_limit("neck_yaw", "neck_pitch"),
}

PHYBOT_C2_JOINT_FORCE_LIMITS = PHYBOT_C2_MOTOR_CTRL_RANGES

PHYBOT_C2_FLOOR_TEXTURE = spec_cfg.TextureCfg(
  name="texplane",
  type="2d",
  builtin="checker",
  rgb1=(0.2, 0.3, 0.4),
  rgb2=(0.1, 0.15, 0.2),
  width=100,
  height=100,
)
PHYBOT_C2_FLOOR_MATERIAL = spec_cfg.MaterialCfg(
  name="MatPlane",
  reflectance=0.5,
  texture="texplane",
  texrepeat=(1.0, 1.0),
  texuniform=True,
  geom_names_expr=("terrain$",),
)
PHYBOT_C2_FLOOR_FRICTION = (1.0, 0.5, 0.5)
PHYBOT_C2_FLOOR_RGBA = (0.8, 0.9, 0.8, 1.0)
PHYBOT_C2_FLOOR_SIZE = (50.0, 50.0, 0.125)
PHYBOT_C2_FLOOR_POS = (0.0, 0.0, 0.0)


def configure_phybot_c2_terrain_cfg(terrain_cfg) -> None:
  if terrain_cfg is None:
    return
  terrain_cfg.textures = (PHYBOT_C2_FLOOR_TEXTURE,)
  terrain_cfg.materials = (PHYBOT_C2_FLOOR_MATERIAL,)


def configure_phybot_c2_floor_geom(
  spec: mujoco.MjSpec,
  *,
  geom_name: str = "terrain/terrain",
  enable_contact: bool = True,
) -> None:
  try:
    geom = spec.geom(geom_name)
  except Exception:
    return
  if geom is None and geom_name == "terrain/terrain":
    try:
      geom = spec.geom("terrain")
    except Exception:
      return
  if geom is None:
    return

  geom.type = mujoco.mjtGeom.mjGEOM_PLANE
  geom.pos = PHYBOT_C2_FLOOR_POS
  geom.size = PHYBOT_C2_FLOOR_SIZE
  geom.friction = PHYBOT_C2_FLOOR_FRICTION
  geom.condim = 3
  geom.contype = 1 if enable_contact else 0
  geom.conaffinity = 1 if enable_contact else 0
  geom.group = 2
  geom.rgba = PHYBOT_C2_FLOOR_RGBA
  try:
    geom.material = PHYBOT_C2_FLOOR_MATERIAL.name
    spec.material(PHYBOT_C2_FLOOR_MATERIAL.name).reflectance = 0.5
  except Exception:
    pass


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PHYBOT_C2_XML.parent.parent / "new_meshes", meshdir)
  return assets


def _lighten_spine_bodies(spec: mujoco.MjSpec) -> None:
  for body in spec.bodies:
    if body.name in SPINE_BODY_NAMES:
      body.mass = 0.05
      body.ipos = [0.0, 0.0, 0.0]
      body.inertia = [1.0e-4, 1.0e-4, 1.0e-4]


def _delete_embedded_floor(spec: mujoco.MjSpec) -> None:
  for geom in list(spec.geoms):
    if geom.name == "floor":
      spec.delete(geom)


def _delete_embedded_keyframes(spec: mujoco.MjSpec) -> None:
  for key in list(spec.keys):
    spec.delete(key)


def _delete_embedded_actuators(spec: mujoco.MjSpec) -> None:
  for actuator in list(spec.actuators):
    spec.delete(actuator)


def _sync_joint_force_limits_to_ctrlrange(spec: mujoco.MjSpec) -> None:
  for joint_name, ctrlrange in PHYBOT_C2_JOINT_FORCE_LIMITS.items():
    joint = spec.joint(joint_name)
    if joint is None:
      continue
    joint.actfrcrange = ctrlrange
    joint.actfrclimited = True


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PHYBOT_C2_XML))
  spec.assets = get_assets(spec.meshdir)
  _delete_embedded_floor(spec)
  _delete_embedded_keyframes(spec)
  _delete_embedded_actuators(spec)
  _sync_joint_force_limits_to_ctrlrange(spec)
  _lighten_spine_bodies(spec)
  return spec


##
# Actuator config.
##

PHYBOT_C2_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=PHYBOT_C2_CTRL_LIMITS["hip_pitch"],
)

PHYBOT_C2_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=PHYBOT_C2_CTRL_LIMITS["hip_roll"],
)

PHYBOT_C2_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw",),
  stiffness=60.0,
  damping=6.0,
  effort_limit=PHYBOT_C2_CTRL_LIMITS["hip_yaw"],
)

PHYBOT_C2_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee",),
  stiffness=60.0,
  damping=6.0,
  effort_limit=PHYBOT_C2_CTRL_LIMITS["knee"],
)

PHYBOT_C2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch",),
  stiffness=30.0,
  damping=3.0,
  effort_limit=PHYBOT_C2_CTRL_LIMITS["ankle_pitch"],
)

PHYBOT_C2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_CTRL_LIMITS["ankle_roll"],
)

PHYBOT_C2_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw",),
  stiffness=20.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_CTRL_LIMITS["waist_yaw"],
)

PHYBOT_C2_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_CTRL_LIMITS["shoulder_pitch"],
)

PHYBOT_C2_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_roll",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_CTRL_LIMITS["shoulder_roll"],
)

PHYBOT_C2_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_yaw",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_CTRL_LIMITS["shoulder_yaw"],
)

PHYBOT_C2_ACTUATOR_ELBOW_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_CTRL_LIMITS["elbow_pitch"],
)

PHYBOT_C2_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
  target_names_expr=("neck_yaw", "neck_pitch"),
  stiffness=20.0,
  damping=2.0,
  effort_limit=PHYBOT_C2_CTRL_LIMITS["neck"],
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
    "left_shoulder_pitch": -2.0,
    "left_shoulder_roll": -0.17,
    "left_shoulder_yaw": 0.0,
    "left_elbow_pitch": -1.14,
    "right_shoulder_pitch": -2.0,
    "right_shoulder_roll": 0.17,
    "right_shoulder_yaw": 0.0,
    "right_elbow_pitch": -1.14,
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
