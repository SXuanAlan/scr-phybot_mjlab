"""PHYBot C2 robot constants and configuration.

23-DOF humanoid (2 legs x 6, waist yaw, 2 arms x 4, neck yaw/pitch).
The 4 hand/foot spine pads mount through passive ball-joint gimbals.
"""

import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils import spec_config as spec_cfg
from mjlab.utils.spec_config import CollisionCfg

from scr_phybot_mjab.assets.utils import update_assets

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

##
# Passive pad gimbal.
#
# The microspine pad chassis mounts to the arm/shin through a passive ball
# joint — three coincident hinges (mjlab rejects a literal ball joint carrying
# joint springs). The explicit ball-center site B is distinct from the
# contact-array center C: C-B = (0, -40, +75) mm in pad coordinates.
# Travel hard-stops at the hardware's +-3 deg. A weak centering spring
# re-zeros the pad whenever it is unloaded, so joint zero is the nominal CAD
# mounting orientation.
#
# Consequences wired in the climb cfg: the weld is rotationally RIGID
# (anchor_torquescale 1.0) and there is no angle-based release — failure is
# unified onto the force/moment billing. Closed-chain PD conflicts that a
# rigid weld would bill as parasitic pad moments live inside the gimbal
# travel at ~0.05 N*m of spring torque.
##

PHYBOT_C2_PAD_GIMBAL_TRAVEL_RAD = math.radians(3.0)  # hardware stop
# Centering spring per hinge. Sized for recentering only (bench 07-31:
# k~0.7, d~0.015 settles a ~1e-4 kg*m^2 pad near-critically); under load it
# is deliberately negligible next to the PD-structure path (~50-260
# N*m/rad, see the calibration doc par.3).
PHYBOT_C2_PAD_GIMBAL_STIFFNESS = 0.7  # N*m/rad
PHYBOT_C2_PAD_GIMBAL_DAMPING = 0.015  # N*m*s/rad
PHYBOT_C2_PAD_GIMBAL_STOP_SOLREF = (0.004, 1.0)  # +-3 deg hard stops
# The XML's phybot_c2 default class stamps armature=0.1 (motor reflected
# inertia) on every joint — 1000x the pad's own ~1.5e-4 kg*m^2, which would
# slow recentering to a ~1 s crawl. A passive ball joint carries none;
# 1e-5 keeps the solver comfortable without burying the pad dynamics.
PHYBOT_C2_PAD_GIMBAL_ARMATURE = 1.0e-5
# Provisional solid-PLA volume integrals from the adjusted CAD. Full COM
# tensors live in the XML, separately for moving pads and fixed arm mounts.
# Replace these estimates with weighed assemblies when available.
PHYBOT_C2_PAD_GEOMETRY = json.loads((PHYBOT_C2_XML.parent.parent / "microspine_cad_geometry.json").read_text())
PHYBOT_C2_PAD_MASS = PHYBOT_C2_PAD_GEOMETRY["moving_pad"]["mass_kg"]
PHYBOT_C2_FOOT_SITES = (
  ("left_ankle_roll", "left_foot"),
  ("right_ankle_roll", "right_foot"),
)

# Bracing tips for climbing: a released spine pad must never slide along the
# wall (dragging unloaded spine tips over asperities destroys them), so foot
# regrips brace the articulated foot's toe instead. The sphere sits at the
# ankle_roll foot mesh's forward tip (mesh reaches x 0.160, sole z -0.061 in
# the body frame) protruding ~2 mm past the mesh so the wall contact lands on
# the sphere pair, not on the pair-less mesh. Sim-only until the hardware toe
# cap is measured; refit pos/radius from the real part then.
PHYBOT_C2_FOOT_TOES = (
  ("left_foot_spine", "left_ankle_roll", (0.150, 0.0, -0.049), 0.012),
  ("right_foot_spine", "right_ankle_roll", (0.150, 0.0, -0.049), 0.012),
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

# Real-robot torque limits from deploy RL_deploy_cpg/config/rl_params.yaml
# (`torque_limits`). The deploy stack clamps commanded torques to these values,
# and the actuator cfgs below use them as effort_limit. Neck is not
# policy-controlled and keeps its XML limit.
PHYBOT_C2_DEPLOY_TORQUE_LIMITS = {
  "hip_pitch": 200.0,
  "hip_roll": 80.0,
  "hip_yaw": 58.0,
  "knee": 80.0,
  "ankle_pitch": 58.0,
  "ankle_roll": 58.0,
  "waist_yaw": 58.0,
  "shoulder_pitch": 58.0,
  "shoulder_roll": 58.0,
  "shoulder_yaw": 58.0,
  "elbow_pitch": 58.0,
}

# XML-side clamps (motor ctrlrange + joint actuatorfrcrange), unified with the
# deploy caps on 2026-08-06 EXCEPT hip_pitch, deliberately held at the
# conservative vendor 160 (user call, 2026-08-06). The joint actuatorfrcrange
# survives the actuator replacement at spec build, so hip_pitch's effective
# sim clamp is min(effort_limit 200, joint clamp 160) = 160 — the regime every
# shipped checkpoint was trained under — while the torque_headroom reward
# keeps pricing against the deploy 200 (it reads the built model's
# actuator_forcerange).
PHYBOT_C2_XML_TORQUE_CLAMPS = {
  **PHYBOT_C2_DEPLOY_TORQUE_LIMITS,
  "hip_pitch": 160.0,
}

# Import-time sync check: keep the XML from drifting against the table above.
for _group, _limit in PHYBOT_C2_XML_TORQUE_CLAMPS.items():
  if PHYBOT_C2_CTRL_LIMITS[_group] != _limit:
    raise ValueError(
      f"phybot_c2.xml ctrlrange for '{_group}' "
      f"({PHYBOT_C2_CTRL_LIMITS[_group]}) != expected clamp ({_limit}); "
      "keep the XML in sync with PHYBOT_C2_XML_TORQUE_CLAMPS"
    )

PHYBOT_C2_POLICY_JOINT_NAMES = (
  "left_hip_pitch",
  "left_hip_roll",
  "left_hip_yaw",
  "left_knee",
  "left_ankle_pitch",
  "left_ankle_roll",
  "right_hip_pitch",
  "right_hip_roll",
  "right_hip_yaw",
  "right_knee",
  "right_ankle_pitch",
  "right_ankle_roll",
  "waist_yaw",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_shoulder_yaw",
  "left_elbow_pitch",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_shoulder_yaw",
  "right_elbow_pitch",
)
# Every joint of the robot body: the policy joints plus the PD-held neck. The
# passive pad gimbals are excluded; a ".*" joint pattern would include them.
PHYBOT_C2_BODY_JOINT_NAMES = (*PHYBOT_C2_POLICY_JOINT_NAMES, "neck_yaw", "neck_pitch")
PHYBOT_C2_ACTION_SCALE = 0.25
PHYBOT_C2_REAL_PD_DAMPING_BY_GROUP = {
  "hip_pitch": 10.0,
  "hip_roll": 10.0,
  "hip_yaw": 10.0,
  "knee": 10.0,
}
# ankle_pitch -0.295 matches the real robot's zero pose (ZERO.yaml /
# RL_deploy default_dof_pos), so ZERO -> policy switching needs no deploy-side
# change. It leaves the sole tilted ~1.1 deg heel-down (flat would be -0.254);
# the feet_flat reward keeps contact flat while walking.
PHYBOT_C2_STANDING_JOINT_POS = {
  "left_hip_pitch": -0.242,
  "left_hip_roll": 0.029,
  "left_hip_yaw": 0.007,
  "left_knee": 0.490,
  "left_ankle_pitch": -0.295,
  "left_ankle_roll": 0.0,
  "right_hip_pitch": -0.242,
  "right_hip_roll": -0.029,
  "right_hip_yaw": -0.007,
  "right_knee": 0.490,
  "right_ankle_pitch": -0.295,
  "right_ankle_roll": 0.0,
  "waist_yaw": 0.0,
  "neck_yaw": 0.0,
  "neck_pitch": 0.0,
  "left_shoulder_pitch": 0.0,
  "left_shoulder_roll": 0.0,
  "left_shoulder_yaw": 0.0,
  "left_elbow_pitch": -0.170,
  "right_shoulder_pitch": 0.0,
  "right_shoulder_roll": 0.0,
  "right_shoulder_yaw": 0.0,
  "right_elbow_pitch": -0.170,
}

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


def _add_pad_gimbals(spec: mujoco.MjSpec) -> None:
  """Add passive hinges at B, retaining the CAD-based XML inertial at COM."""
  travel = PHYBOT_C2_PAD_GIMBAL_TRAVEL_RAD
  for spine in SPINE_BODY_NAMES:
    body = next(b for b in spec.bodies if b.name == spine)
    pivot = next(s for s in spec.sites if s.name == f"{spine}_ball_center")
    anchor = [float(v) for v in pivot.pos]
    for axis_name, axis in (
      ("x", [1.0, 0.0, 0.0]),
      ("y", [0.0, 1.0, 0.0]),
      ("z", [0.0, 0.0, 1.0]),
    ):
      joint = body.add_joint(
        name=f"{spine}_gimbal_{axis_name}",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        pos=anchor,
        axis=axis,
        stiffness=PHYBOT_C2_PAD_GIMBAL_STIFFNESS,
        damping=PHYBOT_C2_PAD_GIMBAL_DAMPING,
        armature=PHYBOT_C2_PAD_GIMBAL_ARMATURE,
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        range=[-travel, travel],
      )
      joint.solref_limit = list(PHYBOT_C2_PAD_GIMBAL_STOP_SOLREF)


def _add_foot_sites(spec: mujoco.MjSpec) -> None:
  for body_name, site_name in PHYBOT_C2_FOOT_SITES:
    body = next(body for body in spec.bodies if body.name == body_name)
    body.add_site(name=site_name, pos=[0.04, 0.0, -0.04], size=[0.02])


def _add_resting_self_collision_excludes(spec: mujoco.MjSpec) -> None:
  """Exclude arm-vs-same-side-hip resting contacts.

  With the arms hanging at the sides, both the hand spine plate and the
  forearm/elbow rest against the hip sphere. The hand spine touches in the
  default keyframe (penetration ~0.7 mm); the elbow rests on the hip whenever
  the policy stands still at near-zero command. Either trips the torso-subtree
  self-collision termination, which then resets the robot every step ("teleport
  to origin and spin"). The official Isaac config sidesteps this by disabling
  self collisions entirely; here we only exclude the offending resting pairs so
  genuine self collisions (e.g. knee-knee) still terminate.
  """
  # Hand spine plate rests on the hip sphere in the default keyframe.
  hand_spine_bodies = (
    "hand_spine",
    *(f"hand_spine_c_{i}_{j}" for i in range(2) for j in range(2)),
  )
  for side in ("left", "right"):
    for spine_body in hand_spine_bodies:
      spec.add_exclude(
        bodyname1=f"{side}_{spine_body}",
        bodyname2=f"{side}_hip_roll",
      )

  # The forearm/elbow hangs against the whole same-side pelvis cluster (hip
  # roll/yaw/pitch) and the torso base (the base_link_collision geom lives on
  # the "torso" body) whenever the policy stands still. Exclude the elbow
  # against each so resting there never trips the self-collision termination.
  # "torso" is not side-specific.
  for side in ("left", "right"):
    for hip_body in ("hip_roll", "hip_yaw", "hip_pitch"):
      spec.add_exclude(
        bodyname1=f"{side}_elbow_pitch",
        bodyname2=f"{side}_{hip_body}",
      )
    spec.add_exclude(bodyname1=f"{side}_elbow_pitch", bodyname2="torso")

  # The upper arm (shoulder) rests against the chest. Exclude shoulder vs both
  # torso bodies on the same side for the same reason.
  for side in ("left", "right"):
    for shoulder_body in ("shoulder_roll", "shoulder_yaw"):
      for torso_body in ("torso", "torso_upper"):
        spec.add_exclude(
          bodyname1=f"{side}_{shoulder_body}",
          bodyname2=torso_body,
        )


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
    group = joint_name.removeprefix("left_").removeprefix("right_")
    deploy_limit = PHYBOT_C2_DEPLOY_TORQUE_LIMITS.get(group)
    if deploy_limit is not None:
      joint.actfrcrange = (-deploy_limit, deploy_limit)
    else:
      joint.actfrcrange = ctrlrange
    joint.actfrclimited = True


def _reset_global_fields_for_scene_attach(spec: mujoco.MjSpec) -> None:
  """Drop child-global settings that ``MjSpec.attach`` cannot propagate.

  The standalone robot XML intentionally keeps its own solver, visual, and
  memory settings for deployment-side MuJoCo programs.  MJLab attaches the
  robot as a child of a scene, where these fields are always owned by the
  parent scene (and MuJoCo 3.10 now warns about every conflict).  Resetting
  only the in-memory child spec keeps standalone behavior unchanged while the
  task's ``SimulationCfg`` remains the single source of truth in MJLab.
  """
  defaults = mujoco.MjSpec()
  spec.option.iterations = defaults.option.iterations
  spec.option.solver = defaults.option.solver
  spec.visual.map.znear = defaults.visual.map.znear
  spec.visual.map.zfar = defaults.visual.map.zfar
  spec.njmax = defaults.njmax
  spec.nconmax = defaults.nconmax


def _trim_spine_pad_visuals(spec: mujoco.MjSpec) -> None:
  """Replace each spine pad's LONG visual mesh with a box matching its
  collision plate (user call 2026-08-25: the stalk part of the visual is
  not the hardware's pad and reads as a 30 cm blade once the gimbal swings
  free). Visual-only: collision plate, contact spheres, toe caps, masses
  (explicit inertials) and every physics quantity stay bit-identical."""
  for spine in SPINE_BODY_NAMES:
    vis = next((g for g in spec.geoms if g.name == f"{spine}_visual"), None)
    col = next((g for g in spec.geoms if g.name == f"{spine}_collision"), None)
    if vis is None or col is None:
      continue
    if vis.meshname.startswith("microspine_"):
      continue  # Calibrated CAD backplate, not the retired long stalk mesh.
    vis.type = mujoco.mjtGeom.mjGEOM_BOX
    vis.meshname = ""
    vis.size = [float(v) for v in col.size]
    vis.pos = [float(v) for v in col.pos]
    vis.quat = [float(v) for v in col.quat]


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PHYBOT_C2_XML))
  spec.assets = get_assets(spec.meshdir)
  _reset_global_fields_for_scene_attach(spec)
  _delete_embedded_floor(spec)
  _delete_embedded_keyframes(spec)
  _delete_embedded_actuators(spec)
  _sync_joint_force_limits_to_ctrlrange(spec)
  _add_pad_gimbals(spec)
  _add_resting_self_collision_excludes(spec)
  _add_foot_sites(spec)
  _trim_spine_pad_visuals(spec)
  return spec


##
# Actuator config.
##

# Nominal sim-PD. The real-PD walk task variant overrides hip/knee damping to
# match MotorList/config/phybot_mini_*.yaml without changing this baseline.
PHYBOT_C2_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch",),
  stiffness=100.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_DEPLOY_TORQUE_LIMITS["hip_pitch"],
)

PHYBOT_C2_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll",),
  stiffness=100.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_DEPLOY_TORQUE_LIMITS["hip_roll"],
)

PHYBOT_C2_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw",),
  stiffness=100.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_DEPLOY_TORQUE_LIMITS["hip_yaw"],
)

PHYBOT_C2_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee",),
  stiffness=100.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_DEPLOY_TORQUE_LIMITS["knee"],
)

PHYBOT_C2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_DEPLOY_TORQUE_LIMITS["ankle_pitch"],
)

PHYBOT_C2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_DEPLOY_TORQUE_LIMITS["ankle_roll"],
)

PHYBOT_C2_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw",),
  stiffness=20.0,
  damping=10.0,
  effort_limit=PHYBOT_C2_DEPLOY_TORQUE_LIMITS["waist_yaw"],
)

PHYBOT_C2_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_DEPLOY_TORQUE_LIMITS["shoulder_pitch"],
)

PHYBOT_C2_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_roll",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_DEPLOY_TORQUE_LIMITS["shoulder_roll"],
)

PHYBOT_C2_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_yaw",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_DEPLOY_TORQUE_LIMITS["shoulder_yaw"],
)

PHYBOT_C2_ACTUATOR_ELBOW_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_pitch",),
  stiffness=50.0,
  damping=5.0,
  effort_limit=PHYBOT_C2_DEPLOY_TORQUE_LIMITS["elbow_pitch"],
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

# Base z = 0.726 puts the lowest sole point (heel) exactly on the floor in
# the -0.295 ankle pose (measured: sole zmin = base - 0.7258); the old 0.70
# embedded the feet ~2.6 cm into the ground at every reset.
DEFAULT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.726),
  joint_pos=PHYBOT_C2_STANDING_JOINT_POS,
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


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_phybot_c2_robot_cfg())
  viewer.launch(robot.spec.compile())
