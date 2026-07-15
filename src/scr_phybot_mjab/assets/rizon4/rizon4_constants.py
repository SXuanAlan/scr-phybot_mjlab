"""Flexiv Rizon4 robot constants and MJLab configuration.

The generated URDF and meshes are vendored from
``flexivrobotics/flexiv_description`` branch ``humble-v1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import mujoco

  from mjlab.entity import EntityCfg

##
# URDF and meshes.
##

RIZON4_DIR: Path = Path(__file__).resolve().parent
RIZON4_URDF: Path = RIZON4_DIR / "urdf" / "rizon4.urdf"
RIZON4_MESHES: Path = RIZON4_DIR / "meshes" / "Rizon4"
RIZON4_COLLISION_MESHES: Path = RIZON4_MESHES / "collision"
RIZON4_VISUAL_MESHES: Path = RIZON4_MESHES / "visual"
assert RIZON4_URDF.exists(), f"URDF not found: {RIZON4_URDF}"
assert RIZON4_COLLISION_MESHES.exists(), f"Mesh dir not found: {RIZON4_COLLISION_MESHES}"
assert RIZON4_VISUAL_MESHES.exists(), f"Visual mesh dir not found: {RIZON4_VISUAL_MESHES}"

RIZON4_JOINT_NAMES = (
  "joint1",
  "joint2",
  "joint3",
  "joint4",
  "joint5",
  "joint6",
  "joint7",
)

RIZON4_JOINT_EFFORT_LIMITS = {
  "joint1": 123.0,
  "joint2": 123.0,
  "joint3": 64.0,
  "joint4": 64.0,
  "joint5": 39.0,
  "joint6": 39.0,
  "joint7": 39.0,
}

RIZON4_POSITION_STIFFNESS = 5000.0
RIZON4_POSITION_DAMPING = 300.0
RIZON4_POSITION_EFFORT_LIMIT = 2000.0

RIZON4_HOME_JOINT_POS = {
  "joint1": 0.0,
  "joint2": -0.698132,
  "joint3": 0.0,
  "joint4": 1.5708,
  "joint5": 0.0,
  "joint6": 0.698132,
  "joint7": 0.0,
}

RIZON4_MICROSPINE_EE_BODY = "microspine_pad"
RIZON4_MICROSPINE_TCP_SITE = "microspine_tcp"
RIZON4_MICROSPINE_C2_HAND_MESH: Path = (
  RIZON4_DIR.parent / "phybot_c2" / "new_meshes" / "right_hand_spine.STL"
)
assert RIZON4_MICROSPINE_C2_HAND_MESH.exists(), (
  f"C2 hand-spine mesh not found: {RIZON4_MICROSPINE_C2_HAND_MESH}"
)

RIZON4_MICROSPINE_BODY_POS = (-0.04, 0.0, 0.04)
RIZON4_MICROSPINE_BODY_QUAT = (0.0, 0.7071068, 0.7071068, 0.0)
RIZON4_MICROSPINE_VISUAL_POS = (0.0, 0.0, 0.2)
RIZON4_MICROSPINE_VISUAL_RGBA = (0.94902, 0.94118, 0.90981, 1.0)
RIZON4_MICROSPINE_COLLISION_SIZE = (0.06, 0.006, 0.04935)
RIZON4_MICROSPINE_COLLISION_POS = (-0.0009236, -0.006, 0.20830)
RIZON4_MICROSPINE_TCP_POS = (-0.0009236, -0.012, 0.20830)
RIZON4_MICROSPINE_SPHERE_SIZE = 0.012
RIZON4_MICROSPINE_SPHERE_POSITIONS = (
  ((-0.04892, -0.012, 0.171), (0.04708, -0.012, 0.171)),
  ((-0.04892, -0.012, 0.2456), (0.04708, -0.012, 0.2456)),
)

RIZON4_TELEOP_POS_STEP = 0.01
RIZON4_TELEOP_ROT_STEP = 0.04

RIZON4_BODY_NAMES = (
  "base_link",
  "link1",
  "link2",
  "link3",
  "link4",
  "link5",
  "link6",
  "link7",
)

RIZON4_VISUAL_SPECS = (
  ("base_link", "rizon4_link0_shell", "link0.obj", (1.0, 1.0, 1.0), (0.62, 0.64, 0.68, 1.0)),
  ("link1", "rizon4_link1_shell", "link1.obj", (1.0, 1.0, 1.0), (0.62, 0.64, 0.68, 1.0)),
  ("link1", "rizon4_link1_ring", "ring.obj", (0.062, 0.062, 0.002), (0.18, 0.2, 0.22, 1.0)),
  ("link2", "rizon4_link2_shell", "link2.obj", (1.0, 1.0, 1.0), (0.62, 0.64, 0.68, 1.0)),
  ("link2", "rizon4_link2_ring", "ring.obj", (0.062, 0.062, 0.002), (0.18, 0.2, 0.22, 1.0)),
  ("link3", "rizon4_link3_shell", "link3.obj", (1.0, 1.0, 1.0), (0.62, 0.64, 0.68, 1.0)),
  ("link3", "rizon4_link3_ring", "ring.obj", (0.051, 0.051, 0.002), (0.18, 0.2, 0.22, 1.0)),
  ("link4", "rizon4_link4_shell", "link4.obj", (1.0, 1.0, 1.0), (0.62, 0.64, 0.68, 1.0)),
  ("link4", "rizon4_link4_ring", "ring.obj", (0.051, 0.051, 0.002), (0.18, 0.2, 0.22, 1.0)),
  ("link5", "rizon4_link5_shell", "link5.obj", (1.0, 1.0, 1.0), (0.62, 0.64, 0.68, 1.0)),
  ("link5", "rizon4_link5_ring", "ring.obj", (0.051, 0.051, 0.002), (0.18, 0.2, 0.22, 1.0)),
  ("link6", "rizon4_link6_shell", "link6.obj", (1.0, 1.0, 1.0), (0.62, 0.64, 0.68, 1.0)),
  ("link6", "rizon4_link6_ring", "ring.obj", (0.051, 0.051, 0.002), (0.18, 0.2, 0.22, 1.0)),
  ("link7", "rizon4_link7_shell", "link7.obj", (1.0, 1.0, 1.0), (0.62, 0.64, 0.68, 1.0)),
  ("link7", "rizon4_link7_ring", "ring.obj", (0.051, 0.051, 0.002), (0.18, 0.2, 0.22, 1.0)),
)


def get_assets(meshdir: str) -> dict[str, bytes]:
  """Return mesh assets embedded into the generated MuJoCo spec."""
  from scr_phybot_mjab.assets.utils import update_assets

  assets: dict[str, bytes] = {}
  update_assets(assets, RIZON4_COLLISION_MESHES, meshdir)
  return assets


def _rename_imported_collision_geoms(spec: mujoco.MjSpec) -> None:
  """Give URDF-imported collision geoms stable names."""
  for geom_index, geom in enumerate(spec.geoms):
    if geom_index < len(RIZON4_JOINT_NAMES) + 1:
      geom.name = f"rizon4_link{geom_index}_collision"


def _configure_imported_collision_geoms(spec: mujoco.MjSpec) -> None:
  for geom in spec.geoms:
    if geom.name.startswith("rizon4_link") and geom.name.endswith("_collision"):
      geom.group = 3
      geom.condim = 3
      geom.friction = [1.0, 0.1, 0.1]


def _add_visual_geoms(spec: mujoco.MjSpec) -> None:
  """Add URDF visual meshes as visible-only geoms for Viser/native viewers."""
  import mujoco
  import trimesh

  for body_name, visual_name, mesh_file, scale, rgba in RIZON4_VISUAL_SPECS:
    body = spec.body(body_name)
    if body is None:
      raise ValueError(f"Rizon4 URDF import did not contain body {body_name!r}.")

    mesh = trimesh.load(RIZON4_VISUAL_MESHES / mesh_file, force="mesh", process=False)
    vertices = mesh.vertices.copy()
    vertices[:, 0] *= scale[0]
    vertices[:, 1] *= scale[1]
    vertices[:, 2] *= scale[2]
    mesh_name = f"{visual_name}_mesh"
    spec.add_mesh(
      name=mesh_name,
      uservert=vertices.reshape(-1).astype(float).tolist(),
      userface=mesh.faces.reshape(-1).astype(int).tolist(),
    )
    body.add_geom(
      name=f"{visual_name}_visual",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh_name,
      group=0,
      contype=0,
      conaffinity=0,
      rgba=rgba,
      mass=0,
    )


def _add_microspine_pad(spec: mujoco.MjSpec) -> None:
  import mujoco
  import trimesh

  flange = spec.body("flange")
  if flange is None:
    raise ValueError("Rizon4 URDF import did not contain a 'flange' body.")

  # Use the same spine mesh, collision box, and 2x2 contact-sphere layout as
  # the PHYBot C2 hand spine. The mounting rotation is the C2 hand-spine pose,
  # applied on the Rizon4 flange.
  pad = flange.add_body(
    name=RIZON4_MICROSPINE_EE_BODY,
    pos=RIZON4_MICROSPINE_BODY_POS,
    quat=RIZON4_MICROSPINE_BODY_QUAT,
  )
  pad.mass = 0.05
  pad.ipos = [0.0, 0.0, 0.0]
  pad.inertia = [1.0e-4, 1.0e-4, 1.0e-4]

  spine_mesh = trimesh.load(
    RIZON4_MICROSPINE_C2_HAND_MESH,
    force="mesh",
    process=False,
  )
  spine_mesh_name = "microspine_pad_visual_mesh"
  spec.add_mesh(
    name=spine_mesh_name,
    uservert=spine_mesh.vertices.reshape(-1).astype(float).tolist(),
    userface=spine_mesh.faces.reshape(-1).astype(int).tolist(),
  )
  pad.add_geom(
    name="microspine_pad_visual",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=spine_mesh_name,
    pos=RIZON4_MICROSPINE_VISUAL_POS,
    group=1,
    rgba=RIZON4_MICROSPINE_VISUAL_RGBA,
    contype=0,
    conaffinity=0,
    mass=0,
  )
  pad.add_geom(
    name="microspine_pad_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=RIZON4_MICROSPINE_COLLISION_SIZE,
    pos=RIZON4_MICROSPINE_COLLISION_POS,
  )
  pad.add_site(
    name=RIZON4_MICROSPINE_TCP_SITE,
    pos=RIZON4_MICROSPINE_TCP_POS,
    size=[0.008],
    rgba=[1.0, 0.2, 0.1, 1.0],
  )

  for row, sphere_row in enumerate(RIZON4_MICROSPINE_SPHERE_POSITIONS):
    for col, sphere_pos in enumerate(sphere_row):
      sphere_name = f"{RIZON4_MICROSPINE_EE_BODY}_c_{row}_{col}"
      sphere_body = pad.add_body(
        name=sphere_name,
        pos=sphere_pos,
      )
      sphere_body.add_geom(
        name=sphere_name,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[RIZON4_MICROSPINE_SPHERE_SIZE],
        contype=1,
        conaffinity=1,
        condim=6,
        priority=1,
        friction=[30.0, 0.01, 0.01],
        solref=[0.02, 1.5],
        solimp=[0.9, 0.99, 0.001, 0.5, 2.0],
        rgba=[0.9, 0.7, 0.5, 0.0],
        mass=0.001,
      )


def get_spec() -> mujoco.MjSpec:
  """Build the Rizon4 MuJoCo spec with a fixed microspine pad on the flange."""
  import mujoco

  spec = mujoco.MjSpec.from_file(str(RIZON4_URDF))
  spec.compiler.discardvisual = False
  spec.meshdir = str(RIZON4_COLLISION_MESHES)
  spec.assets = get_assets(spec.meshdir)
  _rename_imported_collision_geoms(spec)
  _configure_imported_collision_geoms(spec)
  _add_visual_geoms(spec)
  _add_microspine_pad(spec)
  return spec


def get_rizon4_robot_cfg() -> EntityCfg:
  """Get a fresh Rizon4 robot configuration instance."""
  from mjlab.actuator import BuiltinPositionActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
  from mjlab.utils.spec_config import CollisionCfg

  actuators = tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(joint_name,),
      stiffness=RIZON4_POSITION_STIFFNESS,
      damping=RIZON4_POSITION_DAMPING,
      effort_limit=RIZON4_POSITION_EFFORT_LIMIT,
      armature=0.02,
    )
    for joint_name in RIZON4_JOINT_NAMES
  )
  articulation = EntityArticulationInfoCfg(
    actuators=actuators,
    soft_joint_pos_limit_factor=1.0,
  )
  collision = CollisionCfg(
    geom_names_expr=(
      r"^rizon4_link[0-7]_collision$",
      r"^microspine_pad_collision$",
      r"^microspine_pad_c_[0-9]+_[0-9]+$",
    ),
    condim={
      r"^microspine_pad_c_[0-9]+_[0-9]+$": 6,
      r".*": 3,
    },
    friction={
      r"^microspine_pad_c_[0-9]+_[0-9]+$": (30.0, 0.01, 0.01),
      r".*": (1.0, 0.1, 0.1),
    },
    priority={r"^microspine_pad_c_[0-9]+_[0-9]+$": 1, r".*": 0},
  )
  init_state = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos=RIZON4_HOME_JOINT_POS,
    joint_vel={".*": 0.0},
  )
  return EntityCfg(
    init_state=init_state,
    collisions=(collision,),
    spec_fn=get_spec,
    articulation=articulation,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_rizon4_robot_cfg())
  viewer.launch(robot.spec.compile())
