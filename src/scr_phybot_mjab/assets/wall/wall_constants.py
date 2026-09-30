"""Wall asset constants and configuration."""

import mujoco
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# Wall geometry.
##

# Real test-wall dimensions: 2 m along the climbing direction (its height
# when vertical), 1.2 m wide. Thickness stays 1 m — wall_geom.size[0] is the
# half-thickness CONTRACT the microspine projection code reconstructs the
# surface from. The climb task collides against the analytic surface PLANE
# (box is visual), so these numbers change what you SEE, not the contact
# surface at x = WALL_PIVOT_X.
WALL_HALF_SIZE = [0.5, 0.6, 1.0]  # 1m thick, 1.2m wide, 2m tall
WALL_OFFSET = (2.0, 0.0, 1.0)  # center offset from env origin (z = half-height so wall sits on ground)
# Robot-facing surface x at ground level — the hinge line the wall-angle
# representation tilts about (env-local; the surface/floor intersection must
# not move with angle).
WALL_PIVOT_X = WALL_OFFSET[0] - WALL_HALF_SIZE[0]
_WALL_SURFACE_LOCAL_X = -WALL_HALF_SIZE[0]
_LOCAL_Z_TO_NEGATIVE_X_QUAT = [
  0.7071067811865476,
  0.0,
  -0.7071067811865476,
  0.0,
]

##
# Collision config.
##

WALL_COLLISION = CollisionCfg(
  geom_names_expr=("wall_geom",),
  condim=3,
  friction=(1.0, 0.005, 0.0001),  # MuJoCo defaults
  contype=1,
  conaffinity=1,
)

##
# Spec.
##


def get_wall_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="wall_body")
  body.add_geom(
    name="wall_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=WALL_HALF_SIZE,
    rgba=[0.6, 0.6, 0.6, 1.0],
    solref=[0.02, 1.5],  # unified: 20ms time constant, moderate overdamping
    solimp=[0.9, 0.99, 0.001, 0.5, 2],  # unified impedance, 1mm transition zone
  )
  return spec


def get_wall_plane_collision_spec() -> mujoco.MjSpec:
  """Build the wall with a finite visual box and an analytic contact plane.

  The plane lies on the robot-facing surface of the original box (local
  ``x = -WALL_HALF_SIZE[0]``), with its normal pointing toward the robot.
  ``wall_geom.size[0]`` intentionally retains the box half-thickness because
  microspine anchoring and wall-distance code use that value to reconstruct
  the surface from the wall body's center pose.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="wall_body")
  body.add_geom(
    name="wall_visual_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=WALL_HALF_SIZE,
    rgba=[0.6, 0.6, 0.6, 1.0],
    contype=0,
    conaffinity=0,
  )
  body.add_geom(
    name="wall_geom",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    # Plane size[0:2] is visual-only in MuJoCo. Preserve size[0] as the wall
    # half-thickness contract used by the microspine projection code.
    size=[WALL_HALF_SIZE[0], WALL_HALF_SIZE[1], 0.1],
    pos=[_WALL_SURFACE_LOCAL_X, 0.0, 0.0],
    quat=_LOCAL_Z_TO_NEGATIVE_X_QUAT,
    rgba=[0.6, 0.6, 0.6, 0.0],
    solref=[0.02, 1.5],
    solimp=[0.9, 0.99, 0.001, 0.5, 2],
  )
  return spec


##
# Final config.
##


def get_wall_cfg(*, analytic_collision_plane: bool = False) -> EntityCfg:
  """Get a fresh wall configuration instance."""
  return EntityCfg(
    spec_fn=(get_wall_plane_collision_spec if analytic_collision_plane else get_wall_spec),
    init_state=EntityCfg.InitialStateCfg(pos=WALL_OFFSET),
    collisions=(WALL_COLLISION,),
  )
