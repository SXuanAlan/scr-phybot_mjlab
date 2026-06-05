"""Wall asset constants and configuration."""

import mujoco

from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# Wall geometry.
##

WALL_HALF_SIZE = [0.5, 1.5, 7.5]  # 1m thick, 3m wide, 15m tall
WALL_OFFSET = (2.0, 0.0, 7.5)  # center offset from env origin (z = half-height so wall sits on ground)

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
    solref=[0.02, 1.5],   # unified: 20ms time constant, moderate overdamping
    solimp=[0.9, 0.99, 0.001, 0.5, 2],  # unified impedance, 1mm transition zone
  )
  return spec


##
# Final config.
##


def get_wall_cfg() -> EntityCfg:
  """Get a fresh wall configuration instance."""
  return EntityCfg(
    spec_fn=get_wall_spec,
    init_state=EntityCfg.InitialStateCfg(pos=WALL_OFFSET),
    collisions=(WALL_COLLISION,),
  )
