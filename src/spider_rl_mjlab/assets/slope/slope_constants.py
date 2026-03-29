"""Slope asset constants and configuration.

A flat inclined plane for friction model validation.
The slope angle is baked into the geometry via quaternion rotation around the Y axis.
"""

import math

import mujoco

from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# Default geometry.
##

SLOPE_HALF_SIZE = [1.0, 0.5, 0.02]  # 2m long, 1m wide, 4cm thick slab

##
# Collision config.
##

SLOPE_COLLISION = CollisionCfg(
  geom_names_expr=("slope_geom",),
  condim=3,
  friction=(1.0, 0.005, 0.0001),
  contype=1,
  conaffinity=1,
)


def _angle_to_quat(angle_deg: float) -> tuple[float, float, float, float]:
  """Convert slope angle (degrees) to quaternion for rotation around Y axis.

  Positive angle tilts the +x end upward.
  Returns (w, x, y, z) quaternion.
  """
  half = math.radians(angle_deg) / 2.0
  return (math.cos(half), 0.0, math.sin(half), 0.0)


def _slope_center_pos(angle_deg: float) -> tuple[float, float, float]:
  """Compute slope center position so that the lower edge sits at z=0, x=0."""
  length = SLOPE_HALF_SIZE[0]
  thickness = SLOPE_HALF_SIZE[2]
  rad = math.radians(angle_deg)
  cx = length * math.cos(rad) + thickness * math.sin(rad)
  cz = length * math.sin(rad) + thickness * math.cos(rad)
  return (cx, 0.0, cz)


##
# Spec builder.
##


def get_slope_spec(angle_deg: float = 30.0) -> mujoco.MjSpec:
  """Build an MjSpec for a tilted box (slope).

  Args:
    angle_deg: Inclination angle in degrees (0 = flat, 90 = vertical wall).
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="slope_body")
  body.add_geom(
    name="slope_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=SLOPE_HALF_SIZE,
    rgba=[0.45, 0.42, 0.38, 1.0],
  )
  return spec


##
# Config factory.
##


def get_slope_cfg(angle_deg: float = 30.0) -> EntityCfg:
  """Get a slope configuration with given inclination angle.

  Args:
    angle_deg: Slope angle in degrees.
  """
  pos = _slope_center_pos(angle_deg)
  quat = _angle_to_quat(angle_deg)

  return EntityCfg(
    spec_fn=lambda: get_slope_spec(angle_deg),
    init_state=EntityCfg.InitialStateCfg(pos=pos, rot=quat),
    collisions=(SLOPE_COLLISION,),
  )
