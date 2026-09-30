"""Free rigid test block; dimensions are full extents in metres."""

import math
from dataclasses import dataclass

import mujoco
from mjlab.entity import EntityCfg


@dataclass(frozen=True)
class BrickCfg:
  size: tuple[float, float, float] = (0.18, 0.32, 0.16)
  mass: float = 1.0
  start_xy: tuple[float, float] = (0.32, 0.0)
  goal_offset_xy: tuple[float, float] = (0.0, 0.15)
  lift_height: float = 0.12
  friction: float = 0.5

  def __post_init__(self):
    if len(self.size) != 3 or len(self.start_xy) != 2 or len(self.goal_offset_xy) != 2:
      raise ValueError("Brick size must have three entries; planar positions must have two")
    if any(not math.isfinite(v) or v <= 0 for v in (*self.size, self.mass, self.lift_height, self.friction)):
      raise ValueError("Brick dimensions, mass, friction and lift height must be positive and finite")
    if not all(math.isfinite(v) for v in (*self.start_xy, *self.goal_offset_xy)):
      raise ValueError("Brick start and goal positions must be finite")

  def spec(self):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="brick")
    body.add_freejoint(name="free_joint")
    body.add_geom(
      name="brick_geom",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=[s / 2 for s in self.size],
      mass=self.mass,
      rgba=[0.62, 0.30, 0.16, 1.0],
      friction=[self.friction, 0.005, 0.0001],
      condim=3,
      solref=[0.01, 1.0],
    )
    return spec

  def entity(self):
    return EntityCfg(
      spec_fn=self.spec,
      init_state=EntityCfg.InitialStateCfg(pos=(*self.start_xy, self.size[2] / 2 + 0.001), rot=(1.0, 0.0, 0.0, 0.0)),
    )
