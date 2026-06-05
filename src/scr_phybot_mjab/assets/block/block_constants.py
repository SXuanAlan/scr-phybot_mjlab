"""Block asset constants and configuration.

A rigid box for friction model validation.
Uses a grid of small box geoms on the bottom face for dense contact distribution.
"""

import mujoco

from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# Default geometry.
##

BLOCK_HALF_SIZE = [0.05, 0.05, 0.05]  # 10cm cube
BLOCK_MASS = 1.0  # kg
BLOCK_GRID = 8  # NxN grid of sub-geoms on the contact face

##
# Collision config — only match collision sub-geoms.
##

BLOCK_COLLISION = CollisionCfg(
  geom_names_expr=("block_c_.*",),
  condim=3,
  friction=(1.0, 0.005, 0.0001),
  contype=1,
  conaffinity=1,
)


##
# Spec builder.
##


def get_block_spec(
  half_size: list[float] | None = None,
  mass: float = BLOCK_MASS,
  grid: int = BLOCK_GRID,
) -> mujoco.MjSpec:
  """Build an MjSpec for a rigid box using a grid of small sub-geoms.

  The block is composed of:
    - 1 visual-only box geom (for rendering the full cube)
    - NxN small box geoms on the bottom face (for collision)

  Args:
    half_size: Half-extents [hx, hy, hz]. Defaults to BLOCK_HALF_SIZE.
    mass: Total block mass in kg.
    grid: NxN grid subdivisions on the contact face.
  """
  if half_size is None:
    half_size = list(BLOCK_HALF_SIZE)

  hx, hy, hz = half_size

  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="block_body")
  body.add_freejoint(name="block_joint")

  # Visual-only geom for rendering (no collision).
  body.add_geom(
    name="block_visual",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=half_size,
    rgba=[0.2, 0.6, 0.85, 0.4],
    contype=0,
    conaffinity=0,
    mass=0,
  )

  # Grid of small box geoms on the bottom face (-z face in local frame).
  sub_hx = hx / grid
  sub_hy = hy / grid
  sub_hz = 0.002  # thin collision pad
  sub_mass = mass / (grid * grid)

  for i in range(grid):
    for j in range(grid):
      cx = -hx + sub_hx + 2 * sub_hx * i
      cy = -hy + sub_hy + 2 * sub_hy * j
      cz = -hz + sub_hz  # flush with bottom face

      body.add_geom(
        name=f"block_c_{i}_{j}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[sub_hx, sub_hy, sub_hz],
        pos=[cx, cy, cz],
        mass=sub_mass,
        rgba=[0.2, 0.6, 0.85, 0.0],  # invisible
      )

  return spec


##
# Config factory.
##


def get_block_cfg(
  half_size: list[float] | None = None,
  mass: float = BLOCK_MASS,
  grid: int = BLOCK_GRID,
) -> EntityCfg:
  """Get a block configuration.

  Args:
    half_size: Half-extents [x, y, z].
    mass: Block mass in kg.
    grid: NxN contact grid subdivisions.
  """
  return EntityCfg(
    spec_fn=lambda: get_block_spec(half_size, mass, grid),
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    collisions=(BLOCK_COLLISION,),
  )
