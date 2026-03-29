"""Default friction test environment configuration.

Configurable parameters:
  - slope_angle_deg: Inclination of the slope [degrees]
  - normal_force: Extra normal force pressing block into slope [N]
  - friction_coef: Coulomb friction coefficient for block-slope contact
  - block_mass: Mass of the block [kg]

Analytical prediction:
  Block slides when: m*g*sin(theta) > mu * (m*g*cos(theta) + N_ext)
  Critical angle (no extra force): theta_c = arctan(mu)
  With extra normal force: theta_c = arctan(mu * (m*g*cos(theta) + N_ext) / (m*g))
"""

import math

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg

from spider_rl_mjlab.assets.block import get_block_cfg
from spider_rl_mjlab.assets.slope import get_slope_cfg
from spider_rl_mjlab.assets.slope.slope_constants import SLOPE_HALF_SIZE
from spider_rl_mjlab.tasks.friction_test.friction_test_env_cfg import make_friction_test_env_cfg

##
# Adjustable parameters — change these to run different experiments.
##

SLOPE_ANGLE_DEG = 30.0     # slope inclination [degrees]
NORMAL_FORCE = 0.0         # extra normal force pressing block into slope [N]
FRICTION_COEF = 0.578         # Coulomb friction coefficient
BLOCK_MASS = 10.0            # block mass [kg]


def _print_analytical_prediction() -> None:
  """Print analytical friction model prediction for reference."""
  g = 9.81
  theta = math.radians(SLOPE_ANGLE_DEG)
  m = BLOCK_MASS

  f_tangential = m * g * math.sin(theta)  # gravity component along slope
  f_normal_gravity = m * g * math.cos(theta)  # gravity component into slope
  f_normal_total = f_normal_gravity + NORMAL_FORCE  # total normal force
  f_friction_max = FRICTION_COEF * f_normal_total  # max static friction

  print("=" * 60)
  print("Friction Test — Analytical Prediction")
  print("=" * 60)
  print(f"  Slope angle:       {SLOPE_ANGLE_DEG:.1f} deg")
  print(f"  Block mass:        {BLOCK_MASS:.2f} kg")
  print(f"  Friction coef:     {FRICTION_COEF:.3f}")
  print(f"  Extra normal force:{NORMAL_FORCE:.1f} N")
  print(f"  ---")
  print(f"  F_tangential (gravity along slope): {f_tangential:.3f} N")
  print(f"  F_normal (gravity into slope):      {f_normal_gravity:.3f} N")
  print(f"  F_normal (total, incl. external):   {f_normal_total:.3f} N")
  print(f"  F_friction_max (mu * F_normal):     {f_friction_max:.3f} N")
  print(f"  ---")
  if f_tangential > f_friction_max:
    print(f"  PREDICTION: Block SLIDES (tangential {f_tangential:.3f} > friction {f_friction_max:.3f})")
  else:
    print(f"  PREDICTION: Block STAYS (tangential {f_tangential:.3f} <= friction {f_friction_max:.3f})")
  print("=" * 60)


def friction_test_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the friction test environment with slope + block."""
  _print_analytical_prediction()

  cfg = make_friction_test_env_cfg(
    slope_angle_deg=SLOPE_ANGLE_DEG,
    normal_force=NORMAL_FORCE,
    friction_coef=FRICTION_COEF,
    block_mass=BLOCK_MASS,
  )

  slope_cfg = get_slope_cfg(angle_deg=SLOPE_ANGLE_DEG)
  block_cfg = get_block_cfg(mass=BLOCK_MASS)

  # Place block on the slope surface.
  #
  # Slope geometry after R_y(θ):
  #   - local +x end goes DOWN (lower end), local -x end goes UP
  #   - surface normal in world = (sinθ, 0, cosθ)
  #
  # A point at local (d, 0, half_thickness) on the top surface transforms to:
  #   world offset from slope center = (d·cosθ + t·sinθ,  0,  -d·sinθ + t·cosθ)
  # where t = half_thickness.
  #
  # Block center = slope_center + surface_offset + block_half * normal
  theta = math.radians(SLOPE_ANGLE_DEG)
  slope_thickness = SLOPE_HALF_SIZE[2]  # half-thickness of slab
  block_half = 0.05  # block half-size

  slope_pos = slope_cfg.init_state.pos
  # Place block near center of slope (d=0 from slope center)
  d = 0.0
  normal_offset = slope_thickness + block_half  # distance from slope center to block center along normal
  bx = slope_pos[0] + d * math.cos(theta) + normal_offset * math.sin(theta)
  bz = slope_pos[2] - d * math.sin(theta) + normal_offset * math.cos(theta)

  # Rotate block to align with slope surface (same rotation as slope)
  block_quat = slope_cfg.init_state.rot
  block_cfg.init_state.pos = (bx, 0.0, bz)
  block_cfg.init_state.rot = block_quat

  cfg.scene.entities = {
    "block": block_cfg,
    "slope": slope_cfg,
  }

  # Set friction for each block sub-geom vs slope contact pair.
  grid = 8  # must match BLOCK_GRID
  def _add_block_slope_pairs(spec: mujoco.MjSpec) -> None:
    for i in range(grid):
      for j in range(grid):
        pair = spec.add_pair()
        pair.geomname1 = f"block/block_c_{i}_{j}"
        pair.geomname2 = "slope/slope_geom"
        pair.condim = 3
        pair.friction = [FRICTION_COEF, FRICTION_COEF, 0.005, 0.0001, 0.0001]
        pair.solref = [0.002, 1.0]
        pair.solimp = [0.9, 0.99, 0.001, 0.5, 2]

    # Visual flags for contact debugging.
    spec.visual.scale.contactwidth = 0.05
    spec.visual.scale.contactheight = 0.01
    spec.visual.scale.forcewidth = 0.02

  cfg.scene.spec_fn = _add_block_slope_pairs

  if play:
    cfg.episode_length_s = int(1e9)

  return cfg
