"""Microspine contact-grid helpers and physical wall contact pairs."""

from __future__ import annotations

import mujoco

from scr_phybot_mjab.microspine.config import MicrospineCfg


def sphere_names(ee_body_name: str, rows: int, cols: int) -> list[str]:
  """Return sphere body/geom names for one end-effector."""
  return [
    f"{ee_body_name}_c_{row}_{col}"
    for row in range(rows)
    for col in range(cols)
  ]


def get_spheres_per_ee(cfg: MicrospineCfg) -> int:
  """Return the number of contact spheres per end-effector."""
  return cfg.spheres_per_ee_rows * cfg.spheres_per_ee_cols


def get_sphere_geom_names(cfg: MicrospineCfg) -> tuple[str, ...]:
  """Return all microspine sphere geom names in EE-major order."""
  return tuple(
    name
    for ee in cfg.ee_body_names
    for name in sphere_names(
      ee,
      cfg.spheres_per_ee_rows,
      cfg.spheres_per_ee_cols,
    )
  )


def add_contact_pairs(
  spec: mujoco.MjSpec,
  cfg: MicrospineCfg,
  robot_prefix: str = "robot",
) -> None:
  """Add ordinary sphere-to-wall contacts used with constraint grip.

  These contacts model compression and pre-engagement sliding only. The
  equality anchor and limit surface own tangential grip and pull-off behavior,
  so no high-friction or external-adhesion approximation is needed.
  """
  for geom_name in get_sphere_geom_names(cfg):
    pair = spec.add_pair()
    pair.geomname1 = f"{robot_prefix}/{geom_name}"
    pair.geomname2 = cfg.wall_geom_name
    pair.condim = cfg.contact_condim
    pair.friction = list(cfg.contact_friction)
    pair.solref = list(cfg.contact_solref)
    pair.solimp = list(cfg.contact_solimp)
    pair.solreffriction = list(cfg.contact_solref)
    pair.margin = cfg.contact_margin
    pair.gap = cfg.contact_gap


def resolve_sphere_geom_ids(
  mj_model,
  cfg: MicrospineCfg,
  robot_prefix: str = "robot",
) -> list[list[int]]:
  """Map each EE to the compiled model's contact-sphere geom indices.

  Returns one list of geom ids per EE (EE-major, sphere order within).
  Consumed by the latched-contact disable path, which retracts these geoms
  into the pad body per env while latched (runtime ``pair_*`` writes are not
  honored by the collision pipeline in this mjwarp version — verified
  empirically — but ``geom_pos`` feeds the kinematics kernel every step).
  """
  out: list[list[int]] = []
  for ee in cfg.ee_body_names:
    ee_ids: list[int] = []
    for name in sphere_names(
      ee, cfg.spheres_per_ee_rows, cfg.spheres_per_ee_cols
    ):
      gid = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"{robot_prefix}/{name}"
      )
      if gid < 0:
        gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
      if gid < 0:
        raise ValueError(f"Sphere geom '{name}' not found in model.")
      ee_ids.append(gid)
    out.append(ee_ids)
  return out


def add_pad_wall_contact_excludes(
  spec: mujoco.MjSpec,
  cfg: MicrospineCfg,
  robot_prefix: str = "robot",
  wall_body_name: str = "wall/wall_body",
) -> None:
  """Exclude each pad's coarse collision box from the wall.

  The pad body owns a coarse box while its child bodies own the contact-sphere
  grid. Letting both representations collide with the wall duplicates and
  over-constrains the contact patch. MuJoCo body-pair excludes apply only to
  the named bodies, not their children, so all explicit sphere-wall pairs
  added by :func:`add_contact_pairs` remain active.
  """
  for ee_body_name in cfg.ee_body_names:
    spec.add_exclude(
      bodyname1=f"{robot_prefix}/{ee_body_name}",
      bodyname2=wall_body_name,
    )


def virtual_spine_fraction(
  corner_pos,
  wall_point,
  wall_normal,
  sphere_radius: float,
  grid: tuple[int, int],
  travel: float,
):
  """Engaged fraction of a virtual spine field spanning the pad face.

  The real pad carries hundreds of spines, each on a ~force-free normal
  suspension with ``le_max`` of reach (Wang et al. 2017, Sec 2.3: kn = 5
  mN/mm, le_max = 5 mm) — whether a spine can hook is a purely kinematic
  question of whether its site is within reach of the wall. A bilinear
  ``grid`` over the four contact-sphere centers (row-major corners: r0c0,
  r0c1, r1c0, r1c1) stands in for that field; a grid point is engaged when
  its distance to the wall surface, less the sphere radius (the corners sit
  at sphere-center height), is within ``travel``.

  Args:
    corner_pos: (..., 4, 3) world positions of the four corner sphere centers.
    wall_point: (..., 3) any point on the wall surface.
    wall_normal: (..., 3) unit wall normal pointing off the surface toward
      the robot side.
    sphere_radius: contact sphere radius [m].
    grid: (rows, cols) virtual field resolution.
    travel: spine reach beyond the sphere contact surface [m] (le_max).

  Returns:
    (...) engaged fraction in [0, 1]. Flat seated pad -> 1.0; corner poke ->
    the small patch of the field near that corner.
  """
  import torch

  rows, cols = grid
  u = torch.linspace(0.0, 1.0, rows, device=corner_pos.device)
  v = torch.linspace(0.0, 1.0, cols, device=corner_pos.device)
  uu, vv = torch.meshgrid(u, v, indexing="ij")
  uu = uu.reshape(-1)  # (K,)
  vv = vv.reshape(-1)
  p00, p01, p10, p11 = (
    corner_pos[..., 0, :],
    corner_pos[..., 1, :],
    corner_pos[..., 2, :],
    corner_pos[..., 3, :],
  )
  # (..., K, 3) bilinear interpolation over the sphere-corner quad.
  w00 = ((1.0 - uu) * (1.0 - vv)).reshape(*([1] * (corner_pos.dim() - 2)), -1, 1)
  w01 = ((1.0 - uu) * vv).reshape_as(w00)
  w10 = (uu * (1.0 - vv)).reshape_as(w00)
  w11 = (uu * vv).reshape_as(w00)
  points = (
    w00 * p00.unsqueeze(-2)
    + w01 * p01.unsqueeze(-2)
    + w10 * p10.unsqueeze(-2)
    + w11 * p11.unsqueeze(-2)
  )
  dist = (
    (points - wall_point.unsqueeze(-2)) * wall_normal.unsqueeze(-2)
  ).sum(dim=-1)
  hooked = (dist - sphere_radius) <= travel
  return hooked.float().mean(dim=-1)
