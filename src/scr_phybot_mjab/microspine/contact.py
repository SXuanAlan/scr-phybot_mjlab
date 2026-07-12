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
