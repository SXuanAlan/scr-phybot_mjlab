"""Scene edits for the microspine plant: wall contact pairs, bracing toes and weld anchors."""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.contact import get_sphere_geom_names, get_spheres_per_ee


@dataclasses.dataclass(frozen=True)
class ToeContact:
  """A non-spine bracing tip (e.g. the articulated foot's toe).

  A released microspine pad must never slide along the wall — dragging
  unloaded spine tips over asperities damages them. During a foot regrip the
  robot braces this tip against the wall instead: it supplies ordinary
  normal compression (plus mu-friction) while the spine pad travels clear of
  the surface. The geom is collision-disabled (contype/conaffinity 0) and
  reaches the wall exclusively through its explicit contact pair, so it is
  inert in every non-climb task and never collides with the robot itself.
  """

  ee_body_name: str
  """Microspine EE this tip belongs to (the pad it substitutes for)."""

  body_name: str
  """Body carrying the tip (e.g. ``left_ankle_roll``)."""

  pos: tuple[float, float, float]
  """Tip-sphere center in the body frame [m]."""

  radius: float = 0.012
  """Tip-sphere radius [m]."""


def toe_geom_name(toe: ToeContact) -> str:
  return f"{toe.body_name}_toe"


def toe_site_name(toe: ToeContact) -> str:
  return f"{toe.body_name}_toe_site"


def microspine_equality_name(ee_body_name: str) -> str:
  """Return the scene-level equality name for one end-effector."""
  return f"microspine_{ee_body_name}"


def _add_wall_pair(spec: mujoco.MjSpec, cfg: MicrospineCfg, geom: str, target: str) -> None:
  """One explicit contact pair with the ordinary wall contact parameters."""
  pair = spec.add_pair()
  pair.geomname1 = geom
  pair.geomname2 = target
  pair.condim = cfg.contact_condim
  pair.friction = list(cfg.contact_friction)
  pair.solref = list(cfg.contact_solref)
  pair.solimp = list(cfg.contact_solimp)
  pair.solreffriction = list(cfg.contact_solref)
  pair.margin = cfg.contact_margin
  pair.gap = cfg.contact_gap


def add_contact_pairs(
  spec: mujoco.MjSpec,
  cfg: MicrospineCfg,
  robot_prefix: str = "robot",
) -> None:
  """Add ordinary contact-proxy-to-wall pairs used with constraint grip.

  These contacts model compression and pre-engagement sliding only. The
  equality anchor and limit surface own tangential grip and pull-off behavior,
  so no high-friction or external-adhesion approximation is needed.
  """
  surfaces = cfg.ordered_surfaces()
  target_geoms = (
    [s.scoped_geom for s in surfaces for _ in range(get_spheres_per_ee(cfg))]
    if surfaces
    else [cfg.wall_geom_name] * len(get_sphere_geom_names(cfg))
  )
  for geom_name, target_geom in zip(get_sphere_geom_names(cfg), target_geoms):
    _add_wall_pair(spec, cfg, f"{robot_prefix}/{geom_name}", target_geom)


def add_arm_wall_contact_pairs(
  spec: mujoco.MjSpec,
  cfg: MicrospineCfg,
  robot_prefix: str = "robot",
) -> None:
  """Explicit arm-vs-wall pairs for ``cfg.arm_wall_contact_geoms``.

  The listed arm geoms become ordinary physical wall contacts, so the robot
  may lean an arm on the wall and take support through it, like the
  hardware would. The palm plates stay out: the weld anchors own them.
  """
  prefix = f"{robot_prefix}/" if robot_prefix else ""
  for geom_name in cfg.arm_wall_contact_geoms:
    _add_wall_pair(spec, cfg, f"{prefix}{geom_name}", cfg.wall_geom_name)


def add_pad_wall_contact_excludes(
  spec: mujoco.MjSpec,
  cfg: MicrospineCfg,
  robot_prefix: str = "robot",
  wall_body_name: str = "wall/wall_body",
) -> None:
  """Exclude each pad's chassis collision proxies from the wall.

  The pad body owns the chassis boxes; its children own the corner-contact
  grid. Letting both representations collide with the wall duplicates and
  over-constrains the contact patch. MuJoCo body-pair excludes apply only to
  the named bodies, not their children, so all explicit corner-wall pairs
  added by :func:`add_contact_pairs` remain active.
  """
  for ee_body_name in cfg.ee_body_names:
    spec.add_exclude(
      bodyname1=f"{robot_prefix}/{ee_body_name}",
      bodyname2=wall_body_name,
    )


def add_toe_geoms(
  spec: mujoco.MjSpec,
  toes: tuple[ToeContact, ...],
  robot_prefix: str = "robot",
  with_sites: bool = False,
) -> None:
  """Author the bracing-tip spheres (and optionally center sites) on ``spec``.

  Geoms are contype/conaffinity 0: they touch the wall only through the
  explicit pairs from :func:`add_toe_contact_pairs`. ``with_sites`` adds a
  site at each tip center for IK / analysis scripts.
  """
  prefix = f"{robot_prefix}/" if robot_prefix else ""
  for toe in toes:
    body = spec.body(f"{prefix}{toe.body_name}")
    body.add_geom(
      name=f"{prefix}{toe_geom_name(toe)}",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=[toe.radius, 0.0, 0.0],
      pos=list(toe.pos),
      contype=0,
      conaffinity=0,
      group=3,
      rgba=[0.9, 0.4, 0.1, 0.0],
    )
    if with_sites:
      body.add_site(
        name=f"{prefix}{toe_site_name(toe)}",
        pos=list(toe.pos),
        size=[0.004, 0.004, 0.004],
      )


def add_toe_contact_pairs(
  spec: mujoco.MjSpec,
  cfg: MicrospineCfg,
  toes: tuple[ToeContact, ...],
  robot_prefix: str = "robot",
) -> None:
  """Explicit toe-vs-wall pairs with the ordinary wall contact parameters."""
  prefix = f"{robot_prefix}/" if robot_prefix else ""
  for toe in toes:
    _add_wall_pair(spec, cfg, f"{prefix}{toe_geom_name(toe)}", cfg.wall_geom_name)


def add_attachment_equalities(
  spec: mujoco.MjSpec,
  cfg: MicrospineCfg,
  robot_prefix: str = "robot",
) -> None:
  """Add one inactive weld ("virtual anchor") per end-effector.

  The compiler recomputes the anchor and relpose slots at qpos0, so every
  slot is written at runtime; only torquescale is authored here, for spec
  readability, and the executor re-asserts it anyway.
  """
  prefix = f"{robot_prefix}/" if robot_prefix else ""
  surfaces = cfg.ordered_surfaces()
  for index, ee in enumerate(cfg.ee_body_names):
    data = np.zeros(11)
    data[10] = cfg.anchor_torquescale
    spec.add_equality(
      name=microspine_equality_name(ee),
      type=mujoco.mjtEq.mjEQ_WELD,
      objtype=mujoco.mjtObj.mjOBJ_BODY,
      name1=f"{prefix}{ee}",
      name2=surfaces[index].scoped_body if surfaces else "",
      active=False,
      solref=list(cfg.anchor_solref),
      solimp=list(cfg.anchor_solimp),
      data=data.tolist(),
    )
