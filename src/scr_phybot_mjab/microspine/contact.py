"""Microspine contact-grid helpers and physical wall contact pairs."""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.pad_geometry import matching_pad_geometry


def sphere_names(ee_body_name: str, rows: int, cols: int) -> list[str]:
  """Return contact body/geom names (legacy API name; boxes or spheres)."""
  return [f"{ee_body_name}_c_{row}_{col}" for row in range(rows) for col in range(cols)]


def contact_face_offset(geom_type, size, face_axis, quat=(1, 0, 0, 0)) -> np.ndarray:
  """Vector from a proxy's center to its nominal pad-facing surface center.

  The result is in the geom's parent-body frame. Boxes must present one of
  their faces along the pad normal; spheres use the point along that normal.
  Runtime retraction is deliberately excluded from this geometric reference.
  """
  axis = np.asarray(face_axis, dtype=float)
  axis = axis / np.linalg.norm(axis)
  kind = np.asarray(geom_type).item()
  if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
    return float(size[0]) * axis
  if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat, dtype=float))
    rotation = rotation.reshape(3, 3)
    local = rotation.T @ axis
    i = int(np.argmax(np.abs(local)))
    if not np.isclose(abs(local[i]), 1.0, atol=1e-6):
      raise ValueError("A contact box face must align with the pad face axis")
    return float(size[i]) * axis
  raise ValueError(f"Unsupported microspine contact geom type: {geom_type}")


def pad_contact_face_centers(source, cfg: MicrospineCfg, ee: str, robot_prefix: str = "robot") -> np.ndarray:
  """Nominal corner surface centers in the pad body frame, from spec or model."""
  prefix = f"{robot_prefix}/" if robot_prefix else ""
  points = []
  for name in sphere_names(ee, cfg.spheres_per_ee_rows, cfg.spheres_per_ee_cols):
    body, geom = source.body(prefix + name), source.geom(prefix + name)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(body.quat, dtype=float))
    rotation = rotation.reshape(3, 3)
    face = rotation.T @ np.asarray(cfg.face_local_axis)
    offset = contact_face_offset(geom.type, geom.size, face, geom.quat)
    points.append(np.asarray(body.pos) + rotation @ (np.asarray(geom.pos) + offset))
  return np.asarray(points)


def pad_contact_center(source, cfg: MicrospineCfg, ee: str, robot_prefix: str = "robot") -> np.ndarray:
  """Use the authored array centre C; legacy proxy-only rigs use their mean."""
  prefix = f"{robot_prefix}/" if robot_prefix else ""
  try:
    site = source.site(prefix + ee + "_contact_center")
  except KeyError:
    site = None
  if site is not None:
    return np.asarray(site.pos, dtype=float).copy()
  return pad_contact_face_centers(source, cfg, ee, robot_prefix).mean(axis=0)


def pad_grid_spans(
  spec: mujoco.MjSpec,
  cfg: MicrospineCfg,
  robot_prefix: str = "robot",
) -> tuple[float, float]:
  """Unit-center extents (rake_span_m, lateral_span_m) from the spec.

  Single source of truth for the physical pad size: the moment-billing
  fields ``MicrospineLimitSurfaceCfg.pad_span_rake_m`` /
  ``pad_span_lateral_m`` are derived from the robot XML through this helper
  (see the climb env cfg) instead of being hand-maintained constants, and
  clip-generation / analysis scripts can import the same numbers.

  Explicit ``<ee>_unit_<row>_<col>`` sites define the load-bearing array.
  Older assets without these sites fall back to the contact-sphere grid.
  Spans are measured between center extremes along the pad-local rake
  axis (``limit_surface.ee_rake_local_axis``) and the lateral axis
  (``face_local_axis x rake``). All EEs must agree to within 1 mm — one
  scalar pair parameterizes the billing model.
  """
  prefix = f"{robot_prefix}/" if robot_prefix else ""
  if any(site.name and site.name.startswith(f"{prefix}{cfg.ee_body_names[0]}_unit_") for site in spec.sites):
    return matching_pad_geometry(
      spec,
      cfg.ee_body_names,
      robot_prefix,
      rake_local_axis=cfg.limit_surface.ee_rake_local_axis,
      face_local_axis=cfg.face_local_axis,
      unit_layout=cfg.limit_surface.unit_layout,
    ).spans

  rake = np.asarray(cfg.limit_surface.ee_rake_local_axis, dtype=float)
  rake = rake / max(float(np.linalg.norm(rake)), 1.0e-9)
  face = np.asarray(cfg.face_local_axis, dtype=float)
  face = face / max(float(np.linalg.norm(face)), 1.0e-9)
  lateral = np.cross(face, rake)
  lateral = lateral / max(float(np.linalg.norm(lateral)), 1.0e-9)

  prefix = f"{robot_prefix}/" if robot_prefix else ""
  spans: list[tuple[float, float]] = []
  for ee in cfg.ee_body_names:
    unit_sites = [site for site in spec.sites if site.name and site.name.startswith(f"{prefix}{ee}_unit_")]
    centers = np.asarray(
      [site.pos for site in unit_sites]
      if unit_sites
      else [
        spec.body(f"{prefix}{name}").pos for name in sphere_names(ee, cfg.spheres_per_ee_rows, cfg.spheres_per_ee_cols)
      ],
      dtype=float,
    )
    along_rake = centers @ rake
    along_lateral = centers @ lateral
    spans.append(
      (
        float(along_rake.max() - along_rake.min()),
        float(along_lateral.max() - along_lateral.min()),
      )
    )
  spans_arr = np.asarray(spans)
  if np.ptp(spans_arr, axis=0).max() > 1.0e-3:
    raise ValueError(
      "Contact-sphere grids differ across EEs; a single pad-span pair "
      f"cannot represent them: {dict(zip(cfg.ee_body_names, spans))}"
    )
  if spans_arr.min() <= 0.0:
    raise ValueError(f"Degenerate sphere-grid span: {spans}")
  return float(spans_arr[0, 0]), float(spans_arr[0, 1])


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
    pair = spec.add_pair()
    pair.geomname1 = f"{prefix}{toe_geom_name(toe)}"
    pair.geomname2 = cfg.wall_geom_name
    pair.condim = cfg.contact_condim
    pair.friction = list(cfg.contact_friction)
    pair.solref = list(cfg.contact_solref)
    pair.solimp = list(cfg.contact_solimp)
    pair.solreffriction = list(cfg.contact_solref)
    pair.margin = cfg.contact_margin
    pair.gap = cfg.contact_gap


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


def add_arm_wall_contact_pairs(
  spec: mujoco.MjSpec,
  cfg: MicrospineCfg,
  robot_prefix: str = "robot",
) -> None:
  """Explicit arm-vs-wall pairs (v6 dare, 2026-08-24, user decision).

  The upper arms and elbows leave the wall-contact FENCE (death on touch)
  and become ordinary physical contacts instead -- the robot may lean an
  arm on the wall and take support through it, like the hardware would.
  Same wall contact parameters as the toe pairs; the palm plates stay out
  (the weld system owns them). Enabled per task via
  ``cfg.arm_wall_contact_geoms`` (empty = legacy plant, climb untouched).
  """
  prefix = f"{robot_prefix}/" if robot_prefix else ""
  for geom_name in cfg.arm_wall_contact_geoms:
    pair = spec.add_pair()
    pair.geomname1 = f"{prefix}{geom_name}"
    pair.geomname2 = cfg.wall_geom_name
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
    pair = spec.add_pair()
    pair.geomname1 = f"{robot_prefix}/{geom_name}"
    pair.geomname2 = target_geom
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
    for name in sphere_names(ee, cfg.spheres_per_ee_rows, cfg.spheres_per_ee_cols):
      gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"{robot_prefix}/{name}")
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
    corner_pos: (..., 4, 3) world positions of the four corner surface centers
      when sphere_radius=0, or legacy sphere centers otherwise.
    wall_point: (..., 3) any point on the wall surface.
    wall_normal: (..., 3) unit wall normal pointing off the surface toward
      the robot side.
    sphere_radius: legacy radius [m]; use zero for explicit surface points.
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
  points = w00 * p00.unsqueeze(-2) + w01 * p01.unsqueeze(-2) + w10 * p10.unsqueeze(-2) + w11 * p11.unsqueeze(-2)
  dist = ((points - wall_point.unsqueeze(-2)) * wall_normal.unsqueeze(-2)).sum(dim=-1)
  hooked = (dist - sphere_radius) <= travel
  return hooked.float().mean(dim=-1)
