"""Load-bearing unit geometry about C, read from asset sites.

No PHYBot dimensions are stored here. Unit ordering is the asset's numeric
row/column order. The engage frame is (rake, outward x rake, outward), with
outward opposite the pad's contact face. Collision proxies do not define
the number or locations of load-bearing units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PadGeometry:
  center_local: np.ndarray
  unit_positions_tln: np.ndarray
  unit_names: tuple[str, ...]
  rows: int
  cols: int

  def __post_init__(self):
    positions = np.asarray(self.unit_positions_tln, dtype=np.float64).copy()
    center = np.asarray(self.center_local, dtype=np.float64).copy()
    if positions.shape != (self.rows * self.cols, 3) or center.shape != (3,):
      raise ValueError("Pad unit coordinates must be N x 3 with a complete row/column grid")
    if not np.isfinite(positions).all() or not np.isfinite(center).all():
      raise ValueError("Nonfinite pad geometry")
    if self.rows < 2 or self.cols < 2:
      raise ValueError("Moment allocation requires at least a 2 x 2 unit array")
    if not np.allclose(positions.mean(axis=0), 0.0, atol=1e-9):
      raise ValueError("C must be the unit-array centroid")
    # Metric CAD exports round mate transforms and STL vertices. The supplied
    # rigid assembly has < 20 nm normal scatter; do not warp its actual sites
    # to remove that serialization error. The planar allocator projects only
    # this sub-0.1-micrometre residual, rejecting physically nonplanar arrays.
    if not np.allclose(positions[:, 2], 0.0, atol=1e-7):
      raise ValueError("Equal-stiffness planar allocation requires coplanar unit sites")
    positions[:, 2] = 0.0
    if abs(positions[:, 0] @ positions[:, 1]) > 1e-9:
      raise ValueError("This allocator requires uncoupled in-plane second moments")
    if min(positions[:, 0] @ positions[:, 0], positions[:, 1] @ positions[:, 1]) <= 0:
      raise ValueError("Degenerate unit array")
    positions.setflags(write=False)
    center.setflags(write=False)
    object.__setattr__(self, "unit_positions_tln", positions)
    object.__setattr__(self, "center_local", center)

  @property
  def count(self):
    return self.rows * self.cols

  @property
  def x(self):
    return self.unit_positions_tln[:, 0]

  @property
  def y(self):
    return self.unit_positions_tln[:, 1]

  @property
  def sxx(self):
    return float(self.x @ self.x)

  @property
  def syy(self):
    return float(self.y @ self.y)

  @property
  def srr(self):
    return self.sxx + self.syy

  @property
  def spans(self):
    return float(np.ptp(self.x)), float(np.ptp(self.y))

  def select_layout(self, unit_layout: str) -> PadGeometry:
    """Select load-bearing sites without shrinking the physical pad spans.

    Disabled units carry no preload or holding load. The four corners keep
    their original CAD names and lever arms, rather than becoming four
    adjacent units from the full array.
    """
    if unit_layout == "all":
      return self
    if unit_layout != "corners":
      raise ValueError(f"Unknown active unit layout: {unit_layout}")
    indices = [0, self.cols - 1, (self.rows - 1) * self.cols, self.count - 1]
    positions = self.unit_positions_tln[indices].copy()
    # CAD serialization leaves the corner centroid about 2 nm from C.
    # Project only this numerical residual onto the C-centered rectangle;
    # a physically offset array needs a different wrench allocator.
    residual = positions.mean(axis=0)
    if np.max(np.abs(residual)) > 1e-7:
      raise ValueError("Corner-unit centroid does not coincide with C")
    positions -= residual
    return PadGeometry(self.center_local, positions, tuple(self.unit_names[i] for i in indices), 2, 2)


def pad_geometry_from_sites(
  source, ee, robot_prefix="", *, rake_local_axis=(0.0, 0.0, -1.0), face_local_axis=(0.0, -1.0, 0.0), unit_layout="all"
):
  """Read a MuJoCo spec or model without compiling or stepping it.

  Sites must be attached directly to the pad body; C is its explicit
  ``<ee>_contact_center`` site. All actual units are discovered by name.
  """
  prefix = f"{robot_prefix}/" if robot_prefix else ""
  name = prefix + ee
  pattern = re.compile(re.escape(name) + r"_unit_(\d+)_(\d+)$")
  sites = source.sites if hasattr(source, "sites") else (source.site(i) for i in range(source.nsite))
  units = []
  for site in sites:
    match = pattern.fullmatch(site.name or "")
    if match:
      units.append((int(match[1]), int(match[2]), site))
  if not units:
    raise ValueError(f"No explicit unit sites for {name}")
  units.sort(key=lambda item: item[:2])
  rows, cols = max(r for r, _, _ in units) + 1, max(c for _, c, _ in units) + 1
  if [(r, c) for r, c, _ in units] != [(r, c) for r in range(rows) for c in range(cols)]:
    raise ValueError(f"Incomplete or duplicate unit site grid for {name}")
  center_site = source.site(name + "_contact_center")
  if center_site is None:
    raise ValueError(f"Missing contact-array center site for {name}")
  for site in [center_site] + [site for _, _, site in units]:
    parent_matches = (
      (site.parent.name == name) if hasattr(site, "parent") else (int(site.bodyid[0]) == source.body(name).id)
    )
    if not parent_matches:
      raise ValueError(f"Site {site.name} must be attached directly to pad body {name}")
  center = np.asarray(center_site.pos, dtype=np.float64)
  t = np.asarray(rake_local_axis, dtype=np.float64)
  n = -np.asarray(face_local_axis, dtype=np.float64)
  if min(np.linalg.norm(t), np.linalg.norm(n)) < 1e-9:
    raise ValueError("Pad rake and face axes must be nonzero")
  n = n / np.linalg.norm(n)
  t = t - n * (n @ t)
  if np.linalg.norm(t) < 1e-9:
    raise ValueError("Pad rake must have an in-plane component")
  t = t / np.linalg.norm(t)
  basis = np.stack((t, np.cross(n, t), n), axis=-1)
  positions = (np.asarray([s.pos for _, _, s in units]) - center) @ basis
  return PadGeometry(center, positions, tuple(s.name for _, _, s in units), rows, cols).select_layout(unit_layout)


def matching_pad_geometry(source, ee_names, robot_prefix="", **axes):
  """Require the same C-centered array on all pads sharing one model."""
  geometries = [pad_geometry_from_sites(source, ee, robot_prefix, **axes) for ee in ee_names]
  if not geometries:
    raise ValueError("No pad names supplied")
  reference = geometries[0]
  for ee, geometry in zip(ee_names, geometries):
    if (geometry.rows, geometry.cols) != (reference.rows, reference.cols) or not np.allclose(
      geometry.unit_positions_tln, reference.unit_positions_tln, atol=1e-9, rtol=0.0
    ):
      raise ValueError(f"Pad {ee} has a different unit array; one shared allocator cannot represent it")
  return reference
