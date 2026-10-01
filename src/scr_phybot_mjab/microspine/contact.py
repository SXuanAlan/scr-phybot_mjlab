"""Pad contact-proxy geometry: proxy names, face offsets, pad centres and the spine field."""

from __future__ import annotations

import mujoco
import numpy as np

from scr_phybot_mjab.microspine.config import MicrospineCfg


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
