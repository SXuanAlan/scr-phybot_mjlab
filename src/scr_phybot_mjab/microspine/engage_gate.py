"""Kinematic engage gates: the engage frame, pad orientation and the virtual spine field."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.utils.lab_api.math import quat_apply

from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.contact import (
  contact_face_offset,
  pad_contact_center,
  pad_contact_face_centers,
  resolve_sphere_geom_ids,
  sphere_names,
  virtual_spine_fraction,
)
from scr_phybot_mjab.microspine.quat import rotate, rotate_axis
from scr_phybot_mjab.microspine.surfaces import MicrospineSurfaces, pad_face_vertices

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def engage_frame(
  ee_quats: Sequence[torch.Tensor], rake_local_axis: torch.Tensor, n_out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """World-frame engage basis (t̂, l̂, n̂_out), each (E, H, 3).

  t̂ is the rake direction the spines load along ("down the wall"): each
  pad's rake axis projected onto the surface plane. ``n_out`` is the outward
  surface normal, (E, 1, 3) for a wall or (E, H, 3) per pad, so a positive
  f_n is pull-off tension.
  """
  raw = torch.stack([rotate_axis(quat, rake_local_axis) for quat in ee_quats], dim=1)
  t = raw - (raw * n_out).sum(dim=-1, keepdim=True) * n_out
  t_norm = torch.linalg.norm(t, dim=-1, keepdim=True)
  # Degenerate rake (load axis ⟂ wall plane): fall back to world -z projected.
  fallback = torch.tensor([0.0, 0.0, -1.0], device=raw.device).expand_as(t)
  fb = fallback - (fallback * n_out).sum(dim=-1, keepdim=True) * n_out
  fb = fb / torch.linalg.norm(fb, dim=-1, keepdim=True).clamp_min(1.0e-6)
  t_hat = torch.where(t_norm > 1.0e-6, t / t_norm.clamp_min(1.0e-9), fb)
  n_hat = n_out.expand_as(t_hat)
  l_hat = torch.cross(n_hat, t_hat, dim=-1)
  return t_hat, l_hat, n_hat


def to_engage_frame(
  vector: torch.Tensor, basis: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Components of world vectors (E, H, 3) along an engage basis, three (E, H)."""
  t_hat, l_hat, n_hat = basis
  return (vector * t_hat).sum(dim=-1), (vector * l_hat).sum(dim=-1), (vector * n_hat).sum(dim=-1)


def face_wall_cos(
  ee_quats: Sequence[torch.Tensor], face_local_axis: torch.Tensor, inward: torch.Tensor
) -> torch.Tensor:
  """Cosine between each pad's face axis and the inward surface normal, (E, H).

  ``inward`` is (E, 3) for a wall or (E, H, 3) per pad. Near 1 the pad faces
  into the surface; near 0 it is side-on.
  """
  per_pad = inward.dim() == 3
  result = torch.zeros(inward.shape[0], len(ee_quats), device=inward.device)
  for h, quat in enumerate(ee_quats):
    face = rotate_axis(quat, face_local_axis)
    result[:, h] = (face * (inward[:, h] if per_pad else inward)).sum(dim=-1)
  return result


class SpineField:
  """Engaged fraction of a kinematic virtual spine field over each pad face (E, H).

  A ``virtual_spine_grid`` spanning the four corner contact proxies stands
  in for the real pad's individually suspended spines; a spine counts as
  engaged when the wall is within ``virtual_spine_travel`` of it. Purely
  kinematic: the spine suspension is force-free over its travel (Wang et
  al. 2017), so reach is the whole question. Geometry resolves lazily on
  the first call, against the live simulation.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    asset,
    ee_body_ids: list[int],
    mcfg: MicrospineCfg,
    asset_name: str,
    surfaces: MicrospineSurfaces | None,
    wall_inward_normal: Callable[[], torch.Tensor],
  ):
    self._env = env
    self._asset = asset
    self._ee_body_ids = ee_body_ids
    self._mcfg = mcfg
    self._asset_name = asset_name
    self._surfaces = surfaces
    self._wall_inward_normal = wall_inward_normal
    self._num_ee = len(mcfg.ee_body_names)
    # Wall surfaces: corner proxy bodies and their face offsets.
    self._corner_body_ids: torch.Tensor | None = None
    self._corner_offsets: torch.Tensor | None = None
    self._sphere_offsets: torch.Tensor | None = None
    self._corner_radii: torch.Tensor | None = None
    self._wall_geom_id: int | None = None
    self._wall_surface_offset = 0.0
    # Moving grasp surfaces: pad-local face geometry.
    self._surface_corners: torch.Tensor | None = None
    self._surface_vertices: torch.Tensor | None = None
    self.surface_centers: torch.Tensor | None = None
    self.surface_covered: torch.Tensor | None = None
    """(E, H) bool on moving surfaces: the surface still covers the pad face."""

  def fraction(self) -> torch.Tensor:
    if self._surfaces is not None:
      return self._moving_surface_fraction()
    if self._corner_body_ids is None:
      self._resolve_wall_geometry()
    corners = self.corners_world(for_reach=True)
    # Off-surface direction toward the robot is the engage frame's n_out; the
    # surface point is the wall geom's origin pushed out by the type-resolved
    # face offset (0 for a plane, the half-extent for a box).
    n_out = -self._wall_inward_normal()
    wall_point = wp.to_torch(self._env.sim.wp_data.geom_xpos)[:, self._wall_geom_id] + self._wall_surface_offset * n_out
    return virtual_spine_fraction(
      corners,
      wall_point.unsqueeze(1).expand(-1, self._num_ee, -1),
      n_out.unsqueeze(1).expand(-1, self._num_ee, -1),
      0.0,  # Corners already lie on the contact surface, including for boxes.
      self._mcfg.virtual_spine_grid,
      self._mcfg.virtual_spine_travel,
    )

  def corners_world(self, *, for_reach: bool = False) -> torch.Tensor:
    """Corner proxy surface points (E, H, 4, 3), unaffected by latched retraction.

    Corner positions come from the proxy child BODIES, whose world poses are
    latch-invariant (retraction moves the geoms within those bodies). Boxes
    use their face centers. Sphere proxies use their closest point to the
    wall for reach checks (``for_reach``) and their pad-facing poles
    otherwise.
    """
    if self._corner_body_ids is None:
      self._resolve_wall_geometry()
    num_envs = self._env.num_envs
    ids = self._corner_body_ids
    offsets = self._corner_offsets
    if for_reach:
      offsets = offsets - self._sphere_offsets
    points = self._asset.data.body_link_pos_w[:, ids] + quat_apply(
      self._asset.data.body_link_quat_w[:, ids],
      offsets.unsqueeze(0).expand(num_envs, -1, -1),
    )
    if for_reach:
      points = points + self._corner_radii[None, :, None] * self._wall_inward_normal()[:, None, :]
    return points.reshape(num_envs, self._num_ee, 4, 3)

  def _resolve_wall_geometry(self) -> None:
    rows = self._mcfg.spheres_per_ee_rows
    cols = self._mcfg.spheres_per_ee_cols
    if rows < 2 or cols < 2:
      raise ValueError(
        f"The virtual spine field needs a >=2x2 contact-proxy grid to span the pad face; got {rows}x{cols}."
      )
    device = self._env.device
    corner_ids: list[int] = []
    offsets, sphere_offsets, radii = [], [], []
    mj_model = self._env.sim.mj_model
    geom_lists = resolve_sphere_geom_ids(mj_model, self._mcfg)
    for ee_index, ee in enumerate(self._mcfg.ee_body_names):
      names = sphere_names(ee, rows, cols)
      for corner in (0, cols - 1, (rows - 1) * cols, rows * cols - 1):
        name = names[corner]
        ids, _ = self._asset.find_bodies([name])
        if len(ids) != 1:
          raise ValueError(f"Corner proxy body '{name}' not found.")
        corner_ids.append(ids[0])
        gid = geom_lists[ee_index][corner]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, mj_model.body_quat[mj_model.geom_bodyid[gid]])
        face_local = rotation.reshape(3, 3).T @ np.asarray(self._mcfg.face_local_axis)
        offset = contact_face_offset(
          mj_model.geom_type[gid], mj_model.geom_size[gid], face_local, mj_model.geom_quat[gid]
        )
        offsets.append(mj_model.geom_pos[gid] + offset)
        is_sphere = mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
        sphere_offsets.append(offset if is_sphere else np.zeros(3))
        radii.append(float(mj_model.geom_size[gid, 0]) if is_sphere else 0.0)
    self._corner_body_ids = torch.tensor(corner_ids, dtype=torch.long, device=device)
    self._corner_offsets = torch.tensor(np.asarray(offsets), dtype=torch.float32, device=device)
    self._sphere_offsets = torch.tensor(np.asarray(sphere_offsets), dtype=torch.float32, device=device)
    self._corner_radii = torch.tensor(radii, dtype=torch.float32, device=device)
    wall_gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, self._mcfg.wall_geom_name)
    if wall_gid < 0:
      raise ValueError(f"Wall geom '{self._mcfg.wall_geom_name}' not found for the virtual spine field.")
    self._wall_geom_id = wall_gid
    # The climbable surface: a plane's surface passes through its frame
    # origin; a box's is the face half-extent out along the wall normal. The
    # offset is resolved once from the live rotation, since the face-to-centre
    # relation is rigid.
    gtype = int(mj_model.geom_type[wall_gid])
    if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
      self._wall_surface_offset = 0.0
    elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
      rot = wp.to_torch(self._env.sim.wp_data.geom_xmat)[0, wall_gid].reshape(3, 3)
      n0 = -self._wall_inward_normal()[0]
      local = rot.T @ n0
      axis = int(torch.argmax(torch.abs(local)).item())
      self._wall_surface_offset = float(mj_model.geom_size[wall_gid][axis])
    else:
      raise ValueError(
        f"Wall geom '{self._mcfg.wall_geom_name}' has unsupported type {gtype} for the virtual spine field "
        "(plane or box)."
      )

  def _moving_surface_fraction(self) -> torch.Tensor:
    if self._surface_corners is None:
      model = self._env.sim.mj_model
      ees = self._mcfg.ee_body_names

      def tensor(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=self._env.device)

      self._surface_corners = tensor([pad_contact_face_centers(model, self._mcfg, ee, self._asset_name) for ee in ees])
      self._surface_vertices = tensor([pad_face_vertices(model, self._mcfg, ee, self._asset_name) for ee in ees])
      self.surface_centers = tensor([pad_contact_center(model, self._mcfg, ee, self._asset_name) for ee in ees])
    pos = self._asset.data.body_link_pos_w[:, self._ee_body_ids]
    quat = self._asset.data.body_link_quat_w[:, self._ee_body_ids]
    corners = pos.unsqueeze(2) + rotate(quat.unsqueeze(2), self._surface_corners)
    vertices = pos.unsqueeze(2) + rotate(quat.unsqueeze(2), self._surface_vertices)
    point, normal, _, _ = self._surfaces.frame()
    self.surface_covered = self._surfaces.contains(vertices)
    fraction = virtual_spine_fraction(
      corners, point, normal, 0.0, self._mcfg.virtual_spine_grid, self._mcfg.virtual_spine_travel
    )
    return fraction * self.surface_covered
