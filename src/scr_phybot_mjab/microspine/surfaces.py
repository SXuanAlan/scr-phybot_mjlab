"""Rigid target frames and finite contact footprints for moving-object grip."""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from scr_phybot_mjab.microspine.contact import sphere_names


def conjugate(q):
  return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def rotate(q, v):
  v, xyz = torch.broadcast_tensors(v, q[..., 1:])
  t = 2 * torch.cross(xyz, v, dim=-1)
  return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def multiply(a, b):
  av, bv = torch.broadcast_tensors(a[..., 1:], b[..., 1:])
  return torch.cat(
    (
      a[..., :1] * b[..., :1] - (av * bv).sum(-1, keepdim=True),
      a[..., :1] * bv + b[..., :1] * av + torch.cross(av, bv, dim=-1),
    ),
    -1,
  )


def pad_face_vertices(model, cfg, ee, prefix="robot"):
  """Full proxy face footprint in pad coordinates, before runtime retraction."""
  vertices = []
  for name in sphere_names(ee, cfg.spheres_per_ee_rows, cfg.spheres_per_ee_cols):
    body, geom = model.body(f"{prefix}/{name}"), model.geom(f"{prefix}/{name}")
    rb, rg = np.empty(9), np.empty(9)
    mujoco.mju_quat2Mat(rb, body.quat)
    mujoco.mju_quat2Mat(rg, geom.quat)
    rb, rg = rb.reshape(3, 3), rg.reshape(3, 3)
    face = (rb @ rg).T @ np.asarray(cfg.face_local_axis)
    axis = int(np.argmax(abs(face)))
    other = [i for i in range(3) if i != axis]
    for a, b in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
      v = np.zeros(3)
      v[axis] = np.sign(face[axis]) * geom.size[axis]
      v[other] = np.asarray([a, b]) * geom.size[other]
      if geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
        v = np.asarray(cfg.face_local_axis) * geom.size[0]
      elif geom.type != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("Moving-surface contact proxies must be boxes or spheres")
      vertices.append(body.pos + rb @ (geom.pos + rg @ v))
  return np.asarray(vertices)


class MicrospineSurfaces:
  def __init__(self, env, cfg):
    self.bindings = cfg.ordered_surfaces()
    self.targets = []
    for binding in self.bindings:
      entity = env.scene[binding.entity_name]
      ids, _ = entity.find_bodies([binding.body_name])
      if len(ids) != 1:
        raise ValueError(f"Expected one target body: {binding.scoped_body}")
      geom = env.sim.mj_model.geom(binding.scoped_geom)
      if geom.bodyid.item() != env.sim.mj_model.body(binding.scoped_body).id:
        raise ValueError("Surface geom must belong to its declared target body")
      self.targets.append((entity, ids[0]))

    def tensor(rows):
      return torch.tensor(rows, device=env.device, dtype=torch.float32)

    self.local_point = tensor([s.point for s in self.bindings])
    self.local_normal = tensor([s.normal for s in self.bindings])
    self.local_tangent = tensor([s.tangent for s in self.bindings])
    self.extents = tensor([s.half_extents for s in self.bindings])

  def pose(self):
    return (
      torch.stack([e.data.body_link_pos_w[:, i] for e, i in self.targets], 1),
      torch.stack([e.data.body_link_quat_w[:, i] for e, i in self.targets], 1),
    )

  def frame(self):
    pos, quat = self.pose()
    point = pos + rotate(quat, self.local_point)
    normal = rotate(quat, self.local_normal)
    tangent = rotate(quat, self.local_tangent)
    return point, normal, tangent, torch.cross(normal, tangent, dim=-1)

  def velocity_at(self, points):
    pos, _ = self.pose()
    linear = torch.stack([e.data.body_link_lin_vel_w[:, i] for e, i in self.targets], 1)
    angular = torch.stack([e.data.body_link_ang_vel_w[:, i] for e, i in self.targets], 1)
    return linear + torch.cross(angular, points - pos, dim=-1)

  def contains(self, vertices, tolerance=1e-5):
    point, _, tangent, bitangent = self.frame()
    delta = vertices - point.unsqueeze(-2)
    uv = torch.stack(((delta * tangent.unsqueeze(-2)).sum(-1), (delta * bitangent.unsqueeze(-2)).sum(-1)), -1)
    return (uv.abs() <= self.extents[None, :, None, :] + tolerance).all(-1).all(-1)
