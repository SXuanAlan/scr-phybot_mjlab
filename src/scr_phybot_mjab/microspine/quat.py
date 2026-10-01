"""Batched wxyz quaternion helpers."""

from __future__ import annotations

import torch


def conjugate(q: torch.Tensor) -> torch.Tensor:
  return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate vectors ``v`` (..., 3) by quaternions ``q`` (..., 4), broadcasting."""
  v, xyz = torch.broadcast_tensors(v, q[..., 1:])
  t = 2 * torch.cross(xyz, v, dim=-1)
  return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Quaternion product ``a * b``."""
  av, bv = torch.broadcast_tensors(a[..., 1:], b[..., 1:])
  return torch.cat(
    (
      a[..., :1] * b[..., :1] - (av * bv).sum(-1, keepdim=True),
      a[..., :1] * bv + b[..., :1] * av + torch.cross(av, bv, dim=-1),
    ),
    -1,
  )


def rotate_axis(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """Rotate one vector by a batch of quaternions, (E, 4) x (3,) -> (E, 3).

  Written component-wise; it rounds differently from ``rotate`` (whose
  cross products may fuse multiply-adds), and the engage gate's thresholds
  are evaluated with this form.
  """
  w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
  vx, vy, vz = vec[0], vec[1], vec[2]
  tx = 2.0 * (y * vz - z * vy)
  ty = 2.0 * (z * vx - x * vz)
  tz = 2.0 * (x * vy - y * vx)
  rx = vx + w * tx + (y * tz - z * ty)
  ry = vy + w * ty + (z * tx - x * tz)
  rz = vz + w * tz + (x * ty - y * tx)
  return torch.stack([rx, ry, rz], dim=-1)
