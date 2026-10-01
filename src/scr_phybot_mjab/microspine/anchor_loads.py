"""Anchor loads read back from the weld equalities' constraint rows."""

from __future__ import annotations

import torch

# ConstraintType is not re-exported at package level; the efc rows we filter
# are written by mujoco_warp kernels, so its enum is the source of truth.
from mujoco_warp._src.types import ConstraintType

from scr_phybot_mjab.microspine.quat import conjugate, multiply, rotate

_EFC_EQUALITY = int(ConstraintType.EQUALITY)


def weld_rotational_force_to_world_moment(
  rotational_force: torch.Tensor,
  body_quat_w: torch.Tensor,
  anchor_quat_w: torch.Tensor,
  torquescale: float,
  target_quat_w: torch.Tensor | None = None,
) -> torch.Tensor:
  """Convert mujoco_warp weld multipliers to physical world moments.

  The last three weld entries in ``efc.force`` are multipliers in
  quaternion-vector constraint coordinates, not N*m. Physical moment is
  ``J_rot.T @ multiplier``. The runtime kernel has
  ``J_rot = 0.5 * torquescale * (w I - skew(v))`` for residual quaternion
  ``(w, v)``. At a satisfied weld this reduces exactly to
  ``0.5 * torquescale * multiplier``.
  """
  anchor_conj = torch.cat([anchor_quat_w[..., 0:1], -anchor_quat_w[..., 1:4]], dim=-1)
  q_err = multiply(body_quat_w, anchor_conj)
  if target_quat_w is not None:
    # The quaternion residual/multipliers are expressed in body2's axes.
    q_err = multiply(multiply(conjugate(target_quat_w), q_err), target_quat_w)
  w, v = q_err[..., 0:1], q_err[..., 1:4]
  moment = 0.5 * float(torquescale) * (w * rotational_force + torch.cross(v, rotational_force, dim=-1))
  return moment if target_quat_w is None else rotate(target_quat_w, moment)


class WeldRows:
  """Gathers each pad weld's six efc rows from the solver data.

  A weld contributes 6 consecutive efc rows: 3 force rows with the world
  x/y/z positional Jacobian of the pad point, then 3 quaternion-constraint
  multipliers. Only constraint counters are zeroed each step; rows past
  ``ne`` are stale garbage and EQUALITY == 0 would happily match them, so
  the ``row < ne`` mask is mandatory. ``ne`` can also exceed the written
  rows on njmax overflow, hence the clamp.
  """

  def __init__(self, eq_ids: torch.Tensor, num_envs: int, device: torch.device | str):
    self._eq_ids = eq_ids
    self._num_envs = num_envs
    self._num_ee = eq_ids.numel()
    self._device = device
    self._row_arange: torch.Tensor | None = None  # Sized to njmax on first read.
    self._comp_arange = torch.arange(3, device=device)

  def read(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(found (E, H), force rows (E, H, 3), rotational rows (E, H, 3)).

    Rows of a weld that is not in the solver read as row 0; mask them with
    ``found``.
    """
    njmax = int(data.njmax)
    if self._row_arange is None:
      self._row_arange = torch.arange(njmax, device=self._device)
    ne = data.ne[:].to(torch.long).clamp(max=njmax)
    valid = self._row_arange.unsqueeze(0) < ne.unsqueeze(1)
    is_eq = valid & (data.efc.type[:, :njmax] == _EFC_EQUALITY)
    efc_id = data.efc.id[:, :njmax]
    efc_force = data.efc.force[:, :njmax]
    # (E, njmax, H) row-vs-equality match: the first matching row + [0, 1, 2]
    # addresses the force rows and + [3, 4, 5] the rotational ones.
    match = is_eq.unsqueeze(-1) & (efc_id.unsqueeze(-1) == self._eq_ids.view(1, 1, -1))
    found = match.any(dim=1)
    first = match.long().argmax(dim=1)  # 0 where not found.
    rows = (first.unsqueeze(-1) + self._comp_arange).clamp(max=njmax - 1)
    force = efc_force.gather(1, rows.reshape(self._num_envs, -1)).reshape(self._num_envs, self._num_ee, 3)
    trows = (first.unsqueeze(-1) + self._comp_arange + 3).clamp(max=njmax - 1)
    rotational = efc_force.gather(1, trows.reshape(self._num_envs, -1)).reshape(self._num_envs, self._num_ee, 3)
    return found, force, rotational
