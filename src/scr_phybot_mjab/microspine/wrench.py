"""C-referenced wrench mechanics shared by runtime and offline analysis."""

from __future__ import annotations

import numpy as np
import torch

from scr_phybot_mjab.microspine.pad_geometry import PadGeometry


def distribute_wrench_to_units(
  f_t: torch.Tensor,
  f_l: torch.Tensor,
  f_n: torch.Tensor,
  m_t: torch.Tensor,
  m_l: torch.Tensor,
  m_n: torch.Tensor,
  unit_x: torch.Tensor,
  unit_y: torch.Tensor,
  sxx: float,
  syy: float,
  srr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Distribute a pad wrench onto spine units (rigid pad, equal stiffness).

  Unit i sits at ``r_i = x_i t̂ + y_i l̂`` in the pad plane and carries
  ``F_i = a_i t̂ + b_i l̂ + c_i n̂``. The equal-stiffness (least-norm)
  bolt-group solution reproducing ``sum F_i = F`` and ``sum r_i x F_i = M``
  is, with ``sxx = sum x², syy = sum y², srr = sxx + syy``:

    a_i = f_t/N − m_n·y_i/srr          (twist shares tangentially, ∝ radius)
    b_i = f_l/N + m_n·x_i/srr
    c_i = f_n/N − m_l·x_i/sxx + m_t·y_i/syy   (bending: linear in position)

  (cross products in the engage frame: m_t = Σ y c, m_l = −Σ x c,
  m_n = Σ (x b − y a)). Inputs (..., ) broadcast against unit_x/unit_y of
  shape (..., N) or (N,); returns per-unit (a, b, c), each (..., N).
  """
  return _distribute_components(f_t, f_l, f_n, m_t, m_l, m_n, unit_x, unit_y, sxx, syy, srr)


def _distribute_components(f_t, f_l, f_n, m_t, m_l, m_n, unit_x, unit_y, sxx, syy, srr):
  """Backend-neutral array arithmetic; the sole allocation formula."""
  n_units = unit_x.shape[-1]
  ft_u = f_t[..., None] / n_units - m_n[..., None] * unit_y / srr
  fl_u = f_l[..., None] / n_units + m_n[..., None] * unit_x / srr
  fn_u = f_n[..., None] / n_units - m_l[..., None] * unit_x / sxx + m_t[..., None] * unit_y / syy
  return ft_u, fl_u, fn_u


def distribute_wrench_to_units_numpy(wrench, geometry: PadGeometry):
  """Apply the runtime formula to (..., 6) NumPy wrenches, returning (..., N, 3)."""
  w = np.asarray(wrench, dtype=np.float64)
  if w.shape[-1] != 6:
    raise ValueError("Expected [Ft, Fl, Fn, Mt, Ml, Mn] about C")
  components = _distribute_components(
    *np.moveaxis(w, -1, 0), geometry.x, geometry.y, geometry.sxx, geometry.syy, geometry.srr
  )
  return np.stack(components, axis=-1)


def unit_wrench_matrix(geometry: PadGeometry):
  """Linear operator (N, 3, 6), constructed by applying the SAME allocator."""
  return distribute_wrench_to_units_numpy(np.eye(6), geometry).transpose(1, 2, 0)


def shift_wrench_reference(force, moment_at_b, b_to_c):
  """Same axes: M_C = M_B - (C-B) cross F."""
  return moment_at_b - torch.linalg.cross(b_to_c, force, dim=-1)
