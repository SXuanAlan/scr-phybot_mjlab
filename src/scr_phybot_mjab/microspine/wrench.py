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
  """Backend-neutral array arithmetic; the sole unweighted allocation formula."""
  n_units = unit_x.shape[-1]
  ft_u = f_t[..., None] / n_units - m_n[..., None] * unit_y / srr
  fl_u = f_l[..., None] / n_units + m_n[..., None] * unit_x / srr
  fn_u = f_n[..., None] / n_units - m_l[..., None] * unit_x / sxx + m_t[..., None] * unit_y / syy
  return ft_u, fl_u, fn_u


def distribute_wrench_to_units_weighted(
  f_t: torch.Tensor,
  f_l: torch.Tensor,
  f_n: torch.Tensor,
  m_t: torch.Tensor,
  m_l: torch.Tensor,
  m_n: torch.Tensor,
  unit_x: torch.Tensor,
  unit_y: torch.Tensor,
  weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Weighted (stiffness-share) bolt-group wrench distribution.

  Spring-field mode: unit i's effective stiffness is proportional to
  ``weights[..., i]`` (ticket x seating quality — Wang 2017 Eq 6-7 load
  sharing), and an EMPTY unit (weight 0) carries nothing: the wrench routes
  through the caught units only. The minimum-energy solution under weights
  w is ``F_i = w_i (A + B x r_i)``; with weighted-centroid-centered
  coordinates the normal/bending block and the tangential/twist block both
  decouple into closed forms — no batched linear solves.

  Shapes: loads (..., ), unit_x/unit_y (N,), weights (..., N). Returns
  per-unit (a, b, c) each (..., N). Degenerate all-empty pads (sum w ~ 0)
  fall back to the unweighted split — callers guard capacity separately.
  """
  eps = 1.0e-9
  w = weights.clamp_min(0.0)
  sw = w.sum(dim=-1, keepdim=True)
  degenerate = sw <= 1.0e-6
  w_safe = torch.where(degenerate.expand_as(w), torch.ones_like(w), w)
  sw = w_safe.sum(dim=-1, keepdim=True)
  x = unit_x.expand_as(w_safe)
  y = unit_y.expand_as(w_safe)
  xbar = (w_safe * x).sum(dim=-1, keepdim=True) / sw
  ybar = (w_safe * y).sum(dim=-1, keepdim=True) / sw
  xc = x - xbar
  yc = y - ybar
  sxx = (w_safe * xc * xc).sum(dim=-1)
  syy = (w_safe * yc * yc).sum(dim=-1)
  sxy = (w_safe * xc * yc).sum(dim=-1)
  # Normal + bending: c_i = w_i (alpha + beta xc_i + gamma yc_i).
  # alpha carries the net force; (beta, gamma) solve the 2x2 bending system
  # about the weighted centroid (moments re-expressed there).
  alpha = f_n / sw.squeeze(-1)
  m_l_c = -m_l - f_n * xbar.squeeze(-1)  # sum x c = -m_l -> centered rhs
  m_t_c = m_t - f_n * ybar.squeeze(-1)  # sum y c = m_t -> centered rhs
  det = (sxx * syy - sxy * sxy).clamp_min(eps)
  beta = (m_l_c * syy - m_t_c * sxy) / det
  gamma = (m_t_c * sxx - m_l_c * sxy) / det
  fn_u = w_safe * (alpha[..., None] + beta[..., None] * xc + gamma[..., None] * yc)
  # Tangential + twist: a_i = w_i (p - omega yc_i), b_i = w_i (q + omega
  # xc_i); the twist about the weighted centroid decouples the same way.
  srr = (sxx + syy).clamp_min(eps)
  m_n_c = m_n - (xbar.squeeze(-1) * f_l - ybar.squeeze(-1) * f_t)
  omega = m_n_c / srr
  p = f_t / sw.squeeze(-1)
  q = f_l / sw.squeeze(-1)
  ft_u = w_safe * (p[..., None] - omega[..., None] * yc)
  fl_u = w_safe * (q[..., None] + omega[..., None] * xc)
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
