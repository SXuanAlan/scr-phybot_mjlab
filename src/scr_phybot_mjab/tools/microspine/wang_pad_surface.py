"""The Wang-2017 whole-pad limit surface under per-unit moment billing (paper comparison model).

The pad's 6D anchor wrench is distributed onto the 3x4 unit grid with
rigid-pad, equal-stiffness bolt-group statics (distribute_wrench_to_units),
and every unit's 3D load is tested against its F0/N share of the closed
Wang surface; the weakest unit releases the pad. The distribution is linear
in the wrench and the unit criterion is positively homogeneous (radial LUT
ratio), so the pad-level admissible set

  W = { w in R^6 : max_i |f_i(w)|_+ / cap_i(dir f_i(w)) <= 1 }

is an exact star-shaped body whose boundary along any wrench ray is
computable in closed form per direction.

Conventions: engage frame (t = rake, down-wall; l = lateral; n = outward
normal). m_l is the pitch/peel bending about l, m_t the roll bending about
t, m_n the twist about the normal. Compression (f_n < 0) rides the anchor
for free. Everything scales linearly with the per-latch F0 draw.
"""

from __future__ import annotations

import torch

from scr_phybot_mjab.microspine.wrench import distribute_wrench_to_units
from scr_phybot_mjab.tools.microspine.wang_limit_surface import (
  WangPadCfg,
  build_capacity_lut,
  capacity_ratio,
  climb_wang_pad_cfg,
)

HANG_FT, HANG_FN = 70.0, 14.0  # 90-deg frame-0 per-hand hang load [N].


def load_limit_surface_cfg(use_defaults: bool) -> WangPadCfg:
  return WangPadCfg() if use_defaults else climb_wang_pad_cfg()


def unit_grid(cfg) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
  """Mirror of the affordance's per-unit grid construction (CPU float32)."""
  xs = torch.linspace(-0.5, 0.5, cfg.unit_rows_rake) * cfg.pad_span_rake_m
  ys = torch.linspace(-0.5, 0.5, cfg.unit_cols_lateral) * cfg.pad_span_lateral_m
  gx, gy = torch.meshgrid(xs, ys, indexing="ij")
  ux, uy = gx.reshape(-1), gy.reshape(-1)
  sxx = float((ux**2).sum())
  syy = float((uy**2).sum())
  return ux, uy, sxx, syy, sxx + syy


class PadSurface:
  """Occupancy field of the equivalent pad surface (occ = 1 on the boundary)."""

  def __init__(self, cfg, f0: float):
    self.cfg = cfg
    self.f0 = f0
    self.lut = build_capacity_lut(cfg, "cpu")
    self.ux, self.uy, self.sxx, self.syy, self.srr = unit_grid(cfg)
    self.n_units = self.ux.shape[0]

  def occupancy(
    self,
    f_t: torch.Tensor,
    f_l: torch.Tensor,
    f_n: torch.Tensor,
    m_t: torch.Tensor,
    m_l: torch.Tensor,
    m_n: torch.Tensor,
  ) -> torch.Tensor:
    """max_i load_i / cap_i over the unit grid; broadcast inputs, any shape.

    Positively homogeneous of degree 1: the boundary point along ray w is
    exactly w / occupancy(w).
    """
    f_t, f_l, f_n, m_t, m_l, m_n = torch.broadcast_tensors(f_t, f_l, f_n, m_t, m_l, m_n)
    a, b, c = distribute_wrench_to_units(
      f_t,
      f_l,
      f_n,
      m_t,
      m_l,
      m_n,
      self.ux,
      self.uy,
      self.sxx,
      self.syy,
      self.srr,
    )
    cap = (self.f0 / self.n_units) * capacity_ratio(self.lut, a, b, c).clamp_min(self.cfg.wang_r_floor)
    load = torch.sqrt(a**2 + b**2 + c.clamp_min(0.0) ** 2)
    return (load / cap).amax(dim=-1)

  def occupancy6(self, w: torch.Tensor) -> torch.Tensor:
    """Occupancy of stacked wrenches (..., 6) ordered (t, l, n, mt, ml, mn)."""
    return self.occupancy(w[..., 0], w[..., 1], w[..., 2], w[..., 3], w[..., 4], w[..., 5])

  def occupancy_aggregate(self, f_t: torch.Tensor, f_l: torch.Tensor, f_n: torch.Tensor) -> torch.Tensor:
    """Force-only aggregate radial test (the pre-per_unit envelope)."""
    cap = self.f0 * capacity_ratio(self.lut, f_t, f_l, f_n).clamp_min(self.cfg.wang_r_floor)
    load = torch.sqrt(f_t**2 + f_l**2 + f_n.clamp_min(0.0) ** 2)
    return load / cap

  def moment_capacity(
    self,
    axis: str,
    f_t: torch.Tensor,
    f_n: float = 0.0,
    m_max: float = 30.0,
    n_m: int = 3000,
  ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """|m|* on one moment axis vs rake-shear preload (first occ=1 crossing).

    Returns (capacity per f_t, m grid, islands): capacity is 0 where even
    m = 0 is inadmissible; islands counts f_t rows whose admissible set in m
    is non-contiguous (re-entry beyond the first crossing — none expected,
    but the scan reports rather than assumes).
    """
    m = torch.linspace(0.0, m_max, n_m)
    ft2, m2 = torch.meshgrid(f_t, m, indexing="ij")
    zeros = torch.zeros_like(ft2)
    wrench = {"m_t": zeros, "m_l": zeros, "m_n": zeros}
    wrench[axis] = m2
    occ = self.occupancy(
      ft2,
      zeros,
      torch.full_like(ft2, float(f_n)),
      wrench["m_t"],
      wrench["m_l"],
      wrench["m_n"],
    )
    admissible = occ <= 1.0
    lead = admissible.long().cumprod(dim=1)  # leading admissible run
    cap = m[lead.sum(dim=1).clamp(min=1) - 1] * admissible[:, 0]
    islands = int(((admissible.long() - lead) > 0).any(dim=1).sum())
    return cap, m, islands
