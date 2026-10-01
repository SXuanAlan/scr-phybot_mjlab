"""Legacy Wang unit-boundary calculations for offline empirical comparisons."""

import numpy as np
import torch

from scr_phybot_mjab.tools.microspine.wang_limit_surface import build_capacity_lut, capacity_ratio


class WangUnit:
  def __init__(self, cfg):
    self.cfg = cfg
    self.n = cfg.unit_rows_rake * cfg.unit_cols_lateral
    self.lut = build_capacity_lut(cfg, "cpu")

  def normal_boundary(self, s, l, pad_f0):
    """Tension boundary at fixed unit S,L; NaN if in-plane load already fails.

    The old anchor model clamps compression out of the force criterion;
    it therefore cannot predict the negative-Fn release boundaries in the
    data. Missing tensile solutions must not be substituted with zero.
    """
    s, l = np.broadcast_arrays(np.asarray(s, float), np.asarray(l, float))
    s, l = torch.tensor(s), torch.tensor(l)
    f0 = pad_f0 / self.n

    def occupancy(z):
      cap = f0 * capacity_ratio(self.lut, s, l, z).clamp_min(self.cfg.wang_r_floor)
      return torch.sqrt(s * s + l * l + z * z) / cap

    lower, upper = torch.zeros_like(s), torch.full_like(s, f0)
    valid = occupancy(lower) <= 1
    for _ in range(45):
      mid = (lower + upper) / 2
      inside = occupancy(mid) <= 1
      lower = torch.where(inside, mid, lower)
      upper = torch.where(inside, upper, mid)
    residual = (occupancy(lower) - 1)[valid].abs()
    if residual.numel():
      assert float(residual.max()) < 1e-7
    return torch.where(valid, lower, torch.full_like(lower, float("nan"))).numpy()
