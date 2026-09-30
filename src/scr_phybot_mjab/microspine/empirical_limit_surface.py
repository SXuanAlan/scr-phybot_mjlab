"""Protocol-conditioned single-unit failure distributions, in newtons.

The normal pull-off threshold is NOT a radial Wang surface or a mean force.
One strength quantile and one loading-success variate are retained per unit
for an entire engagement. Spatial interpolation couples equal quantile ranks
across tested shear/lateral conditions; that coupling, lateral reflection,
independent units and first-unit pad release are modelling assumptions.

No extrapolation outside the measured shear/lateral domain is allowed. Censored tails
are capped at their observed lower bound and flagged, never called failures.
The table describes the 10 N seating / 3 s normal-step protocol. Seating is
a minimum engagement preload, not a compression failure threshold. Normal
capacity is an upper pull-off bound; additional compression does not increase
that capacity. No structural compression limit, fatigue, dwell-time or dynamic
hazard model is identified by these pull-off trials.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scr_phybot_mjab.assets.microspine import EMPIRICAL_UNIT_TABLE
from scr_phybot_mjab.microspine.wrench import shift_wrench_reference as shift_wrench_reference

DEFAULT_TABLE = EMPIRICAL_UNIT_TABLE


class EmpiricalLimitSurface:
  """Device-resident inverse-CDF table; query path has no CPU synchronization."""

  def __init__(self, path: str | Path = DEFAULT_TABLE, device="cpu"):
    self.path = Path(path)
    with np.load(self.path, allow_pickle=False) as data:
      schema = int(data["schema_version"])
      if schema not in (1, 2):
        raise ValueError("Unsupported empirical limit-surface schema")
      self.arrays = {key: data[key].copy() for key in data.files}
    a = self.arrays
    self.s_min, self.s_max = float(a["shear_n"][0]), float(a["shear_n"][-1])
    self.l_max = float(a["lateral_n"][-1])
    self.ds = float(a["shear_n"][1] - a["shear_n"][0])
    self.dl = float(a["lateral_n"][1] - a["lateral_n"][0])
    # Schema 1 mislabeled the seating protocol as a lower normal-force bound.
    # Read old tables as protocol metadata without restoring that hold limit.
    self.protocol_preload_n = float(a["protocol_preload_n"]) if schema == 2 else -float(a["normal_min_n"])
    if not np.isfinite(self.protocol_preload_n) or self.protocol_preload_n <= 0:
      raise ValueError("Empirical protocol preload must be finite and positive")
    self.ns, self.nl, self.nq = a["normal_quantile_n"].shape
    if not (np.allclose(np.diff(a["shear_n"]), self.ds) and np.allclose(np.diff(a["lateral_n"]), self.dl)):
      raise ValueError("Runtime lookup requires regular shear/lateral axes")
    valid = a["supported"].astype(bool)
    if not np.all(np.isfinite(a["normal_quantile_n"][valid])):
      raise ValueError("Nonfinite threshold in supported empirical cell")
    if np.any(np.diff(a["normal_quantile_n"][valid], axis=-1) < -1e-5):
      raise ValueError("Empirical inverse CDF is not monotone")
    self.q = torch.as_tensor(a["normal_quantile_n"], device=device)
    self.early = torch.as_tensor(a["early_probability"], device=device)
    self.censored = torch.as_tensor(a["censored_tail"], device=device)
    self.supported = torch.as_tensor(a["supported"], device=device)

  def query(self, shear, lateral, strength_u):
    """Return (normal upper bound, early probability, supported, tail-capped).

    Quantile bins use midpoints of a uniform 1024-bin partition (probability
    resolution <= 1/1024), with bilinear interpolation in measured S, |L|.
    The returned support mask MUST be applied; clamping only makes gathers
    safe and must never be interpreted as extrapolated capacity.
    """
    s = torch.nan_to_num((shear - self.s_min) / self.ds).clamp(0, self.ns - 1)
    l = torch.nan_to_num(lateral.abs() / self.dl).clamp(0, self.nl - 1)
    i = s.floor().long().clamp(max=self.ns - 2)
    j = l.floor().long().clamp(max=self.nl - 2)
    us, ul = s - i, l - j
    k = (strength_u.clamp(0, 1) * self.nq).long().clamp(max=self.nq - 1)
    cap, early = torch.zeros_like(s), torch.zeros_like(s)
    support = (
      (shear >= self.s_min)
      & (shear <= self.s_max)
      & (lateral.abs() <= self.l_max)
      & torch.isfinite(shear)
      & torch.isfinite(lateral)
    )
    tail = torch.zeros_like(support)
    for di, dj, weight in (
      (0, 0, (1 - us) * (1 - ul)),
      (1, 0, us * (1 - ul)),
      (0, 1, (1 - us) * ul),
      (1, 1, us * ul),
    ):
      ii, jj = i + di, j + dj
      cap = cap + weight * self.q[ii, jj, k]
      early = early + weight * self.early[ii, jj]
      contributing = weight > 1e-7
      support = support & (~contributing | self.supported[ii, jj])
      tail = tail | (contributing & self.censored[ii, jj, k])
    return cap, early, support, tail

  def query_numpy(self, shear, lateral, strength_u):
    """Same lookup for offline audits/plots, without GPU or simulation."""
    shear, lateral, u = np.broadcast_arrays(shear, lateral, strength_u)
    s = np.clip(np.nan_to_num((shear - self.s_min) / self.ds), 0, self.ns - 1)
    l = np.clip(np.nan_to_num(np.abs(lateral) / self.dl), 0, self.nl - 1)
    i = np.minimum(s.astype(int), self.ns - 2)
    j = np.minimum(l.astype(int), self.nl - 2)
    us, ul = s - i, l - j
    k = np.minimum((np.clip(u, 0, 1) * self.nq).astype(int), self.nq - 1)
    cap, early = np.zeros_like(s), np.zeros_like(s)
    support = (shear >= self.s_min) & (shear <= self.s_max) & (np.abs(lateral) <= self.l_max)
    tail = np.zeros_like(support)
    a = self.arrays
    for di, dj, weight in (
      (0, 0, (1 - us) * (1 - ul)),
      (1, 0, us * (1 - ul)),
      (0, 1, (1 - us) * ul),
      (1, 1, us * ul),
    ):
      ii, jj = i + di, j + dj
      cap += weight * a["normal_quantile_n"][ii, jj, k]
      early += weight * a["early_probability"][ii, jj]
      support &= (weight <= 1e-7) | a["supported"][ii, jj]
      tail |= (weight > 1e-7) & a["censored_tail"][ii, jj, k]
    return cap, early, support, tail

  def margins(self, ft, fl, fn, strength_u, loading_u):
    cap, early, domain, tail = self.query(ft, fl, strength_u)
    domain = domain & torch.isfinite(fn)
    early_failed = loading_u < early
    margin = (cap - fn) / (cap.abs() + self.protocol_preload_n).clamp_min(1.0)
    margin = torch.where(domain & ~early_failed, margin, -torch.ones_like(margin))
    return margin, cap, domain, early_failed, tail
