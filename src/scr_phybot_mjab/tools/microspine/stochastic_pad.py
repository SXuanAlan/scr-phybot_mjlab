"""Offline stochastic pad path boundaries from the shared unit mechanics.

Geometry comes from asset sites, and all force/moment coefficients come from
microspine.wrench. No independent pad dimensions or allocation formula live here.
"""

import numpy as np

from scr_phybot_mjab.microspine.empirical_limit_surface import EmpiricalLimitSurface
from scr_phybot_mjab.microspine.wrench import distribute_wrench_to_units_numpy, unit_wrench_matrix

from .geometry import phybot_pad_geometry


class StochasticPad:
  def __init__(self, draws=1024, seed=905, geometry=None, surface=None):
    self.surface = surface if surface is not None else EmpiricalLimitSurface()
    self.geometry = geometry if geometry is not None else phybot_pad_geometry()
    self.count = self.geometry.count
    self.allocation = unit_wrench_matrix(self.geometry)
    self.draws = draws
    rng = np.random.default_rng(seed)
    self.u = rng.random((draws, self.count))
    self.v = rng.random((draws, self.count))
    self.x, self.y = self.geometry.x, self.geometry.y
    self.sxx, self.syy, self.srr = self.geometry.sxx, self.geometry.syy, self.geometry.srr

  def units(self, ft, fl=0.0, mn=0.0):
    loads = distribute_wrench_to_units_numpy([ft, fl, 0.0, 0.0, 0.0, mn], self.geometry)
    return loads[:, 0], loads[:, 1]

  def draws_at(self, ft, fl=0.0, mn=0.0):
    a, b = self.units(ft, fl, mn)
    cap, p, valid, tail = self.surface.query_numpy(a, b, self.u)
    loaded = np.all(valid & (self.v >= p), axis=-1)
    return cap, loaded

  @staticmethod
  def contour(values, probability):
    # An invalid draw has value -inf. Do not drop it or use nanquantile:
    # conditioning on successful placements would inflate reliability.
    ordered = np.sort(values, axis=0)
    return ordered[max(0, int(np.floor((1 - probability) * len(ordered))))]

  def normal(self, ft, fl=0.0, ml=0.0, mt=0.0, mn=0.0, prob=0.5):
    cap, loaded = self.draws_at(ft, fl, mn)
    offset = ml * self.allocation[:, 2, 4] + mt * self.allocation[:, 2, 3]
    upper = self.count * np.min(cap - offset, axis=-1)
    upper = np.where(loaded, upper, -np.inf)
    value = self.contour(upper, prob)
    return float(value) if np.isfinite(value) else np.nan

  def bending(self, ft, angles, fn=-60.0, mn=0.0, prob=0.5):
    # Path boundary: first establish these fixed Ft/Fn/Mn loads at zero
    # bending, then increase bending along a ray. A draw failing the initial
    # state cannot recover later on that path. This is not an enumeration
    # of every disconnected statically admissible wrench set.
    cap, loaded = self.draws_at(ft, 0.0, mn)
    coefficients = (
      np.cos(angles)[:, None] * self.allocation[:, 2, 4] + np.sin(angles)[:, None] * self.allocation[:, 2, 3]
    )
    baseline = np.all(cap >= fn / self.count, axis=-1) & loaded
    upper = np.where(
      coefficients[None, :, :] > 1e-12,
      (cap[:, None, :] - fn / self.count) / np.maximum(coefficients[None, :, :], 1e-12),
      np.inf,
    )
    radius = upper.min(axis=-1)
    radius = np.where(baseline[:, None], radius, -np.inf)
    value = self.contour(radius, prob)
    return np.where(np.isfinite(value) & (value >= 0), value, np.nan)
