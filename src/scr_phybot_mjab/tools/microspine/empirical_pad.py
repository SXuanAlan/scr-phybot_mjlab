"""A descriptive pad model from the single-unit bench dataset (terminal-command medians).

This is a protocol-conditioned prediction, not a calibrated Wang surface or
a hardware validation of a complete pad. Each unit uses a piecewise-linear
interpolant of the terminal-command median at each tested (signed S, L)
cell; early failures and right-censored holds are counted separately.
Negative lateral forces assume symmetry. The pad uses the same rigid,
equal-stiffness wrench allocation as microspine.wrench.distribute_wrench_to_units;
all units are engaged and identical, and the first unit to reach its
terminal boundary limits the pad. This is NOT the median or reliability of
a random pad. Nothing extrapolates outside the convex hull of the measured
unit force cells or below the tested normal preload.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator


@dataclass
class Geometry:
  rows: int
  cols: int
  span_rake_m: float
  span_lateral_m: float

  def __post_init__(self):
    if min(self.rows, self.cols) < 2 or min(self.span_rake_m, self.span_lateral_m) <= 0:
      raise ValueError("Need at least two rows/columns and positive grid spans.")
    x, y = np.meshgrid(
      np.linspace(-0.5, 0.5, self.rows) * self.span_rake_m,
      np.linspace(-0.5, 0.5, self.cols) * self.span_lateral_m,
      indexing="ij",
    )
    self.x, self.y = x.ravel(), y.ravel()
    self.n = self.x.size
    self.sxx, self.syy = float(self.x @ self.x), float(self.y @ self.y)
    self.srr = self.sxx + self.syy


def load_data(path: Path):
  with path.open() as stream:
    trials = list(csv.DictReader(line for line in stream if not line.startswith("#")))
  if not trials or len({r["trial"] for r in trials}) != len(trials):
    raise ValueError("Missing trials or duplicate trial identifiers.")
  groups = defaultdict(list)
  for r in trials:
    s, l = int(r["side"]) * float(r["s_n"]), float(r["l_n"])
    if l < 0:
      raise ValueError("This dataset reader expects the measured L >= 0 half.")
    groups[s, l].append(r)
  cells = []
  for (s, l), rows in sorted(groups.items()):
    values = np.array([float(r["fz_cmd_n"]) for r in rows if r["result"] == "detached_normal"])
    if not np.isfinite(values).all():
      raise ValueError(f"Non-finite terminal command in cell {(s, l)}")
    cells.append(
      {
        "s_n": s,
        "l_n": l,
        "n_total": len(rows),
        "n_terminal": len(values),
        "n_early": sum(r["result"] in ("detached_drag", "detached_lateral") for r in rows),
        "n_censored": sum(r["result"] == "held" for r in rows),
        "terminal_command_median_n": float(np.median(values)) if len(values) else None,
        "terminal_command_q25_n": float(np.quantile(values, 0.25)) if len(values) else None,
        "terminal_command_q75_n": float(np.quantile(values, 0.75)) if len(values) else None,
        "terminal_trial_ids": " ".join(r["trial"] for r in rows if r["result"] == "detached_normal"),
      }
    )
  return trials, cells


class EmpiricalPad:
  def __init__(self, cells, geom: Geometry, preload_n: float):
    valid = [c for c in cells if c["n_terminal"]]
    self.points = np.array([[c["s_n"], c["l_n"]] for c in valid])
    self.values = np.array([c["terminal_command_median_n"] for c in valid])
    self.interpolator = LinearNDInterpolator(self.points, self.values, fill_value=np.nan)
    self.geom, self.preload_n = geom, float(preload_n)
    self.smin, self.lmin = self.points.min(axis=0)
    self.smax, self.lmax = self.points.max(axis=0)

  def unit_boundary(self, s, l):
    s, l = np.broadcast_arrays(np.asarray(s, float), np.asarray(l, float))
    return self.interpolator(s, np.abs(l))

  def distribute(self, ft, fl=0.0, fn=0.0, mt=0.0, ml=0.0, mn=0.0):
    ft, fl, fn, mt, ml, mn = [a[..., None] for a in np.broadcast_arrays(ft, fl, fn, mt, ml, mn)]
    g = self.geom
    return (
      ft / g.n - mn * g.y / g.srr,
      fl / g.n + mn * g.x / g.srr,
      fn / g.n - ml * g.x / g.sxx + mt * g.y / g.syy,
    )

  def normal_boundary(self, ft, fl=0.0, mt=0.0, ml=0.0, mn=0.0):
    """Largest Fn allowed by every unit; NaN means outside data support."""
    a, b, offset = self.distribute(ft, fl, 0.0, mt, ml, mn)
    cap = self.unit_boundary(a, b)
    upper = self.geom.n * np.min(cap - offset, axis=-1)
    return upper

  def margin(self, ft, fl=0.0, fn=0.0, mt=0.0, ml=0.0, mn=0.0):
    a, b, c = self.distribute(ft, fl, fn, mt, ml, mn)
    upper = self.unit_boundary(a, b)
    result = np.min(upper - c, axis=-1)
    supported = np.all(np.isfinite(upper) & np.isfinite(c), axis=-1)
    return np.where(supported, result, np.nan)

  def moment_capacity(self, ft, axis: str, max_moment=18.0):
    """First failure from zero moment. Data-limited holds are lower bounds."""
    m = np.linspace(0, max_moment, 2401)
    terms = {axis: m[None, :]}
    margin = self.margin(np.asarray(ft)[:, None], **terms)
    capacity = np.full(len(ft), np.nan)
    lower_bound = np.full(len(ft), np.nan)
    for i, values in enumerate(margin):
      if not np.isfinite(values[0]):
        continue
      if values[0] < 0:
        capacity[i] = 0
        continue
      exits = np.flatnonzero(~np.isfinite(values) | (values < 0))
      if not len(exits):
        lower_bound[i] = m[-1]
        continue
      j = int(exits[0])
      if not np.isfinite(values[j]):
        lower_bound[i] = m[max(0, j - 1)]
      else:
        fraction = values[j - 1] / (values[j - 1] - values[j])
        capacity[i] = m[j - 1] + fraction * (m[j] - m[j - 1])
    return capacity, lower_bound


def verify_model(pad: EmpiricalPad):
  """Check wrench conservation and exact zero-moment scaling numerically."""
  rng = np.random.default_rng(20260905)
  w = rng.normal(size=(128, 6))
  a, b, c = pad.distribute(*(w[:, i] for i in range(6)))
  g = pad.geom
  recovered = np.stack(
    [
      a.sum(-1),
      b.sum(-1),
      c.sum(-1),
      (g.y * c).sum(-1),
      -(g.x * c).sum(-1),
      (g.x * b - g.y * a).sum(-1),
    ],
    axis=-1,
  )
  force_moment_error = float(np.max(np.abs(w - recovered)))
  zero = pad.normal_boundary(pad.points[:, 0] * g.n, pad.points[:, 1] * g.n)
  scale_error = float(np.max(np.abs(zero - g.n * pad.values)))
  assert force_moment_error < 1e-10, force_moment_error
  assert scale_error < 1e-8, scale_error
  assert np.isnan(pad.unit_boundary(pad.smax + 1, 0))
  assert np.allclose(pad.unit_boundary(15, 10), pad.unit_boundary(15, -10))
  return {"wrench_reconstruction_max_abs_error": force_moment_error, "zero_moment_scaling_max_abs_error_n": scale_error}
