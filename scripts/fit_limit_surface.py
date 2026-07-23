"""Fit MicrospineLimitSurfaceCfg *wedge* parameters from bench pull-off tests.

NOTE: this fits the legacy piecewise-linear wedge envelope
(``limit_surface.model == "wedge"``). The climb task now defaults to the
closed ``"wang2017"`` surface; to calibrate that model against the bench,
fit F0 from pure-rake pulls (``f_t_max_range``) and tune the shape
parameters against angular sweeps using scripts/plot_wang_limit_surface.py
as the shape readout.

Input: CSV per docs/microspine_calibration.md —
  plane,angle_deg,preload_N,stroke_mm,release_force_N[,f_t_N,f_n_N,f_l_N]

Each row is one trial: the pad is preloaded, stroked along +t̂, then loaded
along a fixed in-plane direction until release. The script converts every
trial to a release point in (f_t, f_n, f_l), fits the four envelope faces,
and prints a cfg snippet whose ranges are the requested quantiles of the
per-face estimates (built-in DR bounds = measured scatter).

Usage:
  python scripts/fit_limit_surface.py data/pull_tests.csv
  python scripts/fit_limit_surface.py data/pull_tests.csv --quantiles 0.05 0.95
"""

from __future__ import annotations

import argparse
import csv
import math
import sys

import numpy as np

# Trials whose load direction is within this of pure +t̂ calibrate the
# ultimate-shear face; near ±180° the reverse face. In between, the tension /
# lateral cones.
ULTIMATE_CONE_DEG = 10.0
REVERSE_CONE_DEG = 25.0


def load_trials(path: str) -> list[dict]:
  trials = []
  with open(path, newline="") as f:
    for row in csv.DictReader(f):
      plane = row["plane"].strip().lower()
      if plane not in ("tn", "tl"):
        raise ValueError(f"Unknown plane {plane!r} (expected 'tn' or 'tl')")
      angle = math.radians(float(row["angle_deg"]))
      if row.get("f_t_N") not in (None, ""):
        f_t = float(row["f_t_N"])
        f_n = float(row.get("f_n_N") or 0.0)
        f_l = float(row.get("f_l_N") or 0.0)
      else:
        r = float(row["release_force_N"])
        f_t = r * math.cos(angle)
        f_n = r * math.sin(angle) if plane == "tn" else 0.0
        f_l = r * math.sin(angle) if plane == "tl" else 0.0
      trials.append({"plane": plane, "angle": angle, "f": (f_t, f_n, f_l)})
  if not trials:
    raise ValueError(f"No trials found in {path}")
  return trials


def fit_cone(points: list[tuple[float, float]]) -> tuple[float, float] | None:
  """Least-squares fit cap = mu * max(f_t, 0) + f0 through release points.

  ``points`` are (f_t, |f_perp|) at release on one cone face. Returns
  (mu, f0), clamping mu >= 0 (a refit with mu = 0 if needed).
  """
  if len(points) < 2:
    return None
  ft = np.array([max(p[0], 0.0) for p in points])
  fp = np.array([p[1] for p in points])
  a = np.stack([ft, np.ones_like(ft)], axis=1)
  (mu, f0), *_ = np.linalg.lstsq(a, fp, rcond=None)
  if mu < 0.0:
    mu, f0 = 0.0, float(fp.mean())
  return float(mu), float(f0)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("csv_path")
  parser.add_argument(
    "--quantiles",
    type=float,
    nargs=2,
    default=(0.10, 0.90),
    help="Range bounds as quantiles of the per-trial estimates.",
  )
  args = parser.parse_args()
  q_lo, q_hi = args.quantiles

  trials = load_trials(args.csv_path)

  # Pass 1: the ultimate-shear and reverse faces from rays nearly along ±t̂
  # (these buckets are unambiguous).
  ultimate, reverse = [], []
  for tr in trials:
    f_t, _, _ = tr["f"]
    angle_deg = math.degrees(tr["angle"])
    folded = min(abs(angle_deg), abs(180.0 - abs(angle_deg)))
    if abs(angle_deg) <= ULTIMATE_CONE_DEG:
      ultimate.append(f_t)
    elif folded <= REVERSE_CONE_DEG and abs(angle_deg) > 90.0:
      reverse.append(abs(f_t))

  # Pass 2: cone faces. Classifying by ray angle would contaminate the cone
  # fits — a ray at e.g. 15° releases on the ultimate face (its direction
  # never crosses the cone), and including it drags the fitted slope down.
  # Instead classify by which face the release point actually sits on: drop
  # points within 10% of the estimated ultimate/reverse faces.
  f_t_hi = 0.9 * float(np.median(ultimate)) if ultimate else float("inf")
  f_t_lo = -0.9 * float(np.median(reverse)) if reverse else -float("inf")
  tension_pts, lateral_pts = [], []
  for tr in trials:
    f_t, f_n, f_l = tr["f"]
    angle_deg = math.degrees(tr["angle"])
    folded = min(abs(angle_deg), abs(180.0 - abs(angle_deg)))
    if abs(angle_deg) <= ULTIMATE_CONE_DEG or (
      folded <= REVERSE_CONE_DEG and abs(angle_deg) > 90.0
    ):
      continue
    if not (f_t_lo < f_t < f_t_hi):
      continue
    if tr["plane"] == "tn" and f_n > 0.0:
      tension_pts.append((f_t, f_n))
    elif tr["plane"] == "tl" and abs(f_l) > 0.0:
      lateral_pts.append((f_t, abs(f_l)))

  def qrange(values: list[float]) -> tuple[float, float] | None:
    if not values:
      return None
    arr = np.array(values)
    return float(np.quantile(arr, q_lo)), float(np.quantile(arr, q_hi))

  f_t_max_range = qrange(ultimate)
  f_rev_max_range = qrange(reverse)
  tension_fit = fit_cone(tension_pts)
  lateral_fit = fit_cone(lateral_pts)

  # Scatter of the cone fits: per-point implied intercept at the fitted slope.
  def cone_ranges(fit, points, label):
    if fit is None or not points:
      return None, None
    mu, _ = fit
    # Ray probing only reaches the high-f_t part of a cone in a narrow
    # angular band near atan(mu); without trials there the slope is poorly
    # identified even though the intercept looks fine.
    ft_span = max(p[0] for p in points) - min(p[0] for p in points)
    ref = max((f_t_max_range or (0.0, 0.0))[1], 1.0)
    if ft_span < 0.3 * ref:
      print(
        f"# WARNING: {label} cone trials span only {ft_span:.0f} N of f_t — "
        f"slope mu is poorly identified. Add trials with angles near "
        f"atan(mu) ≈ {math.degrees(math.atan(max(mu, 1e-3))):.0f}° "
        f"(see docs/microspine_calibration.md)."
      )
    intercepts = [fp - mu * max(ft, 0.0) for ft, fp in points]
    return (round(mu, 3), round(mu, 3)), qrange(intercepts)

  mu_adh_range, f_n0_range = cone_ranges(tension_fit, tension_pts, "tension")
  mu_lat_range, f_l0_range = cone_ranges(lateral_fit, lateral_pts, "lateral")

  print(f"# fitted from {len(trials)} trials in {args.csv_path}")
  print(f"# quantile range: [{q_lo}, {q_hi}]")
  print("MicrospineLimitSurfaceCfg(")

  def emit(name: str, rng, count: int) -> None:
    if rng is None:
      print(f"  # {name}: insufficient data ({count} trials) — keep default")
    else:
      print(f"  {name}=({rng[0]:.1f}, {rng[1]:.1f}),  # n={count}")

  emit("f_t_max_range", f_t_max_range, len(ultimate))
  emit("f_rev_max_range", f_rev_max_range, len(reverse))
  if mu_adh_range is None:
    print(f"  # mu_adhesion/f_n0: insufficient tension-cone data ({len(tension_pts)})")
  else:
    print(f"  mu_adhesion_range=({mu_adh_range[0]}, {mu_adh_range[1]}),  # n={len(tension_pts)}")
    emit("f_n0_range", f_n0_range, len(tension_pts))
  if mu_lat_range is None:
    print(f"  # mu_lateral/f_l0: insufficient lateral-cone data ({len(lateral_pts)})")
  else:
    print(f"  mu_lateral_range=({mu_lat_range[0]}, {mu_lat_range[1]}),  # n={len(lateral_pts)}")
    emit("f_l0_range", f_l0_range, len(lateral_pts))
  print(")")

  if f_t_max_range is None and tension_fit is None:
    sys.exit(1)


if __name__ == "__main__":
  main()
