"""Plot pad wrench limits inferred from the single-unit bench dataset.

Run with env_mjlab15:
  python scripts/plot_empirical_pad_limit_surface.py

This is a descriptive, protocol-conditioned prediction, not a calibrated
Wang surface or a hardware validation of a complete pad. Each unit uses a
piecewise-linear interpolant of the terminal-command median at each tested
(signed S, L) cell. All detached_normal trials are retained. Early failures
and right-censored holds are counted separately, not treated as terminal
normal-load observations. Negative lateral forces assume symmetry.

The pad uses the same rigid, equal-stiffness wrench allocation as
microspine.affordance.distribute_wrench_to_units. All units are assumed
engaged and identical; the first unit to reach its characteristic terminal
boundary limits the pad. This is NOT the median/reliability of a random pad.
No extrapolation outside the convex hull of the measured unit force cells
or below the tested normal preload is allowed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import LinearNDInterpolator

ROOT = Path(__file__).resolve().parents[1]
from scr_phybot_mjab.tools.paths import DEFAULT_UNIT_TRIALS as DEFAULT_DATA


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


def style_axis(ax):
  ax.spines[["right", "top"]].set_visible(False)
  ax.grid(alpha=0.18, linewidth=0.7)
  ax.set_axisbelow(True)
  ax.axhline(0, color="#68717c", linewidth=0.7, alpha=0.7)


def make_figure(pad: EmpiricalPad, trials, stem: Path):
  plt.rcParams.update(
    {
      "font.family": "DejaVu Sans",
      "font.size": 10,
      "axes.titlesize": 12,
      "axes.labelsize": 11,
      "axes.edgecolor": "#46505c",
      "text.color": "#182332",
      "axes.labelcolor": "#182332",
      "pdf.fonttype": 42,
      "svg.fonttype": "none",
    }
  )
  g = pad.geom
  fig = plt.figure(figsize=(15, 10.4), facecolor="white")
  gs = fig.add_gridspec(2, 2, left=0.075, right=0.945, top=0.855, bottom=0.125, hspace=0.40, wspace=0.34)
  fig.suptitle(
    "Predicted pad limit surface from unit bench tests", fontsize=19, fontweight="bold", x=0.075, ha="left", y=0.975
  )
  fig.text(
    0.075,
    0.926,
    f"{g.n}-unit pad  |  {g.rows} × {g.cols} grid  |  "
    f"{g.span_rake_m * 1000:g} × {g.span_lateral_m * 1000:g} mm grid span\n"
    f"Terminal-detachment command · 0.5 N unit steps · 3 s dwell · {pad.preload_n:g} N unit preload",
    fontsize=10.5,
    color="#526174",
    linespacing=1.6,
    va="top",
  )

  ax = fig.add_subplot(gs[0, 0], projection="3d")
  s = np.linspace(pad.smin, pad.smax, 91)
  l = np.linspace(0, pad.lmax, 61)
  ss, ll = np.meshgrid(s, l)
  unit_z = pad.unit_boundary(ss, ll)
  xx, yy, zz = g.n * ss, g.n * ll, g.n * unit_z
  surface = ax.plot_surface(
    xx, yy, zz, cmap="RdBu", vmin=-120, vmax=120, alpha=0.9, linewidth=0, antialiased=True, rcount=61, ccount=91
  )
  ax.plot_wireframe(xx, -yy, zz, rstride=5, cstride=7, color="#8694a7", alpha=0.52, linewidth=0.6)
  ax.scatter(
    g.n * pad.points[:, 0], g.n * pad.points[:, 1], g.n * pad.values, s=11, color="#152435", depthshade=False, alpha=0.9
  )
  held = [r for r in trials if r["result"] == "held"]
  for r in held:
    hx = g.n * int(r["side"]) * float(r["s_n"])
    hy, hz = g.n * float(r["l_n"]), g.n * float(r["fz_cmd_n"])
    ax.scatter(
      [hx], [hy], [hz], marker="^", s=65, facecolor="white", edgecolor="#203d70", linewidth=1.6, depthshade=False
    )
  ax.contour(xx, yy, zz, levels=[0], colors=["#182332"], linewidths=1.3)
  ax.contour(xx, -yy, zz, levels=[0], colors=["#637084"], linestyles="--", linewidths=1)
  ax.set_title("(a) Zero-moment force boundary", loc="left", pad=12, fontweight="bold")
  ax.set_xlabel(r"$F_t$  rake shear [N]", labelpad=8)
  ax.set_ylabel(r"$F_l$  lateral [N]", labelpad=9)
  ax.set_zlabel(r"$F_n^{det}$ [N]", labelpad=5)
  ax.set_xlim(g.n * pad.smin, g.n * pad.smax)
  ax.set_ylim(-g.n * pad.lmax, g.n * pad.lmax)
  ax.set_zlim(-125, 135)
  ax.set_box_aspect((1.2, 1.15, 0.7))
  ax.view_init(elev=25, azim=55)
  ax.tick_params(labelsize=8, pad=1)
  for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_alpha(0.04)
  ax.text2D(
    0.0,
    -0.13,
    "Dots: unit-cell medians × N; triangle: censored hold × N\n"
    "Gray wireframe: untested lateral half, assumed symmetric",
    transform=ax.transAxes,
    fontsize=8.5,
    color="#526174",
  )
  cbax = fig.add_axes([0.425, 0.63, 0.010, 0.17])
  cb = fig.colorbar(surface, cax=cbax, ticks=[-120, -60, 0, 60, 120])
  cb.ax.tick_params(labelsize=8)

  bx = fig.add_subplot(gs[0, 1])
  style_axis(bx)
  ft = np.linspace(g.n * pad.smin, g.n * pad.smax, 721)
  colors = ["#173e72", "#397ab3", "#53a99c", "#cc8744"]
  peel_values = [0.0, 0.5, 1.0, 1.5]
  curves = []
  for moment, color in zip(peel_values, colors):
    fn = pad.normal_boundary(ft, ml=moment)
    curves.append(fn)
    bx.plot(ft, fn, color=color, linewidth=2, label=rf"$|M_l|={moment:g}$ N·m")
  bx.set_title("(b) Peeling reduces normal-load capacity", loc="left", fontweight="bold")
  bx.set_xlabel(r"$F_t$  rake shear [N]")
  bx.set_ylabel(r"$F_n^{det}$  terminal normal load [N]")
  bx.set_xlim(ft[0], ft[-1])
  bx.set_ylim(-125, 105)
  bx.text(0.035, 0.95, r"$F_l=M_t=M_n=0$", transform=bx.transAxes, va="top")
  bx.legend(loc="lower right", frameon=False, fontsize=9)

  cx = fig.add_subplot(gs[1, 0])
  style_axis(cx)
  positive_ft = np.linspace(0, g.n * pad.smax, 241)
  caps = {}
  for axis, color, label in [
    ("ml", "#2873b8", r"Peel $M_l$"),
    ("mt", "#dc8343", r"Roll $M_t$"),
    ("mn", "#259684", r"Twist $M_n$"),
  ]:
    cap, bound = pad.moment_capacity(positive_ft, axis)
    caps[axis] = (cap, bound)
    cx.plot(positive_ft, cap, color=color, linewidth=2, label=label)
    indices = np.flatnonzero(np.isfinite(bound))[::10]
    if len(indices):
      cx.scatter(
        positive_ft[indices], bound[indices], s=27, marker="^", facecolors="none", edgecolors=color, linewidth=1.1
      )
  cx.set_title("(c) Moment limits at zero normal load", loc="left", fontweight="bold")
  cx.set_xlabel(r"$F_t$  rake shear [N]")
  cx.set_ylabel("Moment magnitude [N·m]")
  cx.set_xlim(0, positive_ft[-1])
  cx.set_ylim(bottom=0)
  cx.text(0.025, 0.96, r"$F_l=F_n=0$; other moments zero", transform=cx.transAxes, va="top", fontsize=9)
  handles, labels = cx.get_legend_handles_labels()
  handles.append(Line2D([], [], linestyle="none", marker="^", markerfacecolor="none", markeredgecolor="#526174"))
  labels.append("Data-domain limit (lower bound)")
  cx.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, 0.88), frameon=False, fontsize=8.5)

  dx = fig.add_subplot(gs[1, 1])
  style_axis(dx)
  dx.axvline(0, color="#68717c", linewidth=0.7, alpha=0.7)
  for ft_value, color in zip([g.n * 5.0, g.n * 15.0, g.n * 25.0], ["#73a6ca", "#3d86ab", "#174d78"]):
    normal_cap = float(pad.unit_boundary(ft_value / g.n, 0))
    if not np.isfinite(normal_cap) or normal_cap <= 0:
      continue
    # On this symmetric grid and Ft-only slice the outer normal loads are
    # |Ml| max|x|/Sxx + |Mt| max|y|/Syy. The tensile side sets the bound;
    # the seating preload is not a compression limit on the opposite side.
    allowed = normal_cap
    peel = allowed * g.sxx / np.max(np.abs(g.x))
    roll = allowed * g.syy / np.max(np.abs(g.y))
    px, py = np.array([peel, 0, -peel, 0, peel]), np.array([0, roll, 0, -roll, 0])
    dx.fill(px, py, color=color, alpha=0.065)
    dx.plot(px, py, color=color, linewidth=2, label=rf"$F_t={ft_value:g}$ N")
    assert np.max(np.abs(pad.margin(ft_value, ml=px, mt=py))) < 1e-8
  dx.set_title("(d) Coupled peel–roll boundary", loc="left", fontweight="bold")
  dx.set_xlabel(r"$M_l$  peeling moment [N·m]")
  dx.set_ylabel(r"$M_t$  rolling moment [N·m]")
  dx.set_aspect("equal", adjustable="datalim")
  dx.text(0.035, 0.95, r"$F_l=F_n=M_n=0$", transform=dx.transAxes, va="top", fontsize=9)
  dx.legend(loc="lower left", frameon=False, fontsize=9)

  fig.text(
    0.075,
    0.055,
    "Characteristic prediction: identical engaged units, equal stiffness, first-unit terminal failure. "
    "This is not a measured pad boundary or a pad reliability quantile.\n"
    "Interpolation stays inside the tested unit-force domain; the plotted force range is not an ultimate shear limit.",
    color="#526174",
    fontsize=9,
    linespacing=1.6,
  )
  for suffix in ["png", "pdf", "svg"]:
    fig.savefig(stem.with_suffix("." + suffix), dpi=220, facecolor="white")
  plt.close(fig)
  np.savez_compressed(
    stem.with_suffix(".npz"),
    unit_cell_xy=pad.points,
    unit_cell_terminal_median=pad.values,
    pad_force_ft=xx,
    pad_force_fl=yy,
    pad_force_fn=zz,
    force_slice_ft=ft,
    force_slice_ml=np.array(peel_values),
    force_slice_fn=np.stack(curves),
    moment_ft=positive_ft,
    **{
      f"{axis}_{kind}": array
      for axis, arrays in caps.items()
      for kind, array in zip(["capacity", "domain_lower_bound"], arrays)
    },
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
  parser.add_argument("--out", type=Path, default=ROOT / "docs/figures/fig_pad_limit_surface_empirical")
  parser.add_argument("--rows", type=int)
  parser.add_argument("--cols", type=int)
  parser.add_argument("--span-rake-m", type=float)
  parser.add_argument("--span-lateral-m", type=float)
  args = parser.parse_args()
  geometry_args = [args.rows, args.cols, args.span_rake_m, args.span_lateral_m]
  if any(x is None for x in geometry_args):
    from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import phybot_c2_climb_wall_env_cfg

    cfg = phybot_c2_climb_wall_env_cfg(play=True).commands["microspine_affordance"].microspine_cfg.limit_surface
    defaults = [cfg.unit_rows_rake, cfg.unit_cols_lateral, cfg.pad_span_rake_m, cfg.pad_span_lateral_m]
    geometry_args = [a if a is not None else b for a, b in zip(geometry_args, defaults)]
  geom = Geometry(*geometry_args)
  trials, cells = load_data(args.data)
  preloads = {float(r["fn_n"]) for r in trials}
  if len(preloads) != 1:
    raise ValueError("Stratify different preloads before constructing this figure.")
  pad = EmpiricalPad(cells, geom, next(iter(preloads)))
  checks = verify_model(pad)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  with args.out.with_suffix(".cells.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
    writer.writeheader()
    writer.writerows(cells)
  source_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
  audit = {
    "source": str(args.data.resolve()),
    "source_sha256": source_hash,
    "n_summary_trials": len(trials),
    "outcomes": dict(Counter(r["result"] for r in trials)),
    "n_cells": len(cells),
    "n_terminal_cells": len(pad.points),
    "cell_terminal_repeats_min": min(c["n_terminal"] for c in cells),
    "cell_terminal_repeats_max": max(c["n_terminal"] for c in cells),
    "geometry": {
      "rows": geom.rows,
      "cols": geom.cols,
      "span_rake_m": geom.span_rake_m,
      "span_lateral_m": geom.span_lateral_m,
    },
    "normal_preload_n": next(iter(preloads)),
    "normal_dwell_s": 3.0,
    "normal_step_n": 0.5,
    "protocol_source": "User-supplied acquisition definition",
    "boundary_column": "fz_cmd_n",
    "statistic": "median within nominal (side*S, L) cells",
    "interpolation": "piecewise-linear Delaunay, within measured positive-lateral convex hull",
    "negative_lateral": "assumed reflection symmetry, not measured",
    "early_failures": "reported separately; excluded only from conditional normal-boundary statistic",
    "censored_holds": "reported separately and shown as lower-bound markers, not fitted as failures",
    "outlier_policy": "No detached_normal trials excluded",
    "pad_status": "Prediction from identical conditional characteristic units, not hardware-validated",
    "uncertainty": "No pad survival probability or confidence interval inferred",
    "moment_capacity": "First failure from zero moment; data-domain exits reported as lower bounds",
    "unlisted_trace_files": sorted(
      p.name for p in args.data.parent.glob("*_T*.csv") if p.name not in {r["detail_csv"] for r in trials}
    ),
    "checks": checks,
  }
  args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
  make_figure(pad, trials, args.out)
  outcomes = audit["outcomes"]
  normal_count = outcomes.get("detached_normal", 0)
  early_count = outcomes.get("detached_drag", 0) + outcomes.get("detached_lateral", 0)
  held_count = outcomes.get("held", 0)
  caption = (
    "Predicted pad force and moment limits inferred from single-unit bench tests. "
    f"The pad comprises {geom.rows} x {geom.cols} identical, engaged units with equal stiffness, "
    f"on a {1000 * geom.span_rake_m:g} x {1000 * geom.span_lateral_m:g} mm center-grid span "
    "taken from the current simulation geometry. Positive Ft is rake shear, Fl is lateral, "
    "and positive Fn is tensile pull-off. Ml is peeling about the lateral axis, Mt is "
    "bending about the rake axis, and Mn is twist about the normal. Unit observations "
    "use the instantaneous terminal-event command fz_cmd_n, not the post-slip measured "
    f"reaction or the first-release label. The protocol uses {pad.preload_n:g} N preload, sequential "
    "tangential loading, and a normal staircase with 0.5 N increments and 3 s dwell. "
    f"All {normal_count} detached_normal trials are included in cell medians; {early_count} pre-normal failures "
    f"are counted separately, and {held_count} hold(s) at the test ceiling are right-censored. "
    "This normal boundary is conditional on reaching the normal-loading phase. "
    f"Piecewise-linear interpolation uses {len(pad.points)} sampled cells; there is no held-out model "
    "validation in this descriptive plot. The unmeasured negative-lateral half is mirrored "
    "and rendered as gray wireframe. (a) Zero-moment pad force boundary, Fn=N*g(Ft/N,Fl/N); "
    "black dots are scaled unit medians, not pad measurements. The triangle is a scaled "
    "unit censored hold, not a demonstrated pad load. (b) Force-normal sections under "
    "peeling moments. (c) Individual moment limits at Fn=Fl=0. Hollow triangles indicate "
    "still-admissible loads at the edge of unit-data support, not observed failure limits. "
    "(d) Coupled peeling/rolling limits. The rigid equal-stiffness distribution satisfies "
    "sum fi=F and sum ri cross fi=M; the first unit reaching its terminal characteristic "
    "boundary limits the pad. No extrapolation beyond the sampled shear/lateral domain is drawn. "
    f"The {pad.preload_n:g} N seating preload is not a compression failure limit. The surfaces are open and "
    "their plotted edges do not identify ultimate shear capacities. The characteristic "
    "unit median is not a median pad failure load or a pad safety/reliability quantile. "
    "The prediction assumes mount-compatible unit behavior and does not validate pad "
    "load sharing, progressive failure, or transient behavior on hardware.\n"
  )
  args.out.with_suffix(".caption.txt").write_text(caption)
  assert hashlib.sha256(args.data.read_bytes()).hexdigest() == source_hash
  print(
    json.dumps(
      {
        "figure": str(args.out.with_suffix(".png")),
        "trials": len(trials),
        "geometry": audit["geometry"],
        "outcomes": audit["outcomes"],
        "checks": checks,
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
