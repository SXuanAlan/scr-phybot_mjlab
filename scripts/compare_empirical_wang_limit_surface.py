"""Compare unit bench data with the repository's Wang-inspired baseline.

Run with env_mjlab15. The baseline is the CURRENT local implementation,
not a digitization of all of Wang et al.'s Fig. 8. Unit terminal-command
medians are descriptive and conditional on reaching normal loading.
Pad moments are predictions under the same equal-stiffness distribution.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from plot_empirical_pad_limit_surface import (
  DEFAULT_DATA, ROOT, EmpiricalPad, Geometry, load_data, verify_model,
)
from plot_pad_limit_surface import RC_PARAMS, PadSurface, load_limit_surface_cfg, save_three
from plot_wang_limit_surface import surface_metrics
from scr_phybot_mjab.tools.microspine.wang_comparison import WangUnit


def optional_float(value):
  return float(value) if np.isfinite(value) else None


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
  parser.add_argument("--out", type=Path,
                      default=ROOT / "docs/figures/fig_empirical_vs_wang")
  args = parser.parse_args()
  torch.set_num_threads(4)
  cfg = load_limit_surface_cfg(False)
  wang = WangUnit(cfg)
  trials, cells = load_data(args.data)
  preloads = {float(r["fn_n"]) for r in trials}
  if len(preloads) != 1:
    raise ValueError("Compare preload strata separately.")
  geom = Geometry(cfg.unit_rows_rake, cfg.unit_cols_lateral,
                  cfg.pad_span_rake_m, cfg.pad_span_lateral_m)
  pad = EmpiricalPad(cells, geom, next(iter(preloads)))
  checks = verify_model(pad)
  f0s = cfg.f_t_max_range
  source_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()

  cell_table = []
  for c in cells:
    row = dict(c)
    for f0 in f0s:
      row[f"wang_pad_F0_{f0:g}_unit_tension_limit_n"] = optional_float(
        wang.normal_boundary(c["s_n"], c["l_n"], f0))
    cell_table.append(row)

  pitch_arm = geom.sxx / (geom.n * np.max(np.abs(geom.x)))
  pad_table = []
  for ft in (70, 120, 180, 200, 240, 300, 360):
    empirical_fn = float(pad.normal_boundary(ft))
    row = {"pad_shear_n": ft, "empirical_normal_n": empirical_fn,
           "empirical_peel_nm": pitch_arm * empirical_fn}
    for f0 in f0s:
      normal = geom.n * float(wang.normal_boundary(ft / geom.n, 0, f0))
      row[f"wang_F0_{f0:g}_normal_n"] = normal
      row[f"wang_F0_{f0:g}_peel_nm"] = pitch_arm * normal
      # Check analytic peel capacity against the original six-axis evaluator.
      w = torch.tensor([ft, 0, 0, 0, pitch_arm * normal, 0], dtype=torch.float64)
      assert abs(float(PadSurface(cfg, f0).occupancy6(w)) - 1) < 1e-6
    pad_table.append(row)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  for suffix, rows in ((".cells.csv", cell_table), (".pad.csv", pad_table)):
    with args.out.with_suffix(suffix).open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)

  plt.rcParams.update(RC_PARAMS)
  blue, red, gray = "#2a78d6", "#cc5038", "#989a9d"
  fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.35))
  fig.subplots_adjust(left=0.058, right=0.985, top=0.92, bottom=0.25, wspace=0.31)
  for ax in axes:
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=gray, lw=0.7)

  mesh_data = {}

  def draw_unit(ax, column, fixed, grid):
    s, l = (grid, fixed) if column == "s_n" else (fixed, grid)
    limits = [wang.normal_boundary(s, l, f0) for f0 in f0s]
    for values, ls in zip(limits, ("-", "--")):
      ax.plot(grid, values, color=blue, ls=ls, lw=1.4)
    ax.fill_between(grid, *limits, color=blue, alpha=0.11)
    condition = "l_n" if column == "s_n" else "s_n"
    selected = [c for c in cells if c[condition] == fixed and c["n_terminal"]]
    for c in selected:
      x, y = c[column], c["terminal_command_median_n"]
      if c["n_terminal"] == 1:
        ax.plot(x, y, "D", ms=4, mfc="white", mec=red, zorder=5)
      else:
        ax.errorbar(x, y, yerr=[[y - c["terminal_command_q25_n"]],
                                [c["terminal_command_q75_n"] - y]],
                    fmt="o", color=red, ms=3.6, capsize=2, lw=1.0, zorder=4)
    for r in trials:
      signed = int(r["side"]) * float(r["s_n"])
      fields = {"s_n": signed, "l_n": float(r["l_n"])}
      if fields[condition] == fixed and r["result"] == "held":
        ax.plot(fields[column], float(r["fz_cmd_n"]), "^", color=red,
                mfc="white", ms=5, zorder=5)
        ax.annotate("held at test ceiling", (fields[column], float(r["fz_cmd_n"])),
                    xytext=(3, 2), textcoords="offset points", fontsize=6, color=red)
    mesh_data[f"{column}_grid"] = grid
    for f0, values in zip(f0s, limits):
      mesh_data[f"{column}_F0_{f0:g}_normal"] = values

  draw_unit(axes[0], "s_n", 0, np.linspace(-15, 56, 1001))
  axes[0].set(title="(a) unit: zero lateral load", xlabel=r"unit rake shear $s$ [N]",
              ylabel=r"unit normal limit $f_n$ [N] (tension +)", xlim=(-16, 56), ylim=(-5.5, 11.5))
  axes[0].axvline(30, color=gray, ls=":", lw=0.8)
  draw_unit(axes[1], "l_n", 20, np.linspace(0, 35, 701))
  axes[1].set(title="(b) unit: rake shear = 20 N", xlabel=r"unit lateral load $l$ [N]",
              ylabel=r"unit normal limit $f_n$ [N] (tension +)", xlim=(-1, 35), ylim=(-5.5, 11.5))

  ft = np.linspace(0, f0s[1], 1001)
  old = [geom.n * wang.normal_boundary(ft / geom.n, 0, f0) * pitch_arm for f0 in f0s]
  for values, ls in zip(old, ("-", "--")):
    axes[2].plot(ft, values, color=blue, ls=ls, lw=1.4)
  axes[2].fill_between(ft, *old, color=blue, alpha=0.11)
  new_ft = np.linspace(0, geom.n * pad.smax, 721)
  new_fn = pad.normal_boundary(new_ft)
  new_peel = pitch_arm * new_fn
  new_peel = np.where(new_fn >= 0, new_peel, np.nan)
  axes[2].plot(new_ft, new_peel, color=red, lw=1.4)
  for c in cells:
    if c["l_n"] == 0 and c["s_n"] >= 0 and c["n_terminal"] == 1:
      axes[2].plot(c["s_n"] * geom.n,
                    geom.n * c["terminal_command_median_n"] * pitch_arm,
                    "D", ms=4, mfc="white", mec=red)
  axes[2].axvline(geom.n * pad.smax, color=gray, ls=":", lw=0.8)
  axes[2].set(title="(c) pad prediction: peeling limit", xlabel=r"pad rake shear $F_t$ [N]",
              ylabel=r"$|M_l|$ [N$\cdot$m] at $F_n=F_l=0$", xlim=(-10, 660), ylim=(0, 2.15))
  mesh_data.update(pad_ft=ft, pad_old_peel=np.array(old), empirical_ft=new_ft,
                   empirical_peel=new_peel)
  legend = [
    Line2D([], [], color=blue, lw=1.4, label=f"old implementation: pad F0 = {f0s[0]:g} N"),
    Line2D([], [], color=blue, ls="--", lw=1.4, label=f"old implementation: pad F0 = {f0s[1]:g} N"),
    Line2D([], [], color=red, marker="o", lw=0, ms=3.6, label="unit terminal median / IQR"),
    Line2D([], [], color=red, marker="D", mfc="white", lw=0, ms=4, label="one terminal trial"),
  ]
  fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
  save_three(fig, args.out)
  plt.close(fig)
  np.savez_compressed(args.out.with_suffix(".npz"), **mesh_data)

  audit = {
    "source": str(args.data.resolve()), "source_sha256": source_hash,
    "outcomes": dict(Counter(r["result"] for r in trials)),
    "baseline": "Current local Wang-inspired implementation; not original paper data",
    "baseline_cfg": dataclasses.asdict(cfg), "baseline_normalized_metrics": surface_metrics(wang.lut),
    "geometry": dataclasses.asdict(geom), "empirical_checks": checks,
    "unit_tables": "Conditional median/IQR of all detached_normal command observations; no outlier removal",
    "pad_tables": "Identical fully engaged units, rigid equal-stiffness allocation, first characteristic unit limit",
    "missing_baseline": "Blank when fixed S,L already exceed the zero-tension baseline; negative-Fn release not modeled",
    "original_paper_audit": {
      "source": "Wang et al. 2017, Fig. 8, printed p. 991; local PDF page 7",
      "method": "Approximate visual reading, not precise digitization or fit",
      "forward_shear_n_approx": 24,
      "reverse_shear_magnitude_n_approx": 7.5,
      "pure_lateral_n_approx": 18,
      "forward_peak_tension_magnitude_n_approx": 3.2,
      "note": "Paper uses negative z for adhesion. Local documented 0.05 reverse and 0.56 lateral anchors do not reproduce these ratios.",
    },
  }
  args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
  args.out.with_suffix(".caption.txt").write_text(
    "Bench unit boundaries compared with the current repository Wang-inspired implementation. "
    "Blue solid/dashed curves use pad F0=400/650 N and unit F0=F0/12; the shaded span is a "
    "configuration range, not a confidence interval. Red points and error bars are conditional "
    "terminal-command medians and interquartile ranges; hollow diamonds identify n=1 cells. "
    "The hollow triangle is a right-censored hold. All 275 detached_normal trials are retained; "
    "20 early failures and one censored hold are outside the conditional median statistic. "
    "Normal staircase protocol: 10 N unit preload, 0.5 N steps, 3 s dwell. "
    "(a) Zero lateral load. (b) Fixed 20 N unit rake shear. Positive normal force is tensile. "
    "The old model has no tensile solution when the in-plane load already exceeds capacity; "
    "such curves end rather than reporting a negative normal failure boundary. "
    "(c) Peeling with zero net normal/lateral force and no other moments, using the same "
    "3x4 rigid equal-stiffness grid (74.6 x 96 mm) for both models. The red curve is inferred "
    "from unit data and is not a measured pad moment boundary or reliability quantile. "
    "The empirical 360 N shear edge is the end of the measured unit domain, not ultimate "
    "pad shear capacity. This descriptive comparison is not held-out validation. "
    "The local baseline approximates selected paper features and is not a faithful digitization "
    "of the original Fig. 8; see the audit for its lateral/reverse anchor discrepancy.\n")
  assert hashlib.sha256(args.data.read_bytes()).hexdigest() == source_hash
  print(json.dumps({"zero_lateral_cells": [c for c in cell_table if c["l_n"] == 0],
                    "pad_examples": pad_table}, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
