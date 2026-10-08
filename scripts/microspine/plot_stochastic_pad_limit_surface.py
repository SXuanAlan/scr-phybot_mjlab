"""Original paper figure styling, with measured-distribution pad predictions.

Probability contours are Monte Carlo predictions under independent units and
first-unit release, NOT hardware-calibrated pad reliability/confidence bands.
"""

from __future__ import annotations

import argparse
import json

import matplotlib
import mjlab  # noqa: F401  (initialize mjlab before project imports)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scr_phybot_mjab.tools.microspine.stochastic_pad import StochasticPad
from scr_phybot_mjab.tools.microspine.wang_comparison import WangUnit
from scr_phybot_mjab.tools.paths import EMPIRICAL_UNIT_REPORT, FIGURES_DIR
from scr_phybot_mjab.tools.plotting import RC_PARAMS, save_three, style_3d, style_slice

STEM = FIGURES_DIR / "fig_pad_limit_surface_stochastic_cad"


def main(argv: list[str] | None = None):
  argparse.ArgumentParser(description=__doc__).parse_args(argv)
  plt.rcParams.update(RC_PARAMS)
  pad = StochasticPad()
  ft = np.linspace(pad.count * pad.surface.s_min, pad.count * pad.surface.s_max, 46)
  fl = np.linspace(-pad.count * pad.surface.l_max, pad.count * pad.surface.l_max, 49)
  fx, fy = np.meshgrid(ft, fl, indexing="ij")
  sheets = {p: np.full_like(fx, np.nan) for p in (0.1, 0.5, 0.9)}
  for i, s in enumerate(ft):
    for j, l in enumerate(fl):
      for p in sheets:
        sheets[p][i, j] = pad.normal(s, l, prob=p)
  bend_ft = np.linspace(0, pad.count * pad.surface.s_max, 37)
  psi = np.linspace(0, 2 * np.pi, 97)
  radii = np.stack([pad.bending(s, psi) for s in bend_ft])
  bx = np.broadcast_to(bend_ft[:, None], radii.shape)
  by, bz = radii * np.cos(psi), radii * np.sin(psi)
  twist = np.linspace(-6, 6, 49)
  tx, ty = np.meshgrid(bend_ft, twist, indexing="ij")
  pitch_pos = np.empty_like(tx)
  pitch_neg = np.empty_like(tx)
  for i, s in enumerate(bend_ft):
    for j, mn in enumerate(twist):
      pair = pad.bending(s, np.array([0.0, np.pi]), mn=mn)
      pitch_pos[i, j], pitch_neg[i, j] = pair[0], -pair[1]
  slice_ft = 120.0
  slice_forces = {p: np.array([pad.normal(slice_ft, l, prob=p) for l in fl]) for p in sheets}
  slice_bend = {p: pad.bending(slice_ft, psi, prob=p) for p in sheets}
  slice_twist = {
    p: np.array([pad.bending(slice_ft, np.array([0.0, np.pi]), mn=mn, prob=p) for mn in twist]) for p in sheets
  }
  fig = plt.figure(figsize=(10.5, 6.4), layout="constrained")
  grid = fig.add_gridspec(2, 3)
  top = [fig.add_subplot(grid[0, i], projection="3d") for i in range(3)]
  low = [fig.add_subplot(grid[1, i]) for i in range(3)]
  surf = dict(cmap="Blues", alpha=0.76, linewidth=0, rasterized=True, antialiased=True)
  top[0].plot_surface(fx, fy, sheets[0.5], **surf)
  style_3d(top[0], "$F_t$ [N]", "$F_l$ [N]", "$F_n$ [N]", (1.3, 1, 0.8))
  top[0].view_init(elev=23, azim=-55)
  top[0].set_title("(a) Force boundary, modeled hold = 50%", fontsize=8)
  top[1].plot_surface(bx, by, bz, **surf)
  style_3d(top[1], "$F_t$ [N]", "$M_l$ [N m]", "$M_t$ [N m]", (1.3, 1, 0.8))
  top[1].view_init(elev=22, azim=-58)
  top[1].set_title("(b) Bending path, $F_n$ = -60 N, hold = 50%", fontsize=8)
  for z in (pitch_pos, pitch_neg):
    top[2].plot_surface(tx, ty, z, **surf)
  style_3d(top[2], "$F_t$ [N]", "$M_n$ [N m]", "$M_l$ [N m]", (1.3, 1, 0.8))
  top[2].view_init(elev=23, azim=-55)
  top[2].set_title("(c) Peel paths, $F_n$ = -60 N, hold = 50%", fontsize=8)
  colors = {0.1: "#8eb6db", 0.5: "#165695", 0.9: "#162d42"}
  for p in (0.1, 0.5, 0.9):
    low[0].plot(fl, slice_forces[p], color=colors[p], label=f"{p:.0%}")
    r = slice_bend[p]
    low[1].plot(r * np.cos(psi), r * np.sin(psi), color=colors[p])
    low[2].plot(twist, slice_twist[p][:, 0], color=colors[p])
    low[2].plot(twist, -slice_twist[p][:, 1], color=colors[p])
  low[0].fill_between(fl, slice_forces[0.9], slice_forces[0.1], alpha=0.12, color=colors[0.5])
  low[0].legend(title="Modeled hold", frameon=False, fontsize=7, title_fontsize=7)
  style_slice(low[0], "$F_l$ [N]", "$F_n$ [N]", "(d) Force slice, $F_t$ = 120 N")
  style_slice(low[1], "$M_l$ [N m]", "$M_t$ [N m]", "(e) Bending slice, $F_t$ = 120 N")
  style_slice(low[2], "$M_n$ [N m]", "$M_l$ [N m]", "(f) Twist-peel slice, $F_t$ = 120 N")
  low[1].set_aspect("equal", adjustable="box")
  for ax in low:
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.grid(alpha=0.15)
  fig.suptitle(
    f"CAD pad: {pad.geometry.cols} x {pad.geometry.rows} units, "
    f"{1000 * pad.geometry.spans[1] / (pad.geometry.cols - 1):g} x "
    f"{1000 * pad.geometry.spans[0] / (pad.geometry.rows - 1):g} mm spacing; moments about C\n"
    "Unit-data predictions under IID / first-unit release assumptions; not measured pad reliability",
    fontsize=10,
  )
  save_three(fig, STEM)
  plt.close(fig)
  np.savez_compressed(
    STEM.with_suffix(".npz"),
    force_ft=fx,
    force_fl=fy,
    force_hold10=sheets[0.1],
    force_hold50=sheets[0.5],
    force_hold90=sheets[0.9],
    bending_ft=bx,
    bending_ml=by,
    bending_mt=bz,
    twist_ft=tx,
    twist_mn=ty,
    twist_ml_upper=pitch_pos,
    twist_ml_lower=pitch_neg,
  )
  STEM.with_suffix(".caption.txt").write_text(
    f"Same layout/style as prior figures. {pad.geometry.cols} lateral by {pad.geometry.rows} rake units, "
    f"lateral span {1000 * pad.geometry.spans[1]:g} mm, rake span {1000 * pad.geometry.spans[0]:g} mm. "
    "All moments are about contact-array center C. "
    "1024 Monte Carlo draws (seed 905), independent units with fixed per-engagement quantile ranks. "
    "Curves are modeled holding-probability contours, not confidence intervals or measured full-pad reliability. "
    "Early loading failure draws remain failed draws; they are never dropped from the denominator. "
    "Missing regions have no supported boundary for this path and probability. "
    "Bending/twist sections use Fn=-60 N (compression); this is not extra free support at Fn=0. "
    "For each fixed Ft,Mn, first establish Fn=-60 N at zero bending, then increase bending along a ray; "
    "draws failing that initial state cannot recover on this path. These are path-survival boundaries, "
    "not an enumeration of disconnected statically admissible wrench sets. "
    "The unit data use 10 N seating, 0.5 N normal increments and 3 s dwell. Negative lateral is reflected. "
    "Censored tails are conservatively capped; unsupported shear/lateral loads are rejected. "
    "The 10 N seating preload is not a compression failure limit. "
    "Pad force/moment allocation and IID assumptions still require whole-pad hardware validation.\n"
  )
  # Compact data validation panel, including the sparse-cell baseline.
  audit = json.loads(EMPIRICAL_UNIT_REPORT.read_text())
  held = [r for r in audit["validation"]["trials"] if "observed_n" in r]
  fig, axs = plt.subplots(1, 3, figsize=(10.2, 2.8), layout="constrained")
  x = np.array([r["observed_n"] for r in held])
  y = np.array([r["predicted_median_n"] for r in held])
  axs[0].scatter(x, y, s=10, alpha=0.65, color=colors[0.5])
  axs[0].plot([-10, 10], [-10, 10], "--", color="gray", lw=0.8)
  axs[0].set(
    xlabel="Held-out terminal command [N]",
    ylabel="Predicted unit median [N]",
    title="(a) Spatial validation placements",
  )
  paired = audit["paired_terminal_validation"]
  axs[1].bar(
    ["Direct cells", "Kernel CDF"],
    [paired["raw_10_90_coverage"], paired["kernel_10_90_coverage"]],
    color=[colors[0.1], colors[0.5]],
  )
  axs[1].axhline(0.8, color="gray", ls="--", lw=0.8)
  axs[1].set(ylim=(0, 1), ylabel="10-90% interval coverage", title="(b) Retrospective validation")
  u = np.linspace(0, 1, 1024)
  svals = [5, 15, 25]
  for s in svals:
    q, _, _, _ = pad.surface.query_numpy(s, 0, u)
    axs[2].plot(np.sort(q), u, label=f"S = {s} N")
  axs[2].set(xlabel="Unit normal threshold [N]", ylabel="Conditional CDF", title="(c) Distributions used by DR")
  axs[2].legend(fontsize=7, frameon=False)
  for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
  save_three(fig, STEM.with_name("fig_empirical_distribution_validation"))
  plt.close(fig)
  STEM.with_name("fig_empirical_distribution_validation.caption.txt").write_text(
    f"Retrospective same-panel spatial split: {audit['validation']['total']} validation trials, "
    f"{audit['validation']['supported']} in kernel-table support. Panels (a,b) use the same "
    f"{paired['terminal_count']} supported terminal-failure placements for both estimators. "
    "The dashed 80% line is nominal coverage, not a fitted result. Panel (c) uses the "
    "all-data refit for runtime, not validation predictions. Negative lateral symmetry, "
    "directional rank coupling and array independence are assumptions. A new independent "
    "surface/placement test remains necessary after this model-development split.\n"
  )
  print(f"Wrote {STEM}.png/.pdf/.svg and distribution validation figure")
  from scr_phybot_mjab.tools.microspine.wang_limit_surface import WangPadCfg

  fig, axs = plt.subplots(1, 2, figsize=(9, 3), layout="constrained")
  moments = np.linspace(0, 2.5, 101)
  cap, loaded = pad.draws_at(120.0)
  base_normal = -30 / pad.count
  hold = []
  for ml in moments:
    c = base_normal + ml * pad.allocation[:, 2, 4]
    hold.append(np.mean(loaded & np.all(c <= cap, axis=-1)))
  force_only = np.full_like(moments, hold[0])
  axs[0].plot(moments, hold, label="Full force-moment allocation", color="#165695")
  axs[0].plot(moments, force_only, "--", label="Force-only, same random draws", color="#899bb0")
  axs[0].set(
    xlabel="Peeling moment about C [N m]",
    ylabel="Modeled pad holding fraction",
    title="Numerical ablation: Ft=120 N, Fn=-30 N",
    ylim=(0, 1),
  )
  axs[0].legend(frameon=False, fontsize=7)
  s = np.linspace(-15, 30, 181)
  low, _, _, _ = pad.surface.query_numpy(s, 0, 0.1)
  med, _, _, _ = pad.surface.query_numpy(s, 0, 0.5)
  high, _, _, _ = pad.surface.query_numpy(s, 0, 0.9)
  axs[1].fill_between(s, low, high, alpha=0.18, color="#165695", label="Measured-data unit 10-90%")
  axs[1].plot(s, med, color="#165695", label="Measured-data unit median")
  wang = WangUnit(WangPadCfg())
  for f0, color in [(400, "#ab8d65"), (650, "#815d30")]:
    axs[1].plot(s, wang.normal_boundary(s, 0, f0), ls="--", color=color, label=f"Legacy Wang, pad F0={f0} N")
  axs[1].set(
    xlabel="Single-unit rake shear [N]", ylabel="Normal upper threshold [N]", title="Unit comparison; lateral=0"
  )
  axs[1].legend(frameon=False, fontsize=6)
  for ax in axs:
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
  stem = STEM.with_name("fig_empirical_model_comparison_cad")
  save_three(fig, stem)
  plt.close(fig)
  np.savez_compressed(
    stem.with_suffix(".npz"),
    peel_nm=moments,
    full_hold=hold,
    force_only_hold=force_only,
    unit_shear_n=s,
    unit_q10_n=low,
    unit_q50_n=med,
    unit_q90_n=high,
  )
  stem.with_suffix(".caption.txt").write_text(
    "Left: numerical force-moment ablation under identical persistent unit random draws, "
    "not hardware validation of pad moments. Right: unit conditional thresholds compared "
    "with the repository's previous Wang-inspired parameterization (F0=400..650 N per pad), "
    "not newly digitized paper observations. Shading is the conditional unit distribution, "
    "not a confidence interval or pad distribution.\n"
  )


if __name__ == "__main__":
  main()
