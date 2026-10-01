"""Legacy Wang whole-pad limit surface under per-unit moment billing.

For the active empirical model use plot_stochastic_pad_limit_surface.py.
This script explicitly selects the legacy Wang configuration for comparison.

Since 2026-08-04 the climb hold envelope (``moment_billing_mode="per_unit"``)
judges the pad's 6D anchor wrench by distributing it onto the 3x4 spine-unit
grid with rigid-pad equal-stiffness bolt-group statics
(``distribute_wrench_to_units``) and testing every unit's 3D load against its
F0/N share of the closed Wang-2017 surface — the weakest unit releases the
pad. The distribution is linear in the wrench and the unit criterion is
positively homogeneous (radial LUT ratio), so the pad-level admissible set

  W = { w in R^6 : max_i |f_i(w)|_+ / cap_i(dir f_i(w)) <= 1 }

is an exact star-shaped body whose boundary along any wrench ray is computable
in closed form per direction. This script computes that equivalent pad
surface, prints the derived capacity table (effective moment arms,
pure-moment capacities, force-moment interaction peaks), verifies the
zero-moment collapse onto the aggregate surface F0*R, and renders the paper
figure ``docs/figures/fig_pad_limit_surface.{svg,pdf,png}``.

The validation figure overlays measured GPU-rollout release points (cached
in ``docs/figures/fig_pad_limit_surface_validation.npz``) on these panels.

Conventions: engage frame (t = rake, down-wall; l = lateral; n = outward
normal). m_l is the pitch/peel bending about l (loads rake-edge rows in
tension), m_t the roll bending about t (loads lateral-edge columns), m_n the
twist about the normal (loads units in-plane). All three are sign-symmetric
under the symmetric unit grid; plots show |m|. Compression (f_n < 0) rides
the anchor for free (phi = 0 clamp). Everything scales linearly with the
per-latch F0 draw; plots use the DR floor.

Run inside the env_mjlab15 environment:
  python scripts/plot_pad_limit_surface.py
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import torch

from scr_phybot_mjab.microspine.wrench import distribute_wrench_to_units
from scr_phybot_mjab.tools.microspine.wang_limit_surface import (
  build_capacity_lut,
  capacity_ratio,
)
from scr_phybot_mjab.tools.plotting import CAT, INK, MUT, ORDINAL, RC_PARAMS, SEC, save_three

# Categorical slots (validated trio) for the three moment axes; ordinal blue
# ramp (light -> dark with |m_l|) for the panel-A family; neutral inks.
HANG_FT, HANG_FN = 70.0, 14.0  # 90-deg frame-0 per-hand hang load [N].
WANG_F0_RANGE_N = (400.0, 650.0)  # Per-pad F0 draw range of the Wang model.
BEND_LEVELS = (HANG_FT, 200.0)  # Panel-c shear preloads [N].


def load_limit_surface_cfg(use_defaults: bool):
  if use_defaults:
    from scr_phybot_mjab.microspine.config import MicrospineLimitSurfaceCfg

    return MicrospineLimitSurfaceCfg()
  from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
    phybot_c2_climb_wall_env_cfg,
  )

  climb = phybot_c2_climb_wall_env_cfg(play=True).commands["microspine_affordance"].microspine_cfg.limit_surface
  # The climb pad's CAD unit grid, drawn at the Wang per-pad F0 range.
  return dataclasses.replace(climb, f_t_max_range=WANG_F0_RANGE_N)


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


def check_zero_moment_collapse(surf: PadSurface) -> float:
  """Uniform split at m = 0 must reproduce the aggregate radial test."""
  gen = torch.Generator().manual_seed(0)
  f = torch.randn(20000, 3, generator=gen)
  f = 0.9 * surf.f0 * f / torch.linalg.norm(f, dim=-1, keepdim=True)
  zeros = torch.zeros(f.shape[0])
  occ_pad = surf.occupancy(f[:, 0], f[:, 1], f[:, 2], zeros, zeros, zeros)
  occ_agg = surf.occupancy_aggregate(f[:, 0], f[:, 1], f[:, 2])
  return float((occ_pad - occ_agg).abs().max())


def nice_family(peak: float, fractions=(0.3, 0.6, 0.9)) -> list[float]:
  step = 0.05 if peak < 2.0 else 0.1
  vals = sorted({round(f * peak / step) * step for f in fractions})
  return [v for v in vals if v > 0.0]


def compute_theory(surf: PadSurface) -> dict:
  """All arrays the panels (and the rollout validation) are built from."""
  f0 = surf.f0
  ft_grid = torch.linspace(0.0, f0, 241)
  axes_scan = {}
  islands = {}
  for axis in ("m_l", "m_t", "m_n"):
    cap, _, isl = surf.moment_capacity(axis, ft_grid)
    axes_scan[axis] = cap
    islands[axis] = isl
  peak_ml = float(axes_scan["m_l"].max())
  family = [0.0] + nice_family(peak_ml)

  a_ft = torch.linspace(-0.16 * f0, 1.08 * f0, 640)
  a_fn = torch.linspace(-0.22 * f0, 0.16 * f0, 560)
  ft2, fn2 = torch.meshgrid(a_ft, a_fn, indexing="ij")
  zeros = torch.zeros_like(ft2)
  a_occ = [surf.occupancy(ft2, zeros, fn2, zeros, torch.full_like(ft2, m), zeros) for m in family]

  c_m = torch.linspace(-1.35 * peak_ml, 1.35 * peak_ml, 480)
  ml2, mt2 = torch.meshgrid(c_m, c_m, indexing="ij")
  zeros = torch.zeros_like(ml2)
  c_occ = [surf.occupancy(torch.full_like(ml2, lv), zeros, zeros, mt2, ml2, zeros) for lv in BEND_LEVELS]
  return {
    "ft_grid": ft_grid,
    "axes_scan": axes_scan,
    "islands": islands,
    "peak_ml": peak_ml,
    "family": family,
    "a_ft": a_ft,
    "a_fn": a_fn,
    "a_occ": a_occ,
    "c_m": c_m,
    "c_occ": c_occ,
  }


def draw_theory_panels(axs, th: dict, f0: float) -> None:
  """Render the three theory panels onto three prepared axes."""
  import numpy as np

  for ax in axs:
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

  # (a) t-n force plane under increasing pitch/peel moment |m_l|.
  ax = axs[0]
  # Staggered direct labels: each value sits above its own curve's upper
  # branch at a distinct f_t so the nested family reads as a diagonal walk.
  label_x = [0.30 * f0, 0.425 * f0, 0.55 * f0, 0.675 * f0]
  for m_val, occ, color, lx in zip(th["family"], th["a_occ"], ORDINAL, label_x):
    cs = ax.contour(
      th["a_ft"].numpy(),
      th["a_fn"].numpy(),
      occ.numpy().T,
      levels=[1.0],
      colors=[color],
      linewidths=1.1,
    )
    verts = np.concatenate([p.vertices for p in cs.get_paths()])
    near = verts[np.abs(verts[:, 0] - lx) < 0.02 * f0]
    top = near[near[:, 1].argmax()] if len(near) else verts[verts[:, 1].argmax()]
    ax.annotate(
      f"{m_val:g}",
      xy=tuple(top),
      xytext=(top[0], top[1] + 0.008 * f0),
      ha="center",
      va="bottom",
      fontsize=6.0,
      color=SEC,
    )
  ax.plot([HANG_FT], [HANG_FN], marker="x", ms=4.5, mew=1.1, color=INK, ls="none")
  ax.annotate(
    "hang load",
    xy=(HANG_FT, HANG_FN),
    xytext=(-0.12 * f0, 0.125 * f0),
    fontsize=6.0,
    color=SEC,
    arrowprops=dict(arrowstyle="-", lw=0.5, color=SEC, shrinkB=2.5),
  )
  ax.axhline(0.0, color=SEC, lw=0.5, alpha=0.5)
  ax.axvline(0.0, color=SEC, lw=0.5, alpha=0.5)
  ax.set_xlabel("$f_t$ [N]  (rake shear +)")
  ax.set_ylabel("$f_n$ [N]  (tension +)")
  ax.set_title("(a) force plane vs peel moment $|m_l|$ [N·m]", fontsize=7.5)

  # (b) single-axis moment capacity vs rake-shear preload.
  ax = axs[1]
  labels = {
    "m_l": "$|m_l|$ pitch (peel)",
    "m_t": "$|m_t|$ roll",
    "m_n": "$|m_n|$ twist",
  }
  for axis in ("m_l", "m_t", "m_n"):
    ax.plot(
      th["ft_grid"].numpy(),
      th["axes_scan"][axis].numpy(),
      color=CAT[axis.replace("_", "")],
      lw=1.4,
      label=labels[axis],
    )
  # Unambiguous direct labels: m_l/m_t nearly coincide, so tag the pair
  # above/below at different shears; m_n beside its rising branch.
  ax.annotate(
    "$|m_n|$ twist",
    xy=(0.15 * f0, 0.0),
    xytext=(0.135 * f0, 4.35),
    ha="right",
    fontsize=6.5,
    color="#199e70",
  )
  ax.annotate(
    "$|m_t|$ roll",
    xy=(0.26 * f0, 0.0),
    xytext=(0.26 * f0, 1.62),
    ha="center",
    fontsize=6.5,
    color=CAT["mt"],
  )
  ax.annotate(
    "$|m_l|$ pitch",
    xy=(0.66 * f0, 0.0),
    xytext=(0.66 * f0, 0.52),
    ha="center",
    fontsize=6.5,
    color=CAT["ml"],
  )
  ax.set_xlabel("$f_t$ [N]  (rake-shear preload)")
  ax.set_ylabel("moment capacity [N·m]")
  ax.set_xlim(0.0, f0)
  ax.set_ylim(0.0, 1.12 * float(th["axes_scan"]["m_n"].max()))
  ax.legend(frameon=False, loc="upper right", handlelength=1.2)
  ax.set_title("(b) moment capacity vs shear", fontsize=7.5)

  # (c) bending-moment interaction at two shear preloads.
  ax = axs[2]
  for lv, occ, color, above in zip(BEND_LEVELS, th["c_occ"], (MUT, SEC), (False, True)):
    cs = ax.contour(
      th["c_m"].numpy(),
      th["c_m"].numpy(),
      occ.numpy().T,
      levels=[1.0],
      colors=[color],
      linewidths=1.1,
    )
    verts = np.concatenate([p.vertices for p in cs.get_paths()])
    apex = verts[verts[:, 1].argmax() if above else verts[:, 1].argmin()]
    ax.annotate(
      f"$f_t$ = {lv:g} N",
      xy=tuple(apex),
      xytext=(apex[0], apex[1] + (0.035 if above else -0.035)),
      ha="center",
      va="bottom" if above else "top",
      fontsize=6.0,
      color=SEC,
    )
  ax.axhline(0.0, color=SEC, lw=0.5, alpha=0.5)
  ax.axvline(0.0, color=SEC, lw=0.5, alpha=0.5)
  ax.set_xlabel("$m_l$ [N·m]  (pitch/peel)")
  ax.set_ylabel("$m_t$ [N·m]  (roll)")
  ax.set_aspect("equal")
  ax.set_title("(c) bending interaction", fontsize=7.5)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--out-stem",
    default="docs/figures/fig_pad_limit_surface",
    help="Output path stem; .svg/.pdf/.png are appended.",
  )
  parser.add_argument(
    "--defaults",
    action="store_true",
    help="Use MicrospineLimitSurfaceCfg defaults instead of the climb cfg.",
  )
  parser.add_argument(
    "--f0",
    type=float,
    default=None,
    help="Per-pad F0 [N] for the physical-unit plots (default: DR floor).",
  )
  args = parser.parse_args()

  cfg = load_limit_surface_cfg(args.defaults)
  if cfg.moment_billing_mode != "per_unit":
    print(
      f"note: cfg moment_billing_mode='{cfg.moment_billing_mode}'; the "
      "equivalent surface below is the per_unit billing regardless."
    )
  f0_lo, f0_hi = cfg.f_t_max_range
  f0 = float(args.f0) if args.f0 is not None else float(f0_lo)
  surf = PadSurface(cfg, f0)

  s_r, s_l = cfg.pad_span_rake_m, cfg.pad_span_lateral_m
  x_e, y_e = float(surf.ux.abs().max()), float(surf.uy.abs().max())
  r_c = math.hypot(x_e, y_e)
  arm_pitch = surf.sxx / (surf.n_units * x_e)
  arm_roll = surf.syy / (surf.n_units * y_e)
  arm_twist = surf.srr / (surf.n_units * r_c)

  print(
    f"pad: {cfg.unit_rows_rake}x{cfg.unit_cols_lateral} units, spans "
    f"{s_r * 1e3:.1f} x {s_l * 1e3:.1f} mm (rake x lateral), "
    f"F0 in [{f0_lo:g}, {f0_hi:g}] N, r_floor {cfg.wang_r_floor:g}; "
    f"plots at F0 = {f0:g} N (all capacities scale with F0)"
  )
  print(
    f"effective pad-scale moment arms: pitch |m_l| {arm_pitch * 1e3:.1f} mm "
    f"(= span_rake*(R+1)/(6(R-1))), roll |m_t| {arm_roll * 1e3:.1f} mm, "
    f"twist |m_n| {arm_twist * 1e3:.1f} mm (vs worst_edge fold "
    f"{cfg.moment_arm_fraction:g} * span = "
    f"{cfg.moment_arm_fraction * s_r * 1e3:.1f} / "
    f"{cfg.moment_arm_fraction * s_l * 1e3:.1f} mm)"
  )

  # float32 tolerance: at m = 0 the per-unit path divides by N before the
  # LUT's atan2/bilinear sample, so occupancy re-agrees with the aggregate
  # only to interpolation precision (~2e-4 observed), not exactly.
  collapse = check_zero_moment_collapse(surf)
  ok = collapse < 1.0e-3
  print(
    f"[{'PASS' if ok else 'FAIL'}] zero-moment collapse onto aggregate F0*R: max |occ_pad - occ_agg| = {collapse:.2e}"
  )

  th = compute_theory(surf)
  for axis, isl in th["islands"].items():
    if isl:
      print(f"note: {axis} scan found {isl} non-contiguous rows")
  hang_occ = float(
    surf.occupancy(
      torch.tensor(HANG_FT),
      torch.tensor(0.0),
      torch.tensor(HANG_FN),
      torch.tensor(0.0),
      torch.tensor(0.0),
      torch.tensor(0.0),
    )
  )
  hang_ml, _, _ = surf.moment_capacity("m_l", torch.tensor([HANG_FT]), f_n=HANG_FN)

  print("\nequivalent pad capacities (at F0 = %g N; x%.3f for F0 = %g N):" % (f0, f0_hi / f0, f0_hi))
  print(f"  pure rake shear        {f0:7.1f} N")
  phis = torch.linspace(0.0, 90.0, 361)
  rr = capacity_ratio(
    surf.lut,
    torch.cos(torch.deg2rad(phis)),
    torch.zeros_like(phis),
    torch.sin(torch.deg2rad(phis)),
  )
  fn_lobe = f0 * rr * torch.sin(torch.deg2rad(phis))
  i_pk = int(fn_lobe.argmax())
  print(f"  peak tension (w/ shear){fn_lobe[i_pk]:7.1f} N at elevation {float(phis[i_pk]):.1f} deg")
  lat = f0 * float(capacity_ratio(surf.lut, torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0)))
  rev = f0 * float(capacity_ratio(surf.lut, torch.tensor(-1.0), torch.tensor(0.0), torch.tensor(0.0)))
  print(f"  pure lateral shear     {lat:7.1f} N")
  print(f"  reverse shear (unhook) {rev:7.1f} N")
  for axis, label in (
    ("m_l", "pitch/peel |m_l|"),
    ("m_t", "roll       |m_t|"),
    ("m_n", "twist      |m_n|"),
  ):
    cap = th["axes_scan"][axis]
    i = int(cap.argmax())
    print(f"  {label}: {cap[0]:5.2f} N*m at f_t=0, peak {cap[i]:5.2f} N*m at f_t = {float(th['ft_grid'][i]):5.1f} N")
  print(
    f"  hang point (f_t={HANG_FT:g}, f_n={HANG_FN:g} N): occupancy "
    f"{hang_occ:.3f} (capacity multiple {1.0 / hang_occ:.2f}), "
    f"m_l headroom {float(hang_ml[0]):.2f} N*m"
  )

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update(RC_PARAMS)
  fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.6), layout="constrained")
  draw_theory_panels(axs, th, f0)
  save_three(fig, Path(args.out_stem))
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
