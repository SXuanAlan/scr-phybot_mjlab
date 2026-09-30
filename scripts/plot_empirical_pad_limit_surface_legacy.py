"""Render the empirical pad prediction using the repository's original figures.

Reuses the original paper fonts, colors, 3D styling, and 1x3 / 2x3 layouts.
The underlying unit-cell median interpolation and pad mechanics are unchanged.
No points from the older Wang/GPU-validation figures are used.

Run in env_mjlab15:
  python scripts/plot_empirical_pad_limit_surface_legacy.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_empirical_pad_limit_surface import EmpiricalPad, Geometry, load_data, verify_model
from plot_pad_limit_surface import CAT, INK, MUT, ORDINAL, RC_PARAMS, SEC, save_three
from plot_pad_limit_surface_3d import style_3d, style_slice

ROOT = Path(__file__).resolve().parents[1]


def bending_ring(pad, ft, count=129):
  g = pad.geom
  z = float(pad.unit_boundary(ft / g.n, 0))
  psi = np.linspace(0, 2 * np.pi, count)
  if not np.isfinite(z) or z < 0:
    return np.full((count, 2), np.nan)
  c = z
  denom = np.abs(np.cos(psi)) * np.max(np.abs(g.x)) / g.sxx
  denom += np.abs(np.sin(psi)) * np.max(np.abs(g.y)) / g.syy
  radius = c / denom
  ring = radius[:, None] * np.stack([np.cos(psi), np.sin(psi)], axis=-1)
  assert np.max(np.abs(pad.margin(ft, ml=ring[:, 0], mt=ring[:, 1]))) < 1e-8
  return ring


def twist_peel_limits(pad, ft, mn):
  """Upper/lower Ml sheets for Fl=Fn=Mt=0, including data-support masking."""
  a, b, _ = pad.distribute(ft, mn=mn)
  unit_cap = pad.unit_boundary(a, b)
  g = pad.geom
  edge = np.abs(g.x) > 1e-12
  cap = np.min(
    unit_cap[..., edge]
    * g.sxx / np.abs(g.x[edge]), axis=-1,
  )
  valid = np.all(np.isfinite(unit_cap) & (unit_cap >= -1e-10), axis=-1)
  return np.where(valid, np.maximum(cap, 0), np.nan)


def format_axis(ax):
  ax.grid(alpha=0.25, linewidth=0.5)
  ax.spines[["top", "right"]].set_visible(False)
  ax.axhline(0, color=SEC, linewidth=0.5, alpha=0.5)
  ax.axvline(0, color=SEC, linewidth=0.5, alpha=0.5)


def draw_2d(pad, cache, stem):
  plt.rcParams.update(RC_PARAMS)
  fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.6), layout="constrained")
  g = pad.geom
  ft = np.linspace(g.n * pad.smin, g.n * pad.smax, 721)
  ax = axs[0]
  for ml, color in zip([0, 0.35, 0.7, 1.05], ORDINAL):
    z = pad.normal_boundary(ft, ml=ml)
    ax.plot(ft, z, color=color, linewidth=1.1)
    at = 250 if ml == 1.05 else 150 + 25 * ml
    label_z = float(pad.normal_boundary(at, ml=ml))
    if np.isfinite(label_z):
      ax.annotate(f"{ml:g}", (at, label_z), xytext=(0, 4),
                  textcoords="offset points", fontsize=6, color=SEC, ha="center")
  format_axis(ax)
  ax.set_xlim(ft[0], ft[-1])
  ax.set_ylim(-90, 75)
  ax.set_xlabel("$f_t$ [N]  (rake shear +)")
  ax.set_ylabel("$f_n$ [N]  (tension +)")
  ax.set_title("(a) force plane vs peel moment $|m_l|$ [N·m]", fontsize=7.5)
  ax.text(0.03, 0.96, "unit-derived prediction", transform=ax.transAxes,
          va="top", fontsize=5.5, color=SEC)

  ax = axs[1]
  for axis, label in [("ml", "$|m_l|$ pitch (peel)"),
                       ("mt", "$|m_t|$ roll"), ("mn", "$|m_n|$ twist")]:
    x, y = cache["moment_ft"], cache[f"{axis}_capacity"]
    ax.plot(x, y, color=CAT[axis], linewidth=1.4, label=label)
    lower = cache[f"{axis}_domain_lower_bound"]
    ii = np.flatnonzero(np.isfinite(lower))[::12]
    ax.scatter(x[ii], lower[ii], marker="^", s=9, facecolors="none",
                edgecolors=CAT[axis], linewidths=0.5)
  format_axis(ax)
  ax.set_xlim(0, g.n * pad.smax)
  ax.set_ylim(bottom=0)
  ax.set_xlabel("$f_t$ [N]  (rake-shear preload)")
  ax.set_ylabel("moment capacity [N·m]")
  ax.set_title("(b) moment capacity vs shear", fontsize=7.5)
  ax.legend(frameon=False, loc="upper left", fontsize=5.8, handlelength=1.1)
  ax.text(0.04, 0.04, "△ data-domain lower bound", transform=ax.transAxes,
          fontsize=5, color=SEC)

  ax = axs[2]
  for ft_value, color, above in [(70, MUT, False), (200, SEC, True)]:
    ring = bending_ring(pad, ft_value)
    ax.plot(ring[:, 0], ring[:, 1], color=color, linewidth=1.1)
    j = np.nanargmax(ring[:, 1]) if above else np.nanargmin(ring[:, 1])
    ax.annotate(f"$f_t$ = {ft_value:g} N", ring[j],
                xytext=(0, 5 if above else -5), textcoords="offset points",
                ha="center", va="bottom" if above else "top", fontsize=6, color=SEC)
  format_axis(ax)
  ax.set_aspect("equal")
  ax.margins(0.2)
  ax.set_xlabel("$m_l$ [N·m]  (pitch/peel)")
  ax.set_ylabel("$m_t$ [N·m]  (roll)")
  ax.set_title("(c) bending interaction", fontsize=7.5)
  save_three(fig, stem)
  plt.close(fig)


def draw_3d(pad, stem, slice_ft):
  plt.rcParams.update(RC_PARAMS)
  g = pad.geom
  shear = np.linspace(g.n * pad.smin, g.n * pad.smax, 101)
  lateral = np.linspace(-g.n * pad.lmax, g.n * pad.lmax, 121)
  fx, fy = np.meshgrid(shear, lateral)
  fz = pad.normal_boundary(fx, fy)
  fz = np.where(fz >= 0, fz, np.nan)  # original 3D panel shows tension half
  rings = np.stack([bending_ring(pad, ft) for ft in shear])
  bx = np.broadcast_to(shear[:, None], rings[..., 0].shape)

  twist = np.linspace(-14, 14, 281)
  tx, ty = np.meshgrid(shear, twist, indexing="ij")
  tz = twist_peel_limits(pad, tx, ty)
  finite = np.flatnonzero(np.isfinite(tz.ravel()))
  check = finite[::max(1, len(finite) // 200)]
  # Terminal tension sets these boundaries; the seating preload is not a
  # compression ceiling. Reconstruct zero margin at the boundary.
  assert np.max(np.abs(pad.margin(tx.ravel()[check], mn=ty.ravel()[check],
                                 ml=tz.ravel()[check]))) < 1e-8

  ring_e = bending_ring(pad, slice_ft)
  slice_fl = np.linspace(-g.n * pad.lmax, g.n * pad.lmax, 721)
  slice_fn = pad.normal_boundary(slice_ft, slice_fl)
  slice_twist = np.linspace(-14, 14, 1501)
  slice_peel = twist_peel_limits(pad, slice_ft, slice_twist)
  cut_style = dict(color=INK, linewidth=0.9, linestyle=(0, (3, 2)), alpha=0.7)

  fig = plt.figure(figsize=(7.16, 4.35))
  gs = fig.add_gridspec(2, 3, height_ratios=[1.62, 1], left=0.075, right=0.965,
                        top=0.99, bottom=0.09, wspace=0.28, hspace=0.18)
  top = [fig.add_subplot(gs[0, i], projection="3d") for i in range(3)]
  lower = [fig.add_subplot(gs[1, i]) for i in range(3)]
  surface_style = dict(cmap="Blues", alpha=0.72, linewidth=0.15,
                        edgecolor=(1, 1, 1, 0.18), antialiased=True)

  ax = top[0]
  ax.plot_surface(fx, fy, fz, vmin=0, vmax=np.nanmax(fz) * 1.15, **surface_style)
  pos = pad.values >= 0
  ax.scatter(g.n * pad.points[pos, 0], g.n * pad.points[pos, 1], g.n * pad.values[pos],
              s=4.5, color="#e34948", depthshade=False, label=f"unit medians × {g.n}")
  ax.legend(frameon=False, loc="upper left", fontsize=5.5, handletextpad=0.1,
             borderaxespad=0.1)
  cut = np.where(slice_fn >= 0, slice_fn, np.nan)
  ax.plot(np.full_like(slice_fl, slice_ft), slice_fl, cut, **cut_style)
  style_3d(ax, "$f_t$ [N]", "$f_l$ [N]", "$f_n$ [N]", (1.15, 1, 0.45))
  ax.set_xticks([-150, 0, 150, 300])
  ax.set_yticks([-300, 0, 300])
  ax.set_zticks([0, 50, 100])
  ax.set_zlim(0, 110)
  ax.view_init(elev=24, azim=-58)
  ax.set_title("(a) force boundary ($m$ = 0)", fontsize=7.5, pad=-2)

  ax = top[1]
  ax.plot_surface(bx, rings[..., 0], rings[..., 1], **surface_style)
  ax.plot(np.full(len(ring_e), slice_ft), ring_e[:, 0], ring_e[:, 1], **cut_style)
  style_3d(ax, "$f_t$ [N]", "$m_l$ [N·m]", "$m_t$ [N·m]", (1.3, 0.8, 0.8))
  ax.set_xticks([0, 150, 300])
  ax.set_yticks([-1, 0, 1])
  ax.set_zticks([-1, 0, 1])
  ax.view_init(elev=20, azim=-60)
  ax.set_title("(b) bending–shear boundary", fontsize=7.5, pad=-2)

  ax = top[2]
  for sign in [1, -1]:
    ax.plot_surface(tx, ty, sign * tz, vmin=-2, vmax=2, **surface_style)
    ax.plot(np.full(len(slice_twist), slice_ft), slice_twist,
             sign * slice_peel, **cut_style)
  style_3d(ax, "$f_t$ [N]", "$m_n$ [N·m]", "$m_l$ [N·m]", (1.3, 1, 0.6))
  ax.set_xticks([0, 150, 300])
  ax.set_yticks([-8, 0, 8])
  ax.set_zticks([-1, 0, 1])
  ax.view_init(elev=26, azim=-52)
  ax.set_title("(c) twist–pitch–shear boundary", fontsize=7.5, pad=-2)

  fill_c, edge_c = "#cde2fb", "#2a78d6"
  ax = lower[0]
  ax.fill_between(slice_fl, -60, slice_fn, where=slice_fn >= -60, color=fill_c, alpha=0.55)
  ax.plot(slice_fl, slice_fn, color=edge_c, linewidth=1.1)
  ax.set_xlim(slice_fl[0], slice_fl[-1])
  ax.set_ylim(-60, max(40, float(np.nanmax(slice_fn)) * 1.2))
  style_slice(ax, "$f_l$ [N]", "$f_n$ [N]", f"(d) forces at $f_t$ = {slice_ft:g} N")

  ax = lower[1]
  ax.fill(ring_e[:, 0], ring_e[:, 1], color=fill_c, alpha=0.55)
  ax.plot(ring_e[:, 0], ring_e[:, 1], color=edge_c, linewidth=1.1)
  ax.set_aspect("equal")
  ax.margins(0.15)
  style_slice(ax, "$m_l$ [N·m]", "$m_t$ [N·m]", f"(e) bending at $f_t$ = {slice_ft:g} N")

  ax = lower[2]
  ax.fill_between(slice_twist, -slice_peel, slice_peel, color=fill_c, alpha=0.55)
  ax.plot(slice_twist, slice_peel, color=edge_c, linewidth=1.1)
  ax.plot(slice_twist, -slice_peel, color=edge_c, linewidth=1.1)
  used = slice_twist[np.isfinite(slice_peel)]
  ax.set_xlim(used.min() * 1.12, used.max() * 1.12)
  style_slice(ax, "$m_n$ [N·m]", "$m_l$ [N·m]", f"(f) twist–pitch at $f_t$ = {slice_ft:g} N")
  save_three(fig, stem)
  plt.close(fig)
  np.savez_compressed(stem.with_suffix(".npz"), force_ft=fx, force_fl=fy, force_fn=fz,
                      bending_ft=bx, bending_ml=rings[..., 0], bending_mt=rings[..., 1],
                      twist_ft=tx, twist_mn=ty, twist_ml_upper=tz,
                      slice_ft=slice_ft, slice_fl=slice_fl, slice_fn=slice_fn,
                      slice_twist=slice_twist, slice_peel=slice_peel, slice_bending=ring_e)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-stem", type=Path,
                      default=ROOT / "docs/figures/fig_pad_limit_surface_empirical")
  parser.add_argument("--out-stem", type=Path,
                      default=ROOT / "docs/figures/fig_pad_limit_surface_empirical_legacy")
  parser.add_argument("--slice-ft", type=float, default=70)
  args = parser.parse_args()
  audit = json.loads(args.input_stem.with_suffix(".audit.json").read_text())
  source = Path(audit["source"])
  if hashlib.sha256(source.read_bytes()).hexdigest() != audit["source_sha256"]:
    raise ValueError("Source changed; regenerate the empirical audit and cache first.")
  _, cells = load_data(source)
  pad = EmpiricalPad(cells, Geometry(**audit["geometry"]), audit["normal_preload_n"])
  print(json.dumps(verify_model(pad), indent=2))
  with np.load(args.input_stem.with_suffix(".npz")) as cache:
    draw_2d(pad, cache, args.out_stem)
  three_d_stem = args.out_stem.with_name(args.out_stem.name + "_3d")
  draw_3d(pad, three_d_stem, args.slice_ft)
  shared = (
    "The figures reuse the original pad figures' layout, fonts, colors, and views, "
    "but use the empirical single-unit terminal-command cell medians and equal-stiffness "
    "pad model recorded in fig_pad_limit_surface_empirical.audit.json. The data, cell "
    "statistic, and interpolation have not been changed or smoothed. All pad boundaries "
    "are characteristic predictions, not pad hardware measurements or reliability bounds. "
    "Negative lateral forces assume mirror symmetry. Curves/sheets are truncated where "
    "the unit force data no longer constrain the model; these open ends are not ultimate "
    "capacity limits. No old Wang-model GPU validation points are overlaid. "
  )
  args.out_stem.with_suffix(".caption.txt").write_text(
    shared + "(a) Force-normal boundaries at peeling moments 0, 0.35, 0.7, and 1.05 N*m; "
    "all other lateral loads and moments are zero. (b) Individual moment capacities at "
    "zero normal/lateral force; open triangles mark data-domain lower bounds rather than "
    "predicted failure. (c) Bending interaction at Ft=70 and 200 N.\n"
  )
  three_d_stem.with_suffix(".caption.txt").write_text(
    shared + f"(a) Tensile half of the zero-moment force boundary, with unit medians "
    f"multiplied by {pad.geom.n} shown as red input markers, not pad experiments. "
    "(b) Peeling/rolling boundary versus shear at Fl=Fn=Mn=0. "
    "(c) Upper/lower peeling-moment sheets versus twist and shear at Fl=Fn=Mt=0; "
    "these do not add artificial closing faces at data-domain boundaries. "
    f"(d-f) Corresponding slices at Ft={args.slice_ft:g} N, shown by dashed cuts above. "
    "Shaded areas are model-admissible slices within the displayed data-supported region.\n"
  )


if __name__ == "__main__":
  main()
