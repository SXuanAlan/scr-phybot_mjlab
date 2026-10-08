"""3D views of the equivalent pad wrench limit surface (per-unit billing).

Three surfaces, each with the full rake-shear axis including its reverse
(against-rake) residual:

  (a) force body (m = 0): the closed Wang surface at pad scale in
      (f_t, f_l, f_n) — tension half only, the compression half (f_n < 0)
      is free and unbounded. Identical to the aggregate F0*R by the
      zero-moment collapse.
  (b) bending body: admissible (f_t, m_l, m_t) at f_l = f_n = m_n = 0 —
      the panel-(c) diamonds lofted along shear, pinching to the reverse
      capacity on the left and pure-shear capacity on the right.
  (c) twist-pitch body: admissible (f_t, m_n, m_l) at f_l = f_n = m_t = 0.

The moment bodies are built by per-shear radial scans in the moment plane
(the affine slice is not radial in the 6D wrench, so each cross-section is
scanned for its first occ = 1 crossing). Measured release points from the
rollout validation cache (``fig_pad_limit_surface_validation.npz``) that lie in
each slice are overlaid.

Run inside the env_mjlab15 environment:
  python scripts/microspine/plot_pad_limit_surface_3d.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from scr_phybot_mjab.tools.microspine.wang_pad_surface import HANG_FN, HANG_FT, PadSurface, load_limit_surface_cfg
from scr_phybot_mjab.tools.plotting import INK, RC_PARAMS, SEC, save_three, style_3d, style_slice

REL_C = "#e34948"


def force_surface_mesh(surf: PadSurface):
  """Full-azimuth closed force surface at pad scale (m = 0), numpy mesh.

  The LUT covers azimuth 0..180 deg; the surface mirrors in f_l. The billed
  envelope is F0 * max(R, r_floor), so the floor is applied — it renders the
  small isotropic scrap cap where the deterministic surface pinches to zero
  (pure tension, deep against-rake).
  """
  r_half = np.maximum(surf.lut.numpy(), surf.cfg.wang_r_floor)
  rr = np.concatenate([r_half, r_half[::-1][1:]], axis=0) * surf.f0
  theta = np.radians(np.linspace(0.0, 360.0, rr.shape[0]))
  phi = np.radians(np.linspace(0.0, 90.0, rr.shape[1]))
  tt, pp = np.meshgrid(theta, phi, indexing="ij")
  x = rr * np.cos(pp) * np.cos(tt)
  y = rr * np.cos(pp) * np.sin(tt)
  z = rr * np.sin(pp)
  return x[::4, ::2], y[::4, ::2], z[::4, ::2]


def moment_body_mesh(
  surf: PadSurface,
  axis1: int,
  axis2: int,
  r1_max: float,
  r2_max: float,
  n_ft: int = 84,
  n_psi: int = 97,
):
  """Loft the (f_t, m_axis1, m_axis2) body from per-shear contour rings.

  Returns meshes (X = f_t, Y = m_axis1, Z = m_axis2) plus a normalized
  radius field for coloring. Each cross-section boundary is a grid contour
  (see ``_closed_boundary``: the slices are re-entrant, a radial scan would
  clip the diagonal relief lobes), resampled to a shared seam so the rows
  stitch into a surface.
  """
  rev = surf.f0 * float(
    torch.clamp(
      torch.as_tensor(surf.cfg.wang_r_floor),
      min=surf.lut[-1, 0],
    )
  )
  ft_vals = torch.cat(
    [
      torch.linspace(-0.999 * rev, 0.0, n_ft // 6),
      torch.linspace(0.0, 0.9995 * surf.f0, n_ft - n_ft // 6),
    ]
  )
  rows = []
  for ft in ft_vals:
    m1, m2, occ = moment_slice_grid(surf, float(ft), axis1, axis2, r1_max, r2_max)
    ring = _closed_boundary(m1, m2, occ, n_psi)
    rows.append(ring if ring is not None else np.zeros((n_psi, 2)))
  yz = np.stack(rows)  # (n_ft, n_psi, 2)
  x = np.repeat(ft_vals.numpy()[:, None], n_psi, axis=1)
  rad = np.hypot(yz[..., 0] / r1_max, yz[..., 1] / r2_max)
  rad = rad / max(float(rad.max()), 1.0e-9)
  return x, yz[..., 0], yz[..., 1], rad


def _closed_boundary(x: np.ndarray, y: np.ndarray, occ: np.ndarray, n_psi: int) -> np.ndarray | None:
  """Longest occ = 1 contour as a closed, CCW, arc-length-resampled loop.

  Grid-contour based on purpose: the fixed-shear slices are affine (not
  radial) cuts of the 6D set and are NOT star-shaped — diagonal twist+pitch
  directions grow genuine relief lobes where twist shifts in-plane load off
  the tension-loaded corner unit. A radial first-crossing scan clips them.
  """
  from contourpy import contour_generator

  lines = contour_generator(x=x, y=y, z=occ.T).lines(1.0)
  if not lines:
    return None
  ln = max(lines, key=len)
  if len(ln) < 4:
    return None
  if not np.allclose(ln[0], ln[-1]):
    ln = np.vstack([ln, ln[:1]])
  area = float(np.sum(ln[:-1, 0] * ln[1:, 1] - ln[1:, 0] * ln[:-1, 1]))
  if area < 0.0:
    ln = ln[::-1]
  i0 = int(np.argmax(ln[:-1, 0]))  # consistent seam: the +axis1 crossing
  ln = np.vstack([ln[i0:-1], ln[:i0], ln[i0 : i0 + 1]])
  s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ln[:, 0]), np.diff(ln[:, 1])))])
  tq = np.linspace(0.0, s[-1], n_psi)
  return np.stack([np.interp(tq, s, ln[:, 0]), np.interp(tq, s, ln[:, 1])], axis=-1)


def moment_slice_grid(
  surf: PadSurface,
  ft: float,
  axis1: int,
  axis2: int,
  r1_max: float,
  r2_max: float,
  n1: int = 241,
  n2: int = 201,
):
  """Occupancy grid of the (m_axis1, m_axis2) slice at fixed shear."""
  m1 = torch.linspace(-r1_max, r1_max, n1)
  m2 = torch.linspace(-r2_max, r2_max, n2)
  w = torch.zeros(n1, n2, 6)
  w[..., 0] = float(ft)
  m1g, m2g = torch.meshgrid(m1, m2, indexing="ij")
  w[..., axis1] = m1g
  w[..., axis2] = m2g
  return m1.numpy(), m2.numpy(), surf.occupancy6(w).numpy()


def force_slice_grid(surf: PadSurface, ft: float, fl_max: float, fn_lo: float, fn_hi: float):
  """Occupancy grid of the (f_l, f_n) slice at fixed shear (open below)."""
  fl = torch.linspace(-fl_max, fl_max, 481)
  fn = torch.linspace(fn_lo, fn_hi, 361)
  fl2, fn2 = torch.meshgrid(fl, fn, indexing="ij")
  zeros = torch.zeros_like(fl2)
  occ = surf.occupancy(torch.full_like(fl2, float(ft)), fl2, fn2, zeros, zeros, zeros)
  return fl.numpy(), fn.numpy(), occ.numpy()


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-stem", default="docs/figures/fig_pad_limit_surface_3d")
  parser.add_argument(
    "--points-npz",
    default="docs/figures/fig_pad_limit_surface_validation.npz",
    help="Rollout-validation cache to overlay (skipped if missing).",
  )
  parser.add_argument("--defaults", action="store_true")
  parser.add_argument("--f0", type=float, default=None)
  args = parser.parse_args()

  cfg = load_limit_surface_cfg(args.defaults)
  f0 = float(args.f0) if args.f0 is not None else float(cfg.f_t_max_range[0])
  surf = PadSurface(cfg, f0)

  # Moment spans for the scan scaling, from the single-axis capacity peaks.
  ft_probe = torch.linspace(0.0, f0, 81)
  peak = {axis: float(surf.moment_capacity(axis, ft_probe, n_m=1200)[0].max()) for axis in ("m_l", "m_t", "m_n")}
  print("peaks [N*m]: " + ", ".join(f"{k} {v:.2f}" for k, v in peak.items()) + f" (F0 = {f0:g} N)")

  fx, fy, fz = force_surface_mesh(surf)
  bx, by, bz, brad = moment_body_mesh(surf, axis1=4, axis2=3, r1_max=1.2 * peak["m_l"], r2_max=1.2 * peak["m_t"])
  tx, ty, tz, trad = moment_body_mesh(surf, axis1=5, axis2=4, r1_max=1.15 * peak["m_n"], r2_max=1.2 * peak["m_l"])

  pts = {"s1": None, "s2": None, "s3": None, "d": None, "e": None, "f": None}
  npz = Path(args.points_npz)
  if npz.exists():
    dat = np.load(npz)
    bp, bt = dat["bp"], dat["b_tags"]
    # m = 0 rays: the t-n family member of panel (a) plus the dedicated
    # force-hemisphere family (tag 3, when present in the cache).
    pts["s1"] = bp[((bt[:, 0] == 0) & (bt[:, 1] == 0)) | (bt[:, 0] == 3) | (bt[:, 0] == 4)]
    pts["s2"] = bp[(bt[:, 0] == 2) | ((bt[:, 0] == 1) & (bt[:, 1] <= 1))]
    pts["s3"] = bp[((bt[:, 0] == 1) & (bt[:, 1] == 2)) | (bt[:, 0] == 5)]
    # Rays measured exactly at the f_t = 70 N slices (panels d-f).
    pts["d"] = bp[bt[:, 0] == 4]
    pts["e"] = bp[(bt[:, 0] == 2) & (bt[:, 1] == 0)]
    pts["f"] = bp[bt[:, 0] == 5]
    print(f"overlaying release points: {len(pts['s1'])} force / {len(pts['s2'])} bending / {len(pts['s3'])} twist")
  else:
    print(f"note: {npz} not found; surfaces only")

  # f_t = 70 N (hang shear) cross-sections shown under each body, plus the
  # matching cut rings drawn on the bodies themselves.
  ft_s = HANG_FT
  cap70 = {axis: float(surf.moment_capacity(axis, torch.tensor([ft_s]))[0][0]) for axis in ("m_l", "m_t", "m_n")}
  e_m1, e_m2, e_occ = moment_slice_grid(surf, ft_s, 4, 3, 1.35 * cap70["m_l"], 1.35 * cap70["m_t"], 361, 321)
  f_m1, f_m2, f_occ = moment_slice_grid(surf, ft_s, 5, 4, 1.3 * cap70["m_n"], 1.45 * cap70["m_l"], 361, 321)
  ring_e = _closed_boundary(e_m1, e_m2, e_occ, 361)
  ring_f = _closed_boundary(f_m1, f_m2, f_occ, 361)
  d_fl, d_fn, d_occ = force_slice_grid(surf, ft_s, fl_max=0.72 * f0, fn_lo=-0.15 * f0, fn_hi=0.14 * f0)
  from contourpy import contour_generator

  ring_a = [ln for ln in contour_generator(x=d_fl, y=d_fn, z=d_occ.T).lines(1.0) if len(ln) >= 2]
  cut_style = dict(color=INK, lw=0.9, ls=(0, (3, 2)), alpha=0.7, zorder=9)

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib import cm

  plt.rcParams.update(RC_PARAMS)
  fig = plt.figure(figsize=(7.16, 4.35))
  gs = fig.add_gridspec(
    2,
    3,
    height_ratios=[1.62, 1.0],
    left=0.075,
    right=0.965,
    top=0.99,
    bottom=0.09,
    wspace=0.28,
    hspace=0.18,
  )
  axs = [fig.add_subplot(gs[0, i], projection="3d") for i in range(3)]
  axs2 = [fig.add_subplot(gs[1, i]) for i in range(3)]

  # (a) force body, colored by tension like the paper's Fig 8.
  ax = axs[0]
  ax.plot_surface(
    fx,
    fy,
    fz,
    cmap="Blues",
    vmin=0.0,
    vmax=float(fz.max()) * 1.15,
    alpha=0.72,
    linewidth=0.15,
    edgecolor=(1, 1, 1, 0.18),
    antialiased=True,
  )
  if pts["s1"] is not None and len(pts["s1"]):
    p = pts["s1"]
    ax.scatter(
      p[:, 0],
      p[:, 1],
      p[:, 2],
      s=4.5,
      color=REL_C,
      depthshade=False,
      zorder=10,
      label="measured release",
    )
    ax.legend(
      frameon=False,
      loc="upper left",
      fontsize=6.0,
      handletextpad=0.1,
      borderaxespad=0.1,
    )
  ax.text(-70.0, -170.0, 3.0, "reverse", fontsize=5.5, color=SEC)
  for ln in ring_a:
    seg = ln[ln[:, 1] >= -0.5]
    if len(seg) >= 2:
      ax.plot(np.full(len(seg), ft_s), seg[:, 0], seg[:, 1], **cut_style)
  style_3d(ax, "$f_t$ [N]", "$f_l$ [N]", "$f_n$ [N]", (1.15, 1.0, 0.45))
  ax.set_xticks([0, 200, 400])
  ax.set_yticks([-200, 0, 200])
  ax.set_zticks([0, 25, 50])
  ax.view_init(elev=24, azim=-58)
  ax.set_title("(a) force body ($m$ = 0)", fontsize=7.5, pad=-2)

  # (b) bending x shear body, colored by cross-section radius.
  ax = axs[1]
  ax.plot_surface(
    bx,
    by,
    bz,
    facecolors=cm.Blues(0.25 + 0.62 * brad),
    alpha=0.72,
    linewidth=0.15,
    edgecolor=(1, 1, 1, 0.18),
    antialiased=True,
  )
  if pts["s2"] is not None and len(pts["s2"]):
    p = pts["s2"]
    ax.scatter(
      p[:, 0],
      p[:, 4],
      p[:, 3],
      s=4.5,
      color=REL_C,
      depthshade=False,
      zorder=10,
    )
  ax.text(-105.0, 0.0, 0.55, "reverse", fontsize=5.5, color=SEC)
  ax.plot(np.full(len(ring_e), ft_s), ring_e[:, 0], ring_e[:, 1], **cut_style)
  style_3d(ax, "$f_t$ [N]", "$m_l$ [N·m]", "$m_t$ [N·m]", (1.3, 0.8, 0.8))
  ax.set_xticks([0, 200, 400])
  ax.set_yticks([-1, 0, 1])
  ax.set_zticks([-1, 0, 1])
  ax.view_init(elev=20, azim=-60)
  ax.set_title("(b) bending–shear body", fontsize=7.5, pad=-2)

  # (c) twist-pitch x shear body.
  ax = axs[2]
  ax.plot_surface(
    tx,
    ty,
    tz,
    facecolors=cm.Blues(0.25 + 0.62 * trad),
    alpha=0.6,
    linewidth=0.15,
    edgecolor=(1, 1, 1, 0.18),
    antialiased=True,
  )
  if pts["s3"] is not None and len(pts["s3"]):
    p = pts["s3"]
    ax.scatter(
      p[:, 0],
      p[:, 5],
      p[:, 4],
      s=5.5,
      color=REL_C,
      depthshade=False,
      zorder=10,
    )
  ax.text(-105.0, 0.0, 0.5, "reverse", fontsize=5.5, color=SEC)
  ax.plot(np.full(len(ring_f), ft_s), ring_f[:, 0], ring_f[:, 1], **cut_style)
  style_3d(ax, "$f_t$ [N]", "$m_n$ [N·m]", "$m_l$ [N·m]", (1.3, 1.0, 0.6))
  ax.set_xticks([0, 200, 400])
  ax.set_yticks([-8, 0, 8])
  ax.set_zticks([-1, 0, 1])
  ax.view_init(elev=26, azim=-52)
  ax.set_title("(c) twist–pitch–shear body", fontsize=7.5, pad=-2)

  # (d-f) the dashed f_t = 70 N cuts, flattened: filled admissible regions.
  fill_c, edge_c = "#cde2fb", "#2a78d6"
  ax = axs2[0]
  ax.contourf(d_fl, d_fn, d_occ.T, levels=[0.0, 1.0], colors=[fill_c], alpha=0.55)
  ax.contour(d_fl, d_fn, d_occ.T, levels=[1.0], colors=[edge_c], linewidths=1.1)
  if pts["d"] is not None and len(pts["d"]):
    p = pts["d"]
    ax.scatter(p[:, 1], p[:, 2], s=5.0, color=REL_C, zorder=3, linewidths=0)
  ax.plot([0.0], [HANG_FN], marker="x", ms=4.0, mew=1.0, color=INK, ls="none")
  ax.annotate(
    "hang",
    xy=(0.0, HANG_FN),
    xytext=(0.06 * f0, HANG_FN + 0.05 * f0),
    fontsize=6.0,
    color=SEC,
    arrowprops=dict(arrowstyle="-", lw=0.5, color=SEC, shrinkB=2.0),
  )
  style_slice(ax, "$f_l$ [N]", "$f_n$ [N]", "(d) forces at $f_t$ = 70 N")

  ax = axs2[1]
  ax.contourf(e_m1, e_m2, e_occ.T, levels=[0.0, 1.0], colors=[fill_c], alpha=0.55)
  ax.contour(e_m1, e_m2, e_occ.T, levels=[1.0], colors=[edge_c], linewidths=1.1)
  if pts["e"] is not None and len(pts["e"]):
    p = pts["e"]
    ax.scatter(p[:, 4], p[:, 3], s=5.0, color=REL_C, zorder=3, linewidths=0)
  ax.set_aspect("equal")
  style_slice(ax, "$m_l$ [N·m]", "$m_t$ [N·m]", "(e) bending at $f_t$ = 70 N")

  ax = axs2[2]
  ax.contourf(f_m1, f_m2, f_occ.T, levels=[0.0, 1.0], colors=[fill_c], alpha=0.55)
  ax.contour(f_m1, f_m2, f_occ.T, levels=[1.0], colors=[edge_c], linewidths=1.1)
  if pts["f"] is not None and len(pts["f"]):
    p = pts["f"]
    ax.scatter(p[:, 5], p[:, 4], s=5.0, color=REL_C, zorder=3, linewidths=0)
  style_slice(ax, "$m_n$ [N·m]", "$m_l$ [N·m]", "(f) twist–pitch at $f_t$ = 70 N")

  save_three(fig, Path(args.out_stem))
  return 0


if __name__ == "__main__":
  sys.exit(main())
