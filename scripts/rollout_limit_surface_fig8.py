"""Fig-8-style validation figure for the Wang-2017 microspine limit surface.

Reproduces the layout of Wang et al. (IJRR 2017) Figure 8 — a 3D perspective
of the limit surface plus the (t,l), (t,n) and (n,l) cross-sections — with
the experimental scatter replaced by GPU-parallel sim rollouts:

  * 250 boundary rays: random tension-hemisphere directions pulled with a
    quasistatic per-ray force ramp until the hold envelope releases; the
    anchor load at release (last-3-step median) is a measured boundary point.
  * 250 interior rays: same direction sampling, ramped to a target at
    25-75 % of the predicted capacity and then held; the anchor must NOT
    release, and the measured steady load is an interior point.

All 500 rays run in one batched env on the production configuration (weld
anchors, wang2017 envelope, F0 pinned to SWEEP_PARAMS["f_t_max"]).

Run inside the env_mjlab environment:
  python scripts/rollout_limit_surface_fig8.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_microspine_constraint as vmc  # noqa: E402

from scr_phybot_mjab.microspine.config import (  # noqa: E402
  MicrospineLimitSurfaceCfg,
)
from scr_phybot_mjab.microspine.wang_limit_surface import (  # noqa: E402
  build_capacity_lut,
  capacity_ratio,
)

DEVICE = vmc.DEVICE


def sample_hemisphere_directions(count: int, device) -> torch.Tensor:
  """Uniform unit directions with a non-negative tension component (E, 3)."""
  d = torch.randn(count, 3, device=device)
  d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
  d[:, 2] = d[:, 2].abs()  # f_n >= 0: the anchor carries the whole load.
  return d


def surface_mesh(lut: torch.Tensor, f0: float):
  """Full-azimuth surface mesh in (f_t, f_l, f_n), numpy.

  The LUT covers azimuth 0..180 deg; the surface is mirror-symmetric in f_l,
  so the 180..360 deg half reuses the rows in reverse. One continuous mesh
  avoids the double-darkening artifact of overlaying two translucent halves.
  """
  import numpy as np

  n_theta, n_phi = lut.shape
  r_half = lut.cpu().numpy()
  rr = np.concatenate([r_half, r_half[::-1][1:]], axis=0) * f0
  theta = np.radians(np.linspace(0.0, 360.0, rr.shape[0]))
  phi = np.radians(np.linspace(0.0, 90.0, n_phi))
  tt, pp = np.meshgrid(theta, phi, indexing="ij")
  x = rr * np.cos(pp) * np.cos(tt)
  y = rr * np.cos(pp) * np.sin(tt)
  z = rr * np.sin(pp)
  return x, y, z


def nl_silhouette(lut: torch.Tensor, f0: float, bins: int = 181):
  """Projection silhouette of the surface onto the (f_l, f_n) plane.

  What the paper's (z, y) view shows: the outer envelope over all f_t. The
  f_t = 0 *section* would be misleading here — tension capacity grows so
  quickly with rake shear that points with even a few N of f_t legitimately
  sit far outside that section.

  Computed by dense direction sampling (the (theta, phi) mesh maps very
  unevenly onto (l, n) angles; naive binning leaves empty-bin spikes), plus
  a 3-bin running max to close residual dropouts.
  """
  import numpy as np

  theta = torch.linspace(0.0, math.pi, 2881)
  phi = torch.linspace(0.0, math.pi / 2.0, 721)
  tt, pp = torch.meshgrid(theta, phi, indexing="ij")
  d_t = torch.cos(pp) * torch.cos(tt)
  d_l = torch.cos(pp) * torch.sin(tt)
  d_n = torch.sin(pp)
  dev = lut.device
  rr = (
    capacity_ratio(lut, d_t.to(dev), d_l.to(dev), d_n.to(dev)).cpu() * f0
  )
  y = (rr * torch.cos(pp) * torch.sin(tt)).numpy().ravel()
  z = (rr * torch.sin(pp)).numpy().ravel()
  y = np.concatenate([y, -y])  # mirror symmetry in f_l
  z = np.concatenate([z, z])
  ang = np.arctan2(z, y)  # z >= 0 -> [0, pi], +l to -l
  rad = np.hypot(y, z)
  idx = np.clip((ang / math.pi * (bins - 1)).round().astype(int), 0, bins - 1)
  sil = np.zeros(bins)
  np.maximum.at(sil, idx, rad)
  sil = np.maximum(sil, np.maximum(np.roll(sil, 1), np.roll(sil, -1)))
  a = np.linspace(0.0, math.pi, bins)
  return sil * np.cos(a), sil * np.sin(a)


def section_curve(lut: torch.Tensor, f0: float, plane: str):
  """Model boundary curve of one cross-section, numpy (x, y) arrays.

  "tl": wall plane at f_n = 0; "tn": f_l = 0 plane.
  """
  import numpy as np

  ang = torch.linspace(0.0, math.pi, 721)
  if plane == "tl":
    d = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1)
  else:  # tn
    d = torch.stack([torch.cos(ang), torch.zeros_like(ang), torch.sin(ang)], -1)
  d = d.to(lut.device)
  r = capacity_ratio(lut, d[:, 0], d[:, 1], d[:, 2]).cpu().numpy() * f0
  a = ang.cpu().numpy()
  return r * np.cos(a), r * np.sin(a)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", default="docs/wang2017_limit_surface_fig8.png")
  parser.add_argument("--rays", type=int, default=250, help="per point class")
  parser.add_argument("--seconds", type=float, default=6.0)
  parser.add_argument("--ee-index", type=int, default=2, choices=range(4))
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--replot",
    action="store_true",
    help="Re-render the figure from the cached .npz next to --out (no sim).",
  )
  args = parser.parse_args()

  if args.replot:
    dat = np.load(os.path.splitext(args.out)[0] + ".npz")
    # Shape parameters are the dataclass defaults (the climb cfg only
    # overrides model/F0-range, neither of which changes the LUT).
    lut = build_capacity_lut(MicrospineLimitSurfaceCfg(), "cpu")
    render_figure(
      dat["bp"], dat["ip"], float(dat["f0"]), lut, args.rays, args.out
    )
    return 0

  torch.manual_seed(args.seed)
  num_envs = 2 * args.rays
  target = args.ee_index
  f0 = vmc.SWEEP_PARAMS["f_t_max"]
  print(
    f"== fig8 rollout: {args.rays} boundary + {args.rays} interior rays, "
    f"EE {vmc.EE_NAMES[target]}, F0 = {f0:.0f} N =="
  )

  env = vmc.build_env(num_envs, anchor_lock_rotation=True)
  env.reset()
  vmc.set_gravity(env, 90.0, magnitude=0.0)
  executor, affordance = vmc.get_terms(env)
  executor.assert_now(env)
  robot = env.scene["robot"]
  actions = vmc.hold_pose_actions(env)
  ls_cfg = affordance._ls_cfg
  lut = affordance._wang_lut
  if lut is None:
    raise SystemExit("The task is not configured with the wang2017 envelope.")

  # Unbreakable envelope everywhere during the preload-nulling settle.
  big = 1.0e9
  affordance.limit_f_t_max[:] = big
  affordance.limit_f_rev_max[:] = big
  affordance.limit_mu_adhesion[:] = 0.0
  affordance.limit_f_n0[:] = big
  affordance.limit_mu_lateral[:] = 0.0
  affordance.limit_f_l0[:] = big
  vmc.quasistatic_settle(env, executor, affordance, actions)
  base = executor.compute_anchor_loads()[:, target].norm(dim=-1)
  print(f"  residual preload: p50 {base.quantile(0.5):.2f} N, max {base.max():.2f} N")

  for key, value in vmc.SWEEP_PARAMS.items():
    getattr(affordance, f"limit_{key}")[:, target] = value

  # Engage frame is constant under weld anchors (pad orientation locked).
  t_hat, l_hat, n_hat = affordance._engage_frame()
  t_hat, l_hat, n_hat = t_hat[:, target], l_hat[:, target], n_hat[:, target]

  d = sample_hemisphere_directions(num_envs, DEVICE)  # engage-frame (t, l, n)
  # Stratified boundary rays: 40 % in the wall plane (f_n = 0, full azimuth
  # circle — traces the heart-shaped section), 30 % in the capacity lobe
  # (elevation 2-33 deg, where the dome has structure), 30 % full hemisphere
  # (keeps the tension-dominant dead zone covered, where capacity collapses
  # to the r_floor residual and rays release near the origin).
  n_inplane = int(0.4 * args.rays)
  n_lobe = int(0.3 * args.rays)
  az = torch.empty(n_inplane, device=DEVICE).uniform_(-math.pi, math.pi)
  d[:n_inplane, 0] = torch.cos(az)
  d[:n_inplane, 1] = torch.sin(az)
  d[:n_inplane, 2] = 0.0
  az = torch.empty(n_lobe, device=DEVICE).uniform_(-math.pi, math.pi)
  el = torch.empty(n_lobe, device=DEVICE).uniform_(
    math.radians(2.0), math.radians(33.0)
  )
  sl = slice(n_inplane, n_inplane + n_lobe)
  d[sl, 0] = torch.cos(el) * torch.cos(az)
  d[sl, 1] = torch.cos(el) * torch.sin(az)
  d[sl, 2] = torch.sin(el)
  direction_w = (
    d[:, 0:1] * t_hat + d[:, 1:2] * l_hat + d[:, 2:3] * n_hat
  )
  cap = f0 * capacity_ratio(lut, d[:, 0], d[:, 1], d[:, 2]).clamp_min(
    ls_cfg.wang_r_floor
  )

  is_boundary = torch.arange(num_envs, device=DEVICE) < args.rays
  # Boundary rays ramp past their predicted capacity; interior rays stop at a
  # 25-75 % fraction of it and hold there for the rest of the rollout.
  interior_frac = torch.empty(num_envs, device=DEVICE).uniform_(0.25, 0.75)
  target_mag = torch.where(is_boundary, 2.2 * cap, interior_frac * cap)
  # Reach the target midway so interior rays hold for >= half the rollout.
  rate = target_mag / (0.5 * args.seconds)

  spine_ids, _ = robot.find_bodies([vmc.EE_NAMES[target]])
  zero_torque = torch.zeros(num_envs, 1, 3, device=DEVICE)
  zero_qd = torch.zeros_like(robot.data.joint_vel)
  zero_root = torch.zeros(num_envs, 6, device=DEVICE)
  released = torch.zeros(num_envs, dtype=torch.bool, device=DEVICE)
  corrupted = torch.zeros(num_envs, dtype=torch.bool, device=DEVICE)
  load_release = torch.zeros(num_envs, 3, device=DEVICE)
  load_hist = torch.zeros(3, num_envs, 3, device=DEVICE)
  tail_sum = torch.zeros(num_envs, 3, device=DEVICE)
  tail_n = 0

  steps = int(args.seconds / env.step_dt)
  for step in range(steps):
    # Quasistatic: kill momentum every step (see cmd_sweep for rationale).
    robot.write_root_link_velocity_to_sim(zero_root)
    robot.write_joint_velocity_to_sim(zero_qd)
    for key, value in vmc.SWEEP_PARAMS.items():
      getattr(affordance, f"limit_{key}")[:, target] = value
    magnitude = torch.minimum(
      rate * (step + 1) * env.step_dt, target_mag
    )
    force = torch.where(
      released.unsqueeze(-1),
      torch.zeros_like(direction_w),
      magnitude.unsqueeze(-1) * direction_w,
    )
    robot.write_external_wrench_to_sim(
      forces=force.unsqueeze(1), torques=zero_torque, body_ids=spine_ids
    )
    _, _, terminated, truncated, _ = env.step(actions)
    corrupted |= (terminated | truncated) & ~released
    load_hist = torch.roll(load_hist, 1, dims=0)
    load_hist[0] = affordance.anchor_load_tnl[:, target]
    newly = affordance.forced_release_this_step[:, target] & ~released & ~corrupted
    load_release = torch.where(
      newly.unsqueeze(-1), load_hist.median(dim=0).values, load_release
    )
    released |= newly
    if step >= steps - 25:
      tail_sum += affordance.anchor_load_tnl[:, target]
      tail_n += 1

  valid = ~corrupted
  # --- boundary statistics ------------------------------------------------
  b_hit = is_boundary & released & valid
  b_pts = load_release[b_hit]
  m_rel = vmc._envelope_margin(b_pts, ls_cfg, lut)
  print(
    f"  boundary: released {int(b_hit.sum())}/{int((is_boundary & valid).sum())}, "
    f"|margin| p50 {m_rel.abs().quantile(0.5):.3f}, "
    f"p90 {m_rel.abs().quantile(0.9):.3f}"
  )
  # --- interior statistics ------------------------------------------------
  i_held = ~is_boundary & ~released & valid
  i_pts = tail_sum[i_held] / max(tail_n, 1)
  m_held = vmc._envelope_margin(i_pts, ls_cfg, lut)
  n_int = int((~is_boundary & valid).sum())
  print(
    f"  interior: held {int(i_held.sum())}/{n_int} "
    f"(released {int((~is_boundary & released & valid).sum())}), "
    f"margin p10 {m_held.quantile(0.1):.3f} (all should be > 0)"
  )
  print(f"  excluded (mid-test reset): {int(corrupted.sum())} envs")

  bp = b_pts.cpu().numpy()
  ip = i_pts.cpu().numpy()
  np.savez(os.path.splitext(args.out)[0] + ".npz", bp=bp, ip=ip, f0=f0)
  render_figure(bp, ip, f0, lut, args.rays, args.out)
  env.close()
  return 0


def render_figure(bp, ip, f0: float, lut: torch.Tensor, rays: int, out: str):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  model_c, rel_c, held_c, ink = "#dc2626", "#2563eb", "#94a3b8", "#475569"
  # Surface extents per axis (the dome is a flat pancake: ~F0 along f_t,
  # ~0.7 F0 along f_l but only ~0.12 F0 along f_n). "Near the section plane"
  # must use an absolute threshold per off-plane axis, or tension points at
  # 80 % of the dome height would count as "in the f_n = 0 plane".
  extent = {0: f0, 1: 0.7 * f0, 2: 0.13 * f0}

  fig = plt.figure(figsize=(14.0, 9.5))
  gs = fig.add_gridspec(2, 2, height_ratios=[1.8, 1.0])

  def scatter_classes(ax, xi, yi, other):
    """Scatter both classes in a 2D view; fade points far from the plane."""
    near_thr = 0.15 * extent[other]
    for pts, color, size, label in (
      (ip, held_c, 12, "held interior load"),
      (bp, rel_c, 16, "measured release"),
    ):
      near = np.abs(pts[:, other]) < near_thr
      ax.scatter(
        pts[near, xi], pts[near, yi], s=size, color=color, label=label,
        zorder=3,
      )
      ax.scatter(
        pts[~near, xi], pts[~near, yi], s=size * 0.6, color=color,
        alpha=0.15, zorder=2,
      )
    ax.axhline(0.0, color=ink, lw=0.6, alpha=0.4)
    ax.axvline(0.0, color=ink, lw=0.6, alpha=0.4)
    ax.grid(alpha=0.25)

  # Top-left: wall plane (f_t, f_l), like the paper's (x, y) view.
  ax = fig.add_subplot(gs[0, 0])
  cx, cy = section_curve(lut, f0, "tl")
  ax.plot(cx, cy, color=model_c, lw=2, label="model (f_n = 0 section)")
  ax.plot(cx, -cy, color=model_c, lw=2)
  scatter_classes(ax, 0, 1, 2)
  ax.set_xlabel("f_t [N] (rake shear +)")
  ax.set_ylabel("f_l [N]")
  ax.set_title("wall plane")
  ax.set_aspect("equal")
  ax.legend(fontsize=9, frameon=False, loc="upper right")

  # Top-right: 3D perspective, colored by tension like the paper.
  ax3 = fig.add_subplot(gs[0, 1], projection="3d")
  sx, sy, sz = surface_mesh(lut, f0)
  # Single continuous mesh, colored by tension height like the paper's Fig 8.
  ax3.plot_surface(
    sx, sy, sz, rstride=6, cstride=6, cmap="Blues", vmin=0.0,
    vmax=float(sz.max()), alpha=0.55, linewidth=0,
  )
  ax3.scatter(ip[:, 0], ip[:, 1], ip[:, 2], s=8, color=held_c, alpha=0.7)
  ax3.scatter(bp[:, 0], bp[:, 1], bp[:, 2], s=12, color=rel_c)
  ax3.set_xlabel("f_t [N]")
  ax3.set_ylabel("f_l [N]")
  ax3.set_zlabel("f_n [N]")
  ax3.set_title("3D limit surface")
  ax3.view_init(elev=22, azim=-60)

  # Bottom-left: (f_t, f_n) cross-section, like the paper's (x, z) view.
  # No equal aspect: the dome is ~10x wider than tall, exactly like the
  # paper's stretched z axis in Fig 8.
  ax = fig.add_subplot(gs[1, 0])
  cx, cy = section_curve(lut, f0, "tn")
  ax.plot(cx, cy, color=model_c, lw=2, label="model (f_l = 0 section)")
  scatter_classes(ax, 0, 2, 1)
  ax.set_xlabel("f_t [N] (rake shear +)")
  ax.set_ylabel("f_n [N] (tension +)")
  ax.set_title("t-n plane")
  ax.legend(fontsize=9, frameon=False, loc="upper left")

  # Bottom-right: (f_l, f_n) view, like the paper's (z, y) view. The curve
  # is the projection silhouette (max over f_t), not the f_t = 0 section:
  # tension capacity grows steeply with rake shear, so any point with a few
  # N of f_t would sit far outside that section while being well inside the
  # surface. Every measured point must lie inside the silhouette.
  ax = fig.add_subplot(gs[1, 1])
  cx, cy = nl_silhouette(lut, f0)
  ax.plot(cx, cy, color=model_c, lw=2, label="model silhouette (max over f_t)")
  for pts, color, size, label in (
    (ip, held_c, 12, "held interior load"),
    (bp, rel_c, 16, "measured release"),
  ):
    ax.scatter(pts[:, 1], pts[:, 2], s=size, color=color, label=label, zorder=3)
  ax.axhline(0.0, color=ink, lw=0.6, alpha=0.4)
  ax.axvline(0.0, color=ink, lw=0.6, alpha=0.4)
  ax.grid(alpha=0.25)
  ax.set_xlabel("f_l [N]")
  ax.set_ylabel("f_n [N] (tension +)")
  ax.set_title("n-l view (projection)")
  ax.legend(fontsize=9, frameon=False, loc="upper left")

  fig.suptitle(
    "Wang-2017 microspine limit surface vs GPU rollouts — "
    f"{rays} release + {rays} held rays, F0 = {f0:.0f} N",
    fontsize=13,
  )
  fig.text(
    0.01, 0.005,
    "faded points lie off the panel's section plane; the n-l view uses the "
    "projection silhouette, so all points fall inside it",
    fontsize=9, color=ink,
  )
  fig.tight_layout()
  fig.savefig(out, dpi=140)
  print(f"  figure saved to {out}")


if __name__ == "__main__":
  sys.exit(main())
