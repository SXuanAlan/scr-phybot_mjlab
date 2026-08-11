"""Fig-8-style rollout validation of the equivalent pad wrench limit surface.

Companion to ``scripts/plot_pad_limit_surface.py``: the same three theory
panels (t-n force plane under peel moment, single-axis moment capacity vs
shear, bending interaction), overlaid with measured GPU-rollout points that
now probe FORCE + MOMENT loading:

  * boundary rays: for theory boundary wrenches w* sampled from each panel's
    curves, ramp the applied 6D wrench proportionally, s * w*. Because the
    per-unit occupancy is positively homogeneous, the straight ray stays
    inside the surface until s = 1 and the release wrench (last-3-step
    median of the anchor force AND weld moment, engage frame) is a measured
    boundary point that should land on the theory curve of its panel.
  * interior rays: same directions ramped to s in [0.3, 0.8] and held; the
    anchor must NOT release, and the steady wrench is an interior point.

Loading rig (deviations from the deploy stack, all envelope-neutral):
zero gravity, quasistatic velocity kill, F0 pinned to the DR floor with the
recruitment ramp and seat-quality scaling disabled (the theory surface is
the fully seated, fully recruited envelope — both are capacity scalings,
not shape changes), tilt-peel disabled (a separate geometric release
channel), a stiffened rotational anchor (``anchor_torquescale`` 0.5) so
the commanded moment reaches the envelope through micro-rotations instead
of the deploy stack's compliant 0.01 (which sheds couples by design and
would need geometric rotations far past tilt-peel to transmit N*m-scale
moments — while the bench's 5.0 amplifies the velocity-zeroing rig's
angular jitter into one-step judged-moment spikes), and the release
debounce raised to 3 steps to ride out the rig's per-step impulse
transients (the deploy value 1 assumes smooth anchor-only load paths, not
an external wrench rewritten every step). The applied xfrc force acts at the body CoM
(mujoco_warp ``apply_ft`` offsets from ``xipos``); the commanded couple is
corrected by (pad_center - com) x F as feed-forward, and a 6D integral
servo on the measured anchor wrench (the exact quantity the billing
judges) cancels the residual parasitic paths (wrist PD couples, frame
offsets), so the realized wrench tracks the commanded ray in all six
components. The first second of the ramp runs under an unbreakable
envelope while the servo converges: the raw parasitic pitch moment has a
~16 mm effective arm, which alone holds every unit at ~33 deg elevation —
the closed forward lobe's edge — so an unconverged probe releases pads at
~10 N regardless of ray direction (observed before the warmup existed).

Run inside the env_mjlab15 environment (GPU):
  python scripts/rollout_pad_wrench_surface.py --device cuda:1
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
from plot_pad_limit_surface import (  # noqa: E402
  BEND_LEVELS,
  HANG_FT,
  MUT,
  RC_PARAMS,
  PadSurface,
  compute_theory,
  draw_theory_panels,
  load_limit_surface_cfg,
  save_three,
)

REL_C = "#e34948"  # measured release (failure) points
# 0-2: the 2D theory panels; 3+: extra families for the 3D companion figure
# (force hemisphere, and the two f_t = 70 N slice boundaries).
PANEL_A, PANEL_B, PANEL_C, PANEL_F, PANEL_D70, PANEL_G70 = 0, 1, 2, 3, 4, 5
B_AXES = ("m_t", "m_l", "m_n")  # wrench slot order 3, 4, 5


def _split_runs(line: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
  """Contiguous kept runs of a polyline under a vertex mask."""
  runs, start = [], None
  for i, keep in enumerate(mask):
    if keep and start is None:
      start = i
    elif not keep and start is not None:
      if i - start >= 2:
        runs.append(line[start:i])
      start = None
  if start is not None and len(line) - start >= 2:
    runs.append(line[start:])
  return runs


def _arc_resample(lines: list[np.ndarray], count: int) -> np.ndarray:
  """Evenly sample ``count`` points over polylines, by arc length."""
  lens = [
    float(np.hypot(np.diff(ln[:, 0]), np.diff(ln[:, 1])).sum())
    for ln in lines
  ]
  total = sum(lens)
  out = []
  for ln, ln_len in zip(lines, lens):
    k = max(1, round(count * ln_len / total))
    s = np.concatenate(
      [[0.0], np.cumsum(np.hypot(np.diff(ln[:, 0]), np.diff(ln[:, 1])))]
    )
    tq = np.linspace(0.0, s[-1], k + 2)[1:-1]
    out.append(
      np.stack(
        [np.interp(tq, s, ln[:, 0]), np.interp(tq, s, ln[:, 1])], axis=-1
      )
    )
  return np.concatenate(out)[:count]


def _contour_lines(x: torch.Tensor, y: torch.Tensor, occ: torch.Tensor):
  from contourpy import contour_generator

  cg = contour_generator(
    x=x.numpy(), y=y.numpy(), z=occ.numpy().T
  )
  return [ln for ln in cg.lines(1.0) if len(ln) >= 2]


def build_targets(
  surf: PadSurface, th: dict, per_curve: int = 12, per_axis: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Theory boundary wrenches for every panel.

  Returns (wrench6 (K, 6) in engage components (t, l, n, mt, ml, mn),
  tags (K, 2) = (panel id, member index), and preload6 (K, 6)). The ramp
  applies preload * warm_fraction + s * wrench6: zero-preload rays are
  plain proportional 6D rays (boundary exactly at s = 1 by homogeneity);
  preloaded rays hold a fixed component and ramp the rest, pinning the
  release into an exact slice. The twist-pitch slice needs this: its
  boundary has plateau-and-cliff direction sensitivity, so a ~1 % ray
  drift on a proportional ramp turns into a 10-25 % radial overshoot —
  still on the 6D surface, but at f_t well past the slice.
  """
  targets, tags = [], []
  preloads: list[list[float]] = []

  def add(w, tag, pre=None):
    targets.append(w)
    tags.append(tag)
    preloads.append(pre if pre is not None else [0.0] * 6)

  # Panel a: t-n boundary at each family |m_l|. Clip the compression rails
  # to shallow depth so arc sampling favors the tension lobe, and keep the
  # rest of each contour (upper branch, reverse lobe, moment re-closure).
  for mi, (m_val, occ) in enumerate(zip(th["family"], th["a_occ"])):
    lines = _contour_lines(th["a_ft"], th["a_fn"], occ)
    runs = []
    for ln in lines:
      runs.extend(_split_runs(ln, ln[:, 1] > -20.0))
    for ft_v, fn_v in _arc_resample(runs, per_curve):
      add([ft_v, 0.0, fn_v, 0.0, m_val, 0.0], [PANEL_A, mi])

  # Panel b: single-axis capacity curves, alternating moment sign (the
  # billing is sign-symmetric; the panel plots |m|).
  fracs = np.linspace(0.12, 0.94, per_axis)
  for ai, axis in enumerate(B_AXES):
    cap = th["axes_scan"][axis].numpy()
    ft_np = th["ft_grid"].numpy()
    for j, frac in enumerate(fracs):
      i = int(np.argmin(np.abs(ft_np - frac * surf.f0)))
      if cap[i] < 0.05:
        continue
      w = [float(ft_np[i]), 0.0, 0.0, 0.0, 0.0, 0.0]
      w[3 + ai] = float(cap[i]) * (1.0 if j % 2 == 0 else -1.0)
      add(w, [PANEL_B, ai])

  # Panel c: bending-interaction diamonds at the two shear preloads (the
  # closed contour covers all four sign quadrants).
  for li, (lv, occ) in enumerate(zip(BEND_LEVELS, th["c_occ"])):
    lines = _contour_lines(th["c_m"], th["c_m"], occ)
    for ml_v, mt_v in _arc_resample(lines, per_curve):
      add([float(lv), 0.0, 0.0, mt_v, ml_v, 0.0], [PANEL_C, li])

  # Force hemisphere (m = 0), for the 3D force body: fig8's stratification —
  # in-plane azimuth circle, capacity-lobe elevations, full tension
  # hemisphere. The zero-moment slice IS radial, so the boundary point along
  # direction u is exactly u / occ(u) — no contour scan needed.
  rng = np.random.default_rng(0)
  n_ring, n_lobe, n_hemi = 20, 14, 14
  az = rng.uniform(-math.pi, math.pi, n_ring + n_lobe)
  el = np.concatenate(
    [np.zeros(n_ring), np.radians(rng.uniform(2.0, 33.0, n_lobe))]
  )
  d = np.stack(
    [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1
  )
  h = rng.normal(size=(n_hemi, 3))
  h /= np.linalg.norm(h, axis=-1, keepdims=True)
  h[:, 2] = np.abs(h[:, 2])
  d = np.concatenate([d, h])
  u6 = np.zeros((len(d), 6), dtype=np.float32)
  u6[:, :3] = d * surf.f0
  occ = surf.occupancy6(torch.as_tensor(u6)).numpy()
  for w in u6 / occ[:, None]:
    add([float(v) for v in w], [PANEL_F, 0])

  # f_t = 70 N slice families for the 3D figure's bottom row: the force
  # slice boundary (m = 0, shallow-compression rails clipped like panel a)
  # and the twist-pitch ring (m_t = 0).
  ft_s = float(HANG_FT)
  fl = torch.linspace(-0.72 * surf.f0, 0.72 * surf.f0, 481)
  fn = torch.linspace(-0.15 * surf.f0, 0.14 * surf.f0, 361)
  fl2, fn2 = torch.meshgrid(fl, fn, indexing="ij")
  zeros = torch.zeros_like(fl2)
  occ_d = surf.occupancy(
    torch.full_like(fl2, ft_s), fl2, fn2, zeros, zeros, zeros
  )
  runs = []
  for ln in _contour_lines(fl, fn, occ_d):
    runs.extend(_split_runs(ln, ln[:, 1] > -20.0))
  for fl_v, fn_v in _arc_resample(runs, per_curve):
    add([ft_s, fl_v, fn_v, 0.0, 0.0, 0.0], [PANEL_D70, 0])

  cap_mn = float(surf.moment_capacity("m_n", torch.tensor([ft_s]))[0][0])
  cap_ml = float(surf.moment_capacity("m_l", torch.tensor([ft_s]))[0][0])
  mn = torch.linspace(-1.25 * cap_mn, 1.25 * cap_mn, 401)
  ml = torch.linspace(-1.35 * cap_ml, 1.35 * cap_ml, 321)
  mn2, ml2 = torch.meshgrid(mn, ml, indexing="ij")
  zeros = torch.zeros_like(mn2)
  occ_g = surf.occupancy(
    torch.full_like(mn2, ft_s), zeros, zeros, zeros, ml2, mn2
  )
  # Two-phase rays: hold the shear preload fixed, ramp only the moments, so
  # the release is pinned into the exact f_t = 70 N slice.
  for mn_v, ml_v in _arc_resample(_contour_lines(mn, ml, occ_g), per_curve):
    add(
      [0.0, 0.0, 0.0, 0.0, ml_v, mn_v],
      [PANEL_G70, 0],
      pre=[ft_s, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

  return (
    np.asarray(targets, dtype=np.float32),
    np.asarray(tags),
    np.asarray(preloads, dtype=np.float32),
  )


def render_figure(
  th: dict,
  f0: float,
  bp: np.ndarray,
  ip: np.ndarray,
  b_tags: np.ndarray,
  i_tags: np.ndarray,
  out_stem: str,
) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from pathlib import Path

  plt.rcParams.update(RC_PARAMS)
  fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.6), layout="constrained")
  draw_theory_panels(axs, th, f0)

  def scatter(ax, pts, xy, released):
    if len(pts) == 0:
      return None
    return ax.scatter(
      xy[0], xy[1],
      s=7.0 if released else 5.0,
      color=REL_C if released else MUT,
      linewidths=0.0, zorder=3.0 if released else 2.5,
    )

  handles = {}
  for pts, tags, released in ((ip, i_tags, False), (bp, b_tags, True)):
    a = pts[tags[:, 0] == PANEL_A]
    h = scatter(axs[0], a, (a[:, 0], a[:, 2]), released)
    b_sel = tags[:, 0] == PANEL_B
    b = pts[b_sel]
    m_probe = np.abs(
      b[np.arange(len(b)), 3 + tags[b_sel, 1]]
    ) if len(b) else np.zeros(0)
    scatter(axs[1], b, (b[:, 0], m_probe), released)
    c = pts[tags[:, 0] == PANEL_C]
    scatter(axs[2], c, (c[:, 4], c[:, 3]), released)
    if h is not None:
      handles["measured release" if released else "held (interior)"] = h

  axs[0].legend(
    handles.values(), handles.keys(),
    frameon=False, loc="lower left", handletextpad=0.2, borderpad=0.2,
  )
  save_three(fig, Path(out_stem))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--out-stem", default="docs/figures/fig_pad_limit_surface_validation"
  )
  parser.add_argument("--seconds", type=float, default=8.0)
  parser.add_argument("--ee-index", type=int, default=2, choices=range(4))
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--torquescale", type=float, default=0.5)
  parser.add_argument(
    "--replot",
    action="store_true",
    help="Re-render from the cached .npz next to --out-stem (no sim).",
  )
  args = parser.parse_args()

  ls_cfg = load_limit_surface_cfg(use_defaults=False)
  f0 = float(vmc.SWEEP_PARAMS["f_t_max"])
  surf = PadSurface(ls_cfg, f0)
  th = compute_theory(surf)

  if args.replot:
    dat = np.load(args.out_stem + ".npz")
    render_figure(
      th, f0, dat["bp"], dat["ip"], dat["b_tags"], dat["i_tags"],
      args.out_stem,
    )
    return 0

  torch.manual_seed(args.seed)
  w_boundary, tags_np, pre_np = build_targets(surf, th)
  n_boundary = len(w_boundary)
  interior_idx = np.arange(0, n_boundary, 3)
  n_interior = len(interior_idx)
  num_envs = n_boundary + n_interior
  target = args.ee_index
  print(
    f"== pad wrench surface rollout: {n_boundary} boundary + "
    f"{n_interior} interior rays, EE {vmc.EE_NAMES[target]}, "
    f"F0 = {f0:.0f} N, torquescale {args.torquescale:g}, {args.device} =="
  )

  vmc.DEVICE = args.device
  device = args.device
  env = vmc.build_env(
    num_envs,
    anchor_lock_rotation=True,
    anchor_torquescale=args.torquescale,
  )
  env.reset()
  vmc.set_gravity(env, 90.0, magnitude=0.0)
  executor, affordance = vmc.get_terms(env)
  executor.assert_now(env)
  robot = env.scene["robot"]
  actions = vmc.hold_pose_actions(env)
  if affordance._wang_lut is None or affordance._unit_x is None:
    raise SystemExit("Task is not configured with wang2017 per_unit billing.")

  # Unbreakable envelope everywhere; the target EE gets the pinned F0 after
  # settling. The probe measures the wrench envelope alone at full seated
  # capacity: the recruitment ramp would floor F0_eff at 0.3 F0 (the settle
  # holds zero shear, so the charge never fills) and the seat scale is an
  # engage-time lottery — both are capacity scalings the theory pins at 1,
  # so disable them at the cfg the judge reads each step.
  affordance._ls_cfg.capacity_ramp_time_s = 0.0
  affordance._mcfg.seat_quality_scales_capacity = False
  # The rig rewrites xfrc and zeroes velocities every step, which injects
  # one-step efc transients the smooth anchor-only load path of the deploy
  # stack never sees; debounce them instead of executing on a single spike.
  affordance._ls_cfg.violation_consecutive_steps = 3
  big = 1.0e9
  affordance.limit_f_t_max[:] = big
  affordance.limit_f_rev_max[:] = big
  affordance.limit_mu_adhesion[:] = 0.0
  affordance.limit_f_n0[:] = big
  affordance.limit_mu_lateral[:] = 0.0
  affordance.limit_f_l0[:] = big
  affordance.limit_tilt_peel_max[:] = big
  vmc.quasistatic_settle(env, executor, affordance, actions)
  base_f = executor.compute_anchor_loads()[:, target].norm(dim=-1)
  base_m = affordance.anchor_moment_tnl[:, target].norm(dim=-1)
  print(
    f"  residual preload: |F| p50 {base_f.quantile(0.5):.2f} N "
    f"(max {base_f.max():.2f}), |M| p50 {base_m.quantile(0.5):.3f} N*m "
    f"(max {base_m.max():.3f})"
  )

  def pin_params():
    affordance.limit_f_t_max[:, target] = f0
    affordance.limit_tilt_peel_max[:] = big

  pin_params()

  # Engage frame is constant under stiff weld anchors.
  t_hat, l_hat, n_hat = affordance._engage_frame()
  t_hat, l_hat, n_hat = t_hat[:, target], l_hat[:, target], n_hat[:, target]

  w6 = torch.zeros(num_envs, 6, device=device)
  w6[:n_boundary] = torch.as_tensor(w_boundary, device=device)
  w6[n_boundary:] = torch.as_tensor(w_boundary[interior_idx], device=device)
  pre6 = torch.zeros(num_envs, 6, device=device)
  pre6[:n_boundary] = torch.as_tensor(pre_np, device=device)
  pre6[n_boundary:] = torch.as_tensor(pre_np[interior_idx], device=device)
  is_boundary = torch.arange(num_envs, device=device) < n_boundary

  target_s = torch.where(
    is_boundary,
    torch.full((num_envs,), 1.6, device=device),
    torch.empty(num_envs, device=device).uniform_(0.3, 0.8),
  )
  rate = target_s / (0.55 * args.seconds)

  spine_ids, _ = robot.find_bodies([vmc.EE_NAMES[target]])
  ee_body_id = executor._ee_body_ids[target]
  zero_qd = torch.zeros_like(robot.data.joint_vel)
  zero_root = torch.zeros(num_envs, 6, device=device)
  released = torch.zeros(num_envs, dtype=torch.bool, device=device)
  corrupted = torch.zeros(num_envs, dtype=torch.bool, device=device)
  wrench_release = torch.zeros(num_envs, 6, device=device)
  reason_release = torch.zeros(num_envs, dtype=torch.long, device=device)
  hist = torch.zeros(3, num_envs, 6, device=device)
  tail_sum = torch.zeros(num_envs, 6, device=device)
  tail_n = 0
  # Integral servo state: commanded-minus-measured wrench correction (engage
  # frame). Converges the realized anchor wrench onto the ray against every
  # parasitic path at once; the plant is ~unit-gain and quasistatic, so a
  # plain integrator with gain < 1 is stable, and its residual lag under the
  # ramp is rate_per_step / gain (~1 % of the target here).
  servo_gain = 0.6
  bias6 = torch.zeros(num_envs, 6, device=device)
  # Servo warmup: judge only after the parasitic-wrench servo has converged
  # (every ray is still at s << 1 when the envelope arms).
  warm_steps = 50

  steps = int(args.seconds / env.step_dt)
  for step in range(steps):
    robot.write_root_link_velocity_to_sim(zero_root)
    robot.write_joint_velocity_to_sim(zero_qd)
    if step >= warm_steps:
      pin_params()
    else:
      affordance.limit_f_t_max[:, target] = big
      affordance.limit_tilt_peel_max[:] = big
    s = torch.minimum(rate * (step + 1) * env.step_dt, target_s)
    # Preloads ramp in during the first 40 steps (inside the unbreakable
    # warmup) and then hold; the ray part scales with s on top.
    warm_frac = min((step + 1) / 40.0, 1.0)
    des6 = warm_frac * pre6 + s.unsqueeze(-1) * w6
    cmd6 = des6 + bias6
    live = (~released).float().unsqueeze(-1)
    force = live * (
      cmd6[:, 0:1] * t_hat + cmd6[:, 1:2] * l_hat + cmd6[:, 2:3] * n_hat
    )
    couple = live * (
      cmd6[:, 3:4] * t_hat + cmd6[:, 4:5] * l_hat + cmd6[:, 5:6] * n_hat
    )
    # xfrc force acts at the body CoM; correct the couple so the commanded
    # wrench is exact about the pad-center anchor the billing judges.
    arm = (
      executor.pad_center_world()[:, target]
      - robot.data.body_com_pos_w[:, ee_body_id]
    )
    torque = couple + torch.cross(arm, force, dim=-1)
    robot.write_external_wrench_to_sim(
      forces=force.unsqueeze(1),
      torques=torque.unsqueeze(1),
      body_ids=spine_ids,
    )
    _, _, terminated, truncated, _ = env.step(actions)
    corrupted |= (terminated | truncated) & ~released
    hist = torch.roll(hist, 1, dims=0)
    hist[0] = torch.cat(
      [
        affordance.anchor_load_tnl[:, target],
        affordance.anchor_moment_tnl[:, target],
      ],
      dim=-1,
    )
    upd = (~released & ~corrupted).unsqueeze(-1)
    bias6 = torch.where(
      upd, bias6 + servo_gain * (des6 - hist[0]), bias6
    )
    newly = (
      affordance.forced_release_this_step[:, target] & ~released & ~corrupted
    )
    wrench_release = torch.where(
      newly.unsqueeze(-1), hist.median(dim=0).values, wrench_release
    )
    reason_release = torch.where(
      newly, affordance.release_reason[:, target], reason_release
    )
    released |= newly
    if step >= steps - 25:
      tail_sum += hist[0]
      tail_n += 1

  valid = ~corrupted
  tags = torch.as_tensor(tags_np, device=device)
  tags_all = torch.cat([tags, tags[torch.as_tensor(interior_idx, device=device)]])

  b_hit = is_boundary & released & valid
  bp = wrench_release[b_hit]
  occ_rel = surf.occupancy6(bp.cpu())
  print(
    f"  boundary: released {int(b_hit.sum())}/{int((is_boundary & valid).sum())}, "
    f"|occ - 1| p50 {float((occ_rel - 1).abs().quantile(0.5)):.3f}, "
    f"p90 {float((occ_rel - 1).abs().quantile(0.9)):.3f}"
  )
  for pid, name in (
    (PANEL_A, "a"), (PANEL_B, "b"), (PANEL_C, "c"), (PANEL_F, "force-3d"),
    (PANEL_D70, "force@70"), (PANEL_G70, "twist-pitch@70"),
  ):
    sel = tags_all[b_hit][:, 0] == pid
    if sel.any():
      o = occ_rel[sel.cpu()]
      print(
        f"    panel {name}: {int(sel.sum())} pts, |occ - 1| p50 "
        f"{float((o - 1).abs().quantile(0.5)):.3f}, max "
        f"{float((o - 1).abs().max()):.3f}"
      )
  reasons = reason_release[b_hit]
  counts = {int(k): int((reasons == k).sum()) for k in reasons.unique()}
  print(f"    release reasons (1 rev, 2 tension, 3 lateral, 4 ult): {counts}")

  i_held = ~is_boundary & ~released & valid
  ip = tail_sum[i_held] / max(tail_n, 1)
  occ_held = surf.occupancy6(ip.cpu())
  n_int = int((~is_boundary & valid).sum())
  print(
    f"  interior: held {int(i_held.sum())}/{n_int} "
    f"(released {int((~is_boundary & released & valid).sum())}), "
    f"occ p90 {float(occ_held.quantile(0.9)):.3f} (all should be < 1)"
  )
  print(f"  excluded (mid-test reset): {int(corrupted.sum())} envs")

  bp_np = bp.cpu().numpy()
  ip_np = ip.cpu().numpy()
  b_tags = tags_all[b_hit].cpu().numpy()
  i_tags = tags_all[i_held].cpu().numpy()
  np.savez(
    args.out_stem + ".npz",
    bp=bp_np, ip=ip_np, b_tags=b_tags, i_tags=i_tags, f0=f0,
    family=np.asarray(th["family"]), bend_levels=np.asarray(BEND_LEVELS),
  )
  render_figure(th, f0, bp_np, ip_np, b_tags, i_tags, args.out_stem)
  env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
