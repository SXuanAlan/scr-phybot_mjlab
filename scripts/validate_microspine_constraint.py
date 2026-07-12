"""Validation suite for the microspine connect-equality attachment.

Subcommands:
  hang     16-env hang test on a vertical (90°) wall: all 4 EEs anchored,
           zero actions. Asserts the anchors hold the robot weight without
           sag, jitter, NaNs, or efc overflow.
  sweep    GPU-parallel limit-surface envelope scan: each env pulls one EE in
           a different direction of the t-n / t-l planes with a per-direction
           force ramp, recording the anchor load at forced release. Checks
           the measured release points lie on the configured envelope and
           optionally plots measured-vs-theory (--plot out.png).
  readout  Cross-checks the efc-row force gather against the kinematic
           anchor error (constraint force ∝ pose error) under a time-varying
           torso push.

Run from the repo root inside the env_mjlab environment, e.g.:
  python scripts/validate_microspine_constraint.py hang --num-envs 16
"""

from __future__ import annotations

import argparse
import math
import sys

import torch
import warp as wp

from mjlab.envs import ManagerBasedRlEnv
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)

DEVICE = "cuda:0"
EE_NAMES = (
  "left_hand_spine",
  "right_hand_spine",
  "left_foot_spine",
  "right_foot_spine",
)
# Deterministic target-EE envelope used by sweep (overrides the sampled DR).
SWEEP_PARAMS = {
  "f_t_max": 400.0,
  "f_rev_max": 20.0,
  "mu_adhesion": 0.4,
  "f_n0": 15.0,
  "mu_lateral": 0.6,
  "f_l0": 15.0,
}


def build_env(
  num_envs: int,
  limit_surface_enable: bool = True,
  reanchor_after_grace: bool = True,
  anchor_lock_rotation: bool | None = None,
) -> ManagerBasedRlEnv:
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = 1.0e9  # No time_out resets during tests.
  cfg.curriculum = {}  # Gravity is set explicitly per test.
  cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
  # The nested limit_surface cfg is shared between the affordance command and
  # the physics event term (dataclasses.replace is shallow copy: nested
  # objects shared, scalar fields independent) — so scalars must be written
  # on both copies. anchor_lock_rotation also steers the scene spec_fn, which
  # closes over the commands-side cfg — written before env construction, so
  # the equality type and the executor stay consistent.
  mcfg = cfg.commands["microspine_affordance"].microspine_cfg
  mcfg.limit_surface.enable = limit_surface_enable
  for owner in (mcfg, cfg.events["microspine_grip"].params["microspine_cfg"]):
    owner.reanchor_after_grace = reanchor_after_grace
    if anchor_lock_rotation is not None:
      owner.anchor_lock_rotation = anchor_lock_rotation
  return ManagerBasedRlEnv(cfg=cfg, device=DEVICE)


def set_gravity(env: ManagerBasedRlEnv, angle_deg: float, magnitude: float = 9.81):
  """Same convention as the gravity_angle curriculum (90° = vertical wall)."""
  rad = math.radians(angle_deg)
  view = wp.to_torch(env.sim.wp_model.opt.gravity)
  view[:, 0] = magnitude * math.cos(rad)
  view[:, 1] = 0.0
  view[:, 2] = -magnitude * math.sin(rad)


def get_terms(env: ManagerBasedRlEnv):
  executor = env.event_manager.get_term_cfg("microspine_grip").func
  affordance = env.command_manager._terms["microspine_affordance"]
  return executor, affordance


def zero_actions(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=DEVICE
  )


def robot_weight(env: ManagerBasedRlEnv) -> float:
  robot = env.scene["robot"]
  body_ids = robot.indexing.body_ids
  mass = float(env.sim.mj_model.body_mass[body_ids.cpu().numpy()].sum())
  return mass * 9.81


def check(name: str, ok: bool, detail: str) -> bool:
  print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
  return ok


# ---------------------------------------------------------------------------
# hang
# ---------------------------------------------------------------------------


def cmd_hang(args) -> bool:
  print(f"== hang: {args.num_envs} envs, 90° wall, {args.seconds}s ==")
  # Limit surface ON by default: re-grip re-seats each anchor at the settled
  # pad pose and relieves spawn-pose preload before hold checks begin.
  # "all engaged" passing with hold checks active is therefore the re-grip
  # acceptance test. --no-limit-surface restores the mechanics-only variant.
  env = build_env(
    args.num_envs,
    limit_surface_enable=not args.no_limit_surface,
  )
  env.reset()
  set_gravity(env, 90.0)
  executor, affordance = get_terms(env)
  executor.assert_now(env)

  steps = int(args.seconds / env.step_dt)
  weight = robot_weight(env)
  robot = env.scene["robot"]
  z0 = robot.data.root_link_pos_w[:, 2].clone()

  actions = zero_actions(env)
  support_hist, engaged_hist = [], []
  terminations = 0
  root_z_hist = []
  for _ in range(steps):
    _, _, terminated, truncated, _ = env.step(actions)
    terminations += int((terminated | truncated).sum().item())
    root_z_hist.append(robot.data.root_link_pos_w[:, 2].clone())
    # F_load is EE-on-anchor; the wall's supporting force on the robot is -z
    # of that, so total support = -sum(F_load_z). Gather explicitly so the
    # readout also works with --no-limit-surface.
    support_hist.append(
      -executor.compute_anchor_loads()[..., 2].sum(dim=1).clone()
    )
    engaged_hist.append(affordance.engaged.all(dim=1).clone())

  root_z = torch.stack(root_z_hist)  # (T, E)
  support = torch.stack(support_hist)  # (T, E)
  engaged = torch.stack(engaged_hist)  # (T, E)
  half = steps // 2

  sag = (z0.unsqueeze(0) - root_z).max().item()
  support_mean = support[half:].mean().item()
  support_std = support[half:].std(dim=0).mean().item()
  jitter = root_z[half:].std(dim=0).max().item()
  nefc_max = int(env.sim.data.nefc[:].max().item())
  njmax = int(env.sim.data.njmax)

  ok = True
  ok &= check("all engaged", bool(engaged.all().item()), f"{engaged.float().mean():.3f} mean")
  ok &= check("no terminations", terminations == 0, f"{terminations} resets")
  ok &= check("root sag < 2 cm", sag < 0.02, f"max sag {sag * 1000:.1f} mm")
  ok &= check(
    "anchors carry weight",
    abs(support_mean - weight) / weight < 0.10,
    f"support {support_mean:.1f} N vs weight {weight:.1f} N "
    f"({100 * (support_mean / weight - 1):+.1f}%)",
  )
  ok &= check("root jitter < 1 mm", jitter < 1.0e-3, f"{jitter * 1000:.3f} mm std")
  ok &= check("nefc headroom", nefc_max + 12 <= njmax, f"nefc {nefc_max} / njmax {njmax}")
  ok &= check("finite state", bool(torch.isfinite(root_z).all().item()), "root z finite")
  print(f"  anchor support force std (per-env mean): {support_std:.3f} N")
  # Informational: final per-EE loads in the engage frame vs the default
  # envelope. Tension beyond the cone here means the spawn pose itself is not
  # holdable by a realistic spine envelope (task/pose concern, not physics).
  t_hat, l_hat, n_hat = affordance._engage_frame()
  loads = executor.compute_anchor_loads()
  f_t = (loads * t_hat).sum(-1).mean(0)
  f_l = (loads * l_hat).sum(-1).mean(0)
  f_n = (loads * n_hat).sum(-1).mean(0)
  for h, name in enumerate(EE_NAMES):
    print(
      f"  {name}: f_t={f_t[h]:.1f} N, f_l={f_l[h]:.1f} N, f_n={f_n[h]:.1f} N "
      f"(tension cap @ default mid params ≈ "
      f"{0.4 * max(f_t[h].item(), 0.0) + 15.0:.1f} N)"
    )
  env.close()

  return ok


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def _predicted_release_radius(theta: torch.Tensor, plane: str) -> torch.Tensor:
  """Smallest envelope-violating pull magnitude along ray angle theta.

  Ray direction is cos(theta)·t̂ + sin(theta)·(n̂ or l̂). Returns +inf where no
  finite pull violates (compression-dominated rays the bilateral anchor and
  the contacts hold forever).
  """
  p = SWEEP_PARAMS
  cos_t, sin_t = torch.cos(theta), torch.sin(theta)
  inf = torch.full_like(theta, float("inf"))
  # c1: reverse shear, f_t < -f_rev_max.
  r1 = torch.where(cos_t < 0, p["f_rev_max"] / (-cos_t).clamp_min(1e-9), inf)
  # c4: ultimate shear, f_t > f_t_max.
  r4 = torch.where(cos_t > 0, p["f_t_max"] / cos_t.clamp_min(1e-9), inf)
  if plane == "tn":
    denom = sin_t - p["mu_adhesion"] * cos_t.clamp_min(0.0)
    r2 = torch.where(denom > 0, p["f_n0"] / denom.clamp_min(1e-9), inf)
  else:  # tl
    denom = sin_t.abs() - p["mu_lateral"] * cos_t.clamp_min(0.0)
    r2 = torch.where(denom > 0, p["f_l0"] / denom.clamp_min(1e-9), inf)
  return torch.minimum(torch.minimum(r1, r4), r2)


def _envelope_margin(load_tnl: torch.Tensor) -> torch.Tensor:
  """Normalized min-margin of the fixed SWEEP_PARAMS envelope at given loads."""
  p = SWEEP_PARAMS
  f_t, f_l, f_n = load_tnl[..., 0], load_tnl[..., 1], load_tnl[..., 2]
  ft_pos = f_t.clamp(min=0.0)
  cap_n = p["mu_adhesion"] * ft_pos + p["f_n0"]
  cap_l = p["mu_lateral"] * ft_pos + p["f_l0"]
  m1 = (f_t + p["f_rev_max"]) / p["f_rev_max"]
  m2 = (cap_n - f_n) / cap_n
  m3 = (cap_l - f_l.abs()) / cap_l
  m4 = (p["f_t_max"] - f_t) / p["f_t_max"]
  return torch.minimum(torch.minimum(m1, m2), torch.minimum(m3, m4))


def cmd_sweep(args) -> bool:
  num_envs = args.num_envs
  target = args.ee_index
  print(
    f"== sweep: {num_envs} envs, target EE {EE_NAMES[target]}, "
    f"ramp {args.seconds}s ==",
  )
  # The per-ray release-force prediction assumes a force-only anchor. A weld
  # adds a torque path (the pull's moment about the pad rides the rotational
  # rows and re-shapes contact compression), which legitimately carries ~5%
  # of rays past the ramp end — their releases still land on the envelope
  # (measured p50 0.012 in weld mode), but the all-rays-release criterion
  # breaks. The sweep validates the limit-surface geometry, which is anchor-
  # type independent, so pin the original pure-force connect anchor.
  env = build_env(num_envs, anchor_lock_rotation=args.lock_rotation)
  env.reset()
  set_gravity(env, 90.0, magnitude=0.0)  # Zero gravity: pure scripted pulls.
  executor, affordance = get_terms(env)
  executor.assert_now(env)
  robot = env.scene["robot"]
  actions = zero_actions(env)

  # All EEs unbreakable while the static preload is nulled out. Intercept-only
  # caps — a huge mu would produce inf*0 NaNs at zero shear.
  big = 1.0e9
  affordance.limit_f_t_max[:] = big
  affordance.limit_f_rev_max[:] = big
  affordance.limit_mu_adhesion[:] = 0.0
  affordance.limit_f_n0[:] = big
  affordance.limit_mu_lateral[:] = 0.0
  affordance.limit_f_l0[:] = big

  # The spawn pose internally preloads the anchors (PD fights the anchored
  # closed chain — tens of N even at zero g), which would trip the small test
  # envelope before any scripted pull. Null it: alternately let the robot
  # settle and re-place the anchors at the settled pad positions. The
  # executor rewrites its cached anchor_pos_w into eq_data every step, so
  # overriding the cache is sufficient.
  for _ in range(8):
    for _ in range(10):
      env.step(actions)
    executor.anchor_pos_w = executor.pad_center_world().clone()
  for _ in range(20):
    env.step(actions)
  baseline = executor.compute_anchor_loads()[:, target]
  print(
    f"  residual static preload on target EE: "
    f"max |F| = {baseline.norm(dim=-1).max().item():.2f} N"
  )

  # Deterministic envelope on the target EE only.
  for key, value in SWEEP_PARAMS.items():
    getattr(affordance, f"limit_{key}")[:, target] = value
  others = [h for h in range(len(EE_NAMES)) if h != target]

  # Engage frame of the target EE (constant: wall static, zero gravity makes
  # the rake fall back to world -z projected onto the wall).
  t_hat, l_hat, n_hat = affordance._engage_frame()
  t_hat = t_hat[:, target]
  l_hat = l_hat[:, target]
  n_hat = n_hat[:, target]

  # First half scans the t-n plane, second half t-l.
  half = num_envs // 2
  plane_tn = torch.arange(num_envs, device=DEVICE) < half
  theta = torch.empty(num_envs, device=DEVICE)
  theta[:half] = torch.linspace(-math.pi, math.pi, half + 1, device=DEVICE)[:-1]
  theta[half:] = torch.linspace(
    -math.pi, math.pi, num_envs - half + 1, device=DEVICE
  )[:-1]
  second_axis = torch.where(plane_tn.unsqueeze(-1), n_hat, l_hat)
  direction = (
    torch.cos(theta).unsqueeze(-1) * t_hat
    + torch.sin(theta).unsqueeze(-1) * second_axis
  )

  r_pred = torch.where(
    plane_tn,
    _predicted_release_radius(theta, "tn"),
    _predicted_release_radius(theta, "tl"),
  )
  # Rays grazing the envelope faces (denominators → 0 near pure compression)
  # predict astronomical release forces; cap those ramps instead. A capped
  # (compression-dominated) ray may legitimately still release: pressing the
  # pad hard through a bent limb builds secondary shear/lateral load that
  # genuinely crosses the envelope — the assertion below therefore only
  # requires that *whatever releases does so on the envelope*.
  finite = r_pred < 1.5 * SWEEP_PARAMS["f_t_max"]
  # Per-direction ramp rate: release ~mid-ramp for finite predictions; cap
  # no-release rays at a force that proves they hold.
  steps = int(args.seconds / env.step_dt)
  rate = torch.where(
    finite, 2.2 * r_pred / args.seconds, 1.5 * SWEEP_PARAMS["f_t_max"] / args.seconds
  )

  spine_ids, _ = robot.find_bodies([EE_NAMES[target]])
  released = torch.zeros(num_envs, dtype=torch.bool, device=DEVICE)
  load_at_release = torch.zeros(num_envs, 3, device=DEVICE)
  zero_torque = torch.zeros(num_envs, 1, 3, device=DEVICE)

  corrupted = torch.zeros(num_envs, dtype=torch.bool, device=DEVICE)
  for step in range(steps):
    # A mid-test env reset (e.g. violent post-release limb motion tripping
    # bad_state) resamples the DR params; pin the test envelope every step
    # and exclude envs that reset before their release from the stats.
    for key, value in SWEEP_PARAMS.items():
      getattr(affordance, f"limit_{key}")[:, target] = value
    magnitude = rate * (step + 1) * env.step_dt
    force = torch.where(
      released.unsqueeze(-1), torch.zeros_like(direction), magnitude.unsqueeze(-1) * direction
    )
    robot.write_external_wrench_to_sim(
      forces=force.unsqueeze(1), torques=zero_torque, body_ids=spine_ids
    )
    _, _, terminated, truncated, _ = env.step(actions)
    corrupted |= (terminated | truncated) & ~released
    newly = affordance.forced_release_this_step[:, target] & ~released & ~corrupted
    load_at_release = torch.where(
      newly.unsqueeze(-1), affordance.anchor_load_tnl[:, target], load_at_release
    )
    released |= newly

  # --- evaluation ---
  ok = True
  valid = ~corrupted
  print(f"  excluded (reset before release): {int(corrupted.sum())} envs")
  others_held = bool(affordance.engaged[valid][:, others].all().item())
  ok &= check("non-target EEs held", others_held, "no spurious releases")

  finite_release_frac = released[finite & valid].float().mean().item()
  if args.lock_rotation:
    # Weld adds a torque path that legitimately carries a few % of rays past
    # the ramp end (see cmd_sweep docstring). The envelope-geometry check below
    # is anchor-type independent and is the real assertion, so report the
    # release fraction here without failing on it.
    print(
      f"  [INFO] finite-prediction rays released (weld): "
      f"{100 * finite_release_frac:.1f}%"
    )
  else:
    ok &= check(
      "finite-prediction rays released",
      finite_release_frac > 0.98,
      f"{100 * finite_release_frac:.1f}%",
    )
  capped = ~finite & valid
  print(
    f"  capped (compression-dominated) rays: {int(capped.sum())}, "
    f"of which released on secondary load: {int(released[capped].sum())}"
  )

  hit = released & valid
  margin = _envelope_margin(load_at_release[hit])
  abs_margin = margin.abs()
  p50 = abs_margin.quantile(0.5).item()
  p90 = abs_margin.quantile(0.9).item()
  ok &= check(
    "release on envelope (|margin| p50 < 0.10)", p50 < 0.10, f"p50 {p50:.3f}"
  )
  ok &= check(
    "release on envelope (|margin| p90 < 0.20)", p90 < 0.20, f"p90 {p90:.3f}"
  )

  if args.plot:
    try:
      import matplotlib

      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
      import numpy as np

      p = SWEEP_PARAMS
      fig, axes = plt.subplots(1, 2, figsize=(13, 6))
      for ax, plane, mask, ylab, col in (
        (axes[0], "tn", hit & plane_tn, "f_n [N] (tension +)", 2),
        (axes[1], "tl", hit & ~plane_tn, "f_l [N]", 1),
      ):
        pts = load_at_release[mask].cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, col], s=8, label="measured release")
        ft = np.linspace(-p["f_rev_max"] * 1.2, p["f_t_max"] * 1.05, 200)
        if plane == "tn":
          cap = p["mu_adhesion"] * np.clip(ft, 0, None) + p["f_n0"]
          ax.plot(ft, cap, "r-", label="tension cone")
        else:
          cap = p["mu_lateral"] * np.clip(ft, 0, None) + p["f_l0"]
          ax.plot(ft, cap, "r-", label="lateral cone")
          ax.plot(ft, -cap, "r-")
        ax.axvline(-p["f_rev_max"], color="orange", label="reverse limit")
        ax.axvline(p["f_t_max"], color="purple", label="ultimate shear")
        ax.set_xlabel("f_t [N] (engage shear +)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{plane} plane")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
      fig.suptitle(f"Microspine limit-surface sweep — {EE_NAMES[target]}")
      fig.tight_layout()
      fig.savefig(args.plot, dpi=130)
      print(f"  plot saved to {args.plot}")
    except ImportError:
      print("  matplotlib not available; skipping plot")

  env.close()
  return ok


# ---------------------------------------------------------------------------
# readout
# ---------------------------------------------------------------------------


def cmd_readout(args) -> bool:
  print(f"== readout: {args.num_envs} envs, efc gather vs kinematic error ==")
  # Re-grip off: this test regresses load against anchor error, and grace-
  # window reseating decouples the two by design (the anchor follows the pad).
  env = build_env(args.num_envs, reanchor_after_grace=False)
  env.reset()
  set_gravity(env, 90.0)
  executor, _ = get_terms(env)
  executor.assert_now(env)
  robot = env.scene["robot"]
  torso_ids, _ = robot.find_bodies(["torso"])

  steps = int(args.seconds / env.step_dt)
  settle = min(25, steps // 4)
  actions = zero_actions(env)
  zero_torque = torch.zeros(env.num_envs, 1, 3, device=DEVICE)
  loads_hist, err_hist = [], []
  amp = 60.0
  for step in range(steps):
    t = step * env.step_dt
    push = torch.tensor(
      [
        amp * math.sin(2 * math.pi * 0.30 * t),
        amp * math.sin(2 * math.pi * 0.23 * t + 1.0),
        amp * math.sin(2 * math.pi * 0.17 * t + 2.0),
      ],
      device=DEVICE,
    ).expand(env.num_envs, 1, 3)
    robot.write_external_wrench_to_sim(
      forces=push, torques=zero_torque, body_ids=torso_ids
    )
    env.step(actions)
    if step >= settle:
      loads_hist.append(executor.anchor_load_w.clone())
      err_hist.append(executor.anchor_error_w().clone())

  loads = torch.stack(loads_hist)  # (T, E, H, 3)
  errs = torch.stack(err_hist)

  ok = True
  for h, name in enumerate(EE_NAMES):
    load = loads[:, :, h].reshape(-1, 3)
    err = errs[:, :, h].reshape(-1, 3)
    # The anchor load (EE-on-anchor) tracks the pose error through the
    # constraint impedance: F_load ≈ K_eff · err with K_eff a 3x3 stiffness
    # (direction-dependent — the soft-constraint force scales with the
    # per-axis effective inverse mass, so a scalar spring underfits). A
    # gather bug (wrong rows, wrong sign) would destroy this fit entirely.
    k_mat = torch.linalg.lstsq(err, load).solution  # (3, 3): load ≈ err @ K
    pred = err @ k_mat
    ss_res = (load - pred).pow(2).sum()
    ss_tot = (load - load.mean(dim=0)).pow(2).sum().clamp_min(1e-12)
    r2 = (1 - ss_res / ss_tot).item()
    big = load.norm(dim=-1) > 10.0
    cosine = torch.nn.functional.cosine_similarity(
      load[big], pred[big], dim=-1
    ).mean().item() if big.any() else float("nan")
    k_diag = k_mat.diagonal()
    print(
      f"  {name}: K_diag=({k_diag[0]:.0f}, {k_diag[1]:.0f}, {k_diag[2]:.0f}) "
      f"N/m, R²={r2:.3f}, dir-cos={cosine:.3f} (|F|>10N, n={int(big.sum())})"
    )
    # Thresholds are deliberately loose: the linear map cannot fully explain
    # the load (solimp impedance roughly doubles the effective stiffness
    # across the mm displacement range, hand contacts add stick-slip
    # hysteresis, and the four anchors couple through the body). What this
    # test discriminates is gather bugs — a sign flip lands at cos ≈ -0.8 and
    # wrong/stale rows at cos ≈ 0. Absolute magnitude correctness is covered
    # by the hang test's weight-closure assertion.
    ok &= check(f"{name} stiffness-map fit", r2 > 0.25, f"R² {r2:.3f}")
    ok &= check(f"{name} direction agreement", cosine > 0.6, f"cos {cosine:.3f}")
    ok &= check(
      f"{name} load tracks error (+K)",
      bool((k_diag > 0).all().item()),
      "diagonal stiffness positive",
    )
  env.close()
  return ok


# ---------------------------------------------------------------------------
# dynamic  (debounce / grace tolerance under fast, non-quasistatic loads)
# ---------------------------------------------------------------------------


def _dynamic_settle(env, executor, affordance, target, actions):
  """Reset, zero gravity, pin the test envelope, null the spawn preload, and
  leave the target EE engaged and well past the grace window."""
  env.reset()
  set_gravity(env, 90.0, magnitude=0.0)
  executor.assert_now(env)
  for key, value in SWEEP_PARAMS.items():
    getattr(affordance, f"limit_{key}")[:, target] = value
  # Same preload-nulling trick as cmd_sweep: settle, then re-seat the anchors
  # at the settled pad so the closed-chain PD preload does not pre-trip the
  # small test envelope.
  for _ in range(8):
    for _ in range(10):
      env.step(actions)
    executor.anchor_pos_w = executor.pad_center_world().clone()
  for _ in range(20):
    env.step(actions)


def cmd_dynamic(args) -> bool:
  """Probe the release debounce/grace under fast (non-quasistatic) loads.

  cmd_sweep validates the envelope quasi-statically (release lands on the
  boundary). This probes the *dynamic* behavior the debounce introduces:
    A. a sustained step overload must still release (after the debounce count);
    B. a load that flips outside/inside every step can evade release entirely
       — the F5 exploit surface. Both live in the affordance debounce, so they
       are anchor-type independent; connect is the clean default.
  """
  target = args.ee_index
  env = build_env(args.num_envs, anchor_lock_rotation=args.lock_rotation)
  executor, affordance = get_terms(env)
  robot = env.scene["robot"]
  actions = zero_actions(env)
  spine_ids, _ = robot.find_bodies([EE_NAMES[target]])
  zero_torque = torch.zeros(args.num_envs, 1, 3, device=DEVICE)

  if args.violation_steps is not None:
    affordance._ls_cfg.violation_consecutive_steps = args.violation_steps
  vsteps = max(1, affordance._ls_cfg.violation_consecutive_steps)
  mode = "weld" if args.lock_rotation else "connect"
  print(
    f"== dynamic: {args.num_envs} envs, target {EE_NAMES[target]}, {mode}, "
    f"violation_consecutive_steps={vsteps} =="
  )

  def pull(mag_vec):
    force = torch.zeros(args.num_envs, 1, 3, device=DEVICE)
    force[:, 0, :] = mag_vec
    robot.write_external_wrench_to_sim(
      forces=force, torques=zero_torque, body_ids=spine_ids
    )
    env.step(actions)

  # Pure pull-off (t-n plane, 90°): crosses the tension face at f_n0, a
  # debounced (soft) condition — exactly what the oscillation probes.
  _dynamic_settle(env, executor, affordance, target, actions)
  _, _, n_hat = affordance._engage_frame()
  n_hat = n_hat[:, target]
  r_cross = SWEEP_PARAMS["f_n0"]
  hi, lo = 1.8 * r_cross, 0.4 * r_cross

  # --- A: sustained step overload (constant, clearly outside) ---
  base_engaged = affordance.engaged[:, target].clone()
  steps_to_rel = torch.full((args.num_envs,), -1, dtype=torch.long, device=DEVICE)
  for s in range(12):
    pull(hi * n_hat)
    newly = affordance.forced_release_this_step[:, target] & (steps_to_rel < 0)
    steps_to_rel = torch.where(
      newly, torch.full_like(steps_to_rel, s + 1), steps_to_rel
    )
  rel_mask = (steps_to_rel >= 0) & base_engaged
  frac_rel_A = rel_mask.float().mean().item()
  lat = steps_to_rel[rel_mask]
  med_lat = int(lat.median().item()) if lat.numel() else -1

  # --- B: oscillation evasion (1 step outside, 1 step inside) ---
  _dynamic_settle(env, executor, affordance, target, actions)
  b_steps = 40
  ever_released = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
  outside = torch.zeros(args.num_envs, device=DEVICE)
  for s in range(b_steps):
    pull((hi if s % 2 == 0 else lo) * n_hat)
    ever_released |= affordance.forced_release_this_step[:, target]
    outside += (affordance.hold_margin[:, target] < 0).float()
  frac_evaded = (~ever_released).float().mean().item()
  frac_outside = (outside / b_steps).mean().item()

  # --- C: isolated overload spike, average load INSIDE the envelope ---
  # The tightest debounce-evasion probe: one outside spike every spike_period
  # steps, deep-inside otherwise, so the *mean* load sits inside the cone —
  # the pattern a policy would need to "ride" the debounce and hold an
  # effectively-outside load without ever tripping the consecutive count.
  _dynamic_settle(env, executor, affordance, target, actions)
  spike_period, c_steps, deep_lo = 4, 48, 0.1 * r_cross
  avg_load = (hi + (spike_period - 1) * deep_lo) / spike_period
  ever_released_c = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
  outside_c = torch.zeros(args.num_envs, device=DEVICE)
  for s in range(c_steps):
    pull((hi if s % spike_period == 0 else deep_lo) * n_hat)
    ever_released_c |= affordance.forced_release_this_step[:, target]
    outside_c += (affordance.hold_margin[:, target] < 0).float()
  frac_evaded_c = (~ever_released_c).float().mean().item()
  frac_outside_c = (outside_c / c_steps).mean().item()
  env.close()

  print(
    f"  [A] sustained 1.8x pull-off: released {100 * frac_rel_A:.1f}%, "
    f"median steps-to-release = {med_lat}"
  )
  print(
    f"  [B] oscillation 1-out/1-in: evaded release {100 * frac_evaded:.1f}%, "
    f"mean time spent outside envelope {100 * frac_outside:.0f}%"
  )
  print(
    f"  [C] isolated spike (avg load {avg_load / r_cross:.2f}x cap, INSIDE): "
    f"evaded release {100 * frac_evaded_c:.1f}%, "
    f"outside {100 * frac_outside_c:.0f}% of steps"
  )
  ok = True
  ok &= check(
    "sustained overload releases", frac_rel_A > 0.95, f"{100 * frac_rel_A:.1f}%"
  )
  ok &= check(
    "release latency within debounce+lag",
    1 <= med_lat <= vsteps + 2,
    f"{med_lat} steps (debounce={vsteps})",
  )
  print(
    f"  -> F5: at vsteps={vsteps}, an oscillating load rode "
    f"{100 * frac_outside:.0f}% of steps outside the envelope while "
    f"{100 * frac_evaded:.0f}% of envs never released."
  )
  return ok


# ---------------------------------------------------------------------------


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="cmd", required=True)

  p_hang = sub.add_parser("hang")
  p_hang.add_argument("--num-envs", type=int, default=16)
  p_hang.add_argument("--seconds", type=float, default=10.0)
  p_hang.add_argument(
    "--no-limit-surface",
    action="store_true",
    help="Validate the raw anchor mechanics with hold checks disabled "
    "(pre-re-grip behavior; engaged spines stay latched forever).",
  )

  p_sweep = sub.add_parser("sweep")
  p_sweep.add_argument("--num-envs", type=int, default=1024)
  p_sweep.add_argument("--seconds", type=float, default=6.0)
  p_sweep.add_argument("--ee-index", type=int, default=2, choices=range(4))
  p_sweep.add_argument("--plot", type=str, default="")
  p_sweep.add_argument(
    "--lock-rotation",
    action="store_true",
    help="Weld anchors (production path). Relaxes the all-rays-release "
    "criterion; still asserts releases land on the envelope.",
  )

  p_read = sub.add_parser("readout")
  p_read.add_argument("--num-envs", type=int, default=8)
  p_read.add_argument("--seconds", type=float, default=10.0)

  p_dyn = sub.add_parser("dynamic")
  p_dyn.add_argument("--num-envs", type=int, default=512)
  p_dyn.add_argument("--ee-index", type=int, default=2, choices=range(4))
  p_dyn.add_argument("--lock-rotation", action="store_true")
  p_dyn.add_argument(
    "--violation-steps",
    type=int,
    default=None,
    help="Override MicrospineLimitSurfaceCfg.violation_consecutive_steps.",
  )

  args = parser.parse_args()
  ok = {
    "hang": cmd_hang,
    "sweep": cmd_sweep,
    "readout": cmd_readout,
    "dynamic": cmd_dynamic,
  }[args.cmd](args)
  print("OVERALL:", "PASS" if ok else "FAIL")
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
