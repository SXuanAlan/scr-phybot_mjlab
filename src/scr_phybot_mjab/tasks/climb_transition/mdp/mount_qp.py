"""Static contact-force QP for the scripted mount base (B-line teacher).

Extends the climb scripted base's contact QP (climb-osc-wbc-base branch,
``tasks/climb_wall/mdp/contact_qp.py``) with GROUND contacts: the mount
prologue stands on two feet, so the support set mixes microspine anchors
(Wang-2017 envelope cones) with unilateral floor contacts (Coulomb friction
pyramids at the heel/toe sole points — the validator's ground model).

Per control step, for one env, solve the quasi-static load distribution over
the current support set together with the CoM position that realizes it:

  variables   λ_k ∈ R³ per contact unit, expressed in the unit's own frame:
                wang   — (t, l, n_out) wall components, anchor_load_tnl
                         convention (f_t > 0 loads the spines along the rake,
                         f_n > 0 is pull-off tension);
                ground — world (x, y, z_up) components of the force the
                         FLOOR applies to the robot (f_z > 0 pushes up).
              the CoM target c ∈ R³ and a bounded moment slack s.
  equalities  Σ F_k(λ_k) = m g⃗ + F_ext          (world force balance)
              Σ r_k × F_k = c × (m g⃗) + M_ext + s
              NOTE the sign convention difference: wang λ is the load the
              ROBOT puts on the wall (reaction ON the robot is −R λ), while
              ground λ is the force ON the robot directly. F_k resolves both
              to "force on the robot" before the balance.
  cones       wang:   f_t ≥ f_t_min; f_n ≤ f_t/4; |f_l| ≤ 0.3 f_t;
                      ‖(f_t, f_l, f_n₊)‖ ≤ γ·F0·seat
              ground: f_z ≥ 0; |f_x| ≤ μ̂ f_z; |f_y| ≤ μ̂ f_z
                      (μ̂ = μ/√2, inner pyramid of the Coulomb cone)
              plus c_x ≤ com_x_max and optional per-unit ‖λ‖ share caps
              (the pre-unload input — works for lifting a foot off the
              floor exactly like unloading a pad before release).
  objective   wang margin balancing + a small ground-force regularizer +
              strong CoM regularization toward the measurement + a pin on
              the ĝ-parallel CoM component + cheap slack.

All the hard-won numerics of the parent survive verbatim: normalized
decision variables (SLSQP line search dies on mixed O(100 N)/O(10⁻³)
scales), exact analytic jacobians (numerical differencing false-converges
on the flat force-redistribution direction), c re-seeded at the measurement
every solve, and a fresh-seed retry + penalty fallback on infeasibility so
the caller always gets a usable direction plus ``feasible=False`` to freeze
the phase clock on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:  # scipy is an optional dependency of the scripted base only.
  from scipy.optimize import minimize
except ImportError:  # pragma: no cover - exercised only on broken installs
  minimize = None


@dataclass
class MountQpCfg:
  """Cone/objective constants (wang side bench-calibrated 2026-07-25)."""

  f_t_min: float = 1.0
  """Minimum rake load per latched pad (N). The climb parent used 5 N, but
  the mount holds anchors while STANDING — the feet carry the weight and a
  5 N floor per anchor is an over-constraint the CoM trim then has to
  manufacture. 1 N keeps the load direction honest without demanding
  artificial loading."""

  tension_ratio: float = 0.25
  """tension ≤ tension_ratio · f_t — the 4:1 shear:tension sweet line."""

  lateral_ratio: float = 0.3
  """|f_l| ≤ lateral_ratio · f_t."""

  gamma_target: float = 0.5
  """Target utilization ceiling: ‖load‖ ≤ gamma_target · f0_min · seat."""

  f0_min: float = 400.0
  """Conservative per-pad ultimate capacity (N)."""

  ground_friction: float = 0.6
  """Coulomb μ at the sole points. 0.6 = the reference validator's
  conservative figure (training DR floor is 0.7 effective)."""

  ground_reg: float = 0.05
  """Objective weight on ground forces in weight units — small: standing is
  free, the term only discourages gratuitous internal shear between the
  feet. Must stay well below 1.0 or the QP starts 'saving' floor force by
  hanging weight on the anchors' utilization budget instead."""

  com_reg: float = 50.0
  """Objective weight (per m²) pulling c toward the measured CoM. Large on
  purpose: the QP only moves the body when the load geometry demands it."""

  com_gravity_pin: float = 1.0e2
  """Weight pinning the ĝ-parallel component of c (unobservable in the
  moment equation) to the measured CoM."""

  com_x_max: float = 1.45
  """Kinematic ceiling on the CoM's wall-normal coordinate (m). The mount
  wall surface sits at x = 1.5; the standing CoM measures ~1.27."""

  pad_torque_cap: float = 4.0
  """Per-contact local torque budget (N·m) granted to the moment slack.
  Ground contacts get it too: the heel/toe pair already carries the pitch
  couple, the slack absorbs only the residual (roll/yaw of the sole)."""

  slack_weight: float = 0.05
  penalty_weight: float = 1.0e3
  softplus_eps: float = 1.0
  """Smoothing (N²) of the tension hinge inside the wang capacity norm."""


@dataclass
class ContactUnit:
  """One support-set member handed to the QP."""

  name: str
  kind: str
  """"wang" (microspine anchor) or "ground" (unilateral floor point)."""

  pos_w: np.ndarray
  """(3,) world contact point."""

  frame_w: np.ndarray
  """(3, 3) world columns: wang = [t̂, l̂, n̂_out]; ground = [x̂, ŷ, ẑ_up]."""

  cap_scale: float = 1.0
  """wang: per-latch capacity scale (spine fraction at engage)."""

  share_cap: float = float("nan")
  """Optional cap on ‖λ‖ as a fraction of total weight (pre-unload /
  ramp-in input). NaN = unconstrained."""


@dataclass
class MountQpSolution:
  lambda_local: np.ndarray
  """(K, 3) per-unit loads in each unit's own frame (see module doc for
  the wang/ground sign conventions)."""

  force_on_robot_w: np.ndarray
  """(K, 3) world force each contact applies TO the robot."""

  com_target: np.ndarray
  utilization: np.ndarray
  """(K,) wang: ‖(f_t, f_l, f_n₊)‖ / (f0·seat); ground: ‖λ‖ / ‖W‖."""

  margin: float
  """1 − max wang utilization (1.0 with no anchors)."""

  feasible: bool
  residual: float
  unit_names: tuple[str, ...] = ()
  info: dict = field(default_factory=dict)


def _tension_plus(f_n: np.ndarray, eps: float) -> np.ndarray:
  return 0.5 * (f_n + np.sqrt(f_n * f_n + eps))


def _tension_plus_grad(f_n: np.ndarray, eps: float) -> np.ndarray:
  return 0.5 * (1.0 + f_n / np.sqrt(f_n * f_n + eps))


def _skew(v: np.ndarray) -> np.ndarray:
  return np.array(
    [
      [0.0, -v[2], v[1]],
      [v[2], 0.0, -v[0]],
      [-v[1], v[0], 0.0],
    ]
  )


def solve_mount_qp(
  units: list[ContactUnit],
  mass: float,
  gravity_w: np.ndarray,
  com_meas: np.ndarray,
  cfg: MountQpCfg | None = None,
  warm_start: MountQpSolution | None = None,
  extra_force_w: np.ndarray | None = None,
  extra_moment_w: np.ndarray | None = None,
  com_anchor: np.ndarray | None = None,
) -> MountQpSolution:
  """Solve the mixed anchor/floor load-distribution QP (module docstring).

  ``com_anchor``: the point the CoM regularizer pulls toward (default: the
  measurement). On floor support the moment equality leaves c free anywhere
  over the support polygon, so anchoring the regularizer at the MEASURED
  CoM is zero-stiffness self-tracking — the standing robot falls over while
  the QP happily re-approves every new lean. Anchoring at the clip's
  reference CoM gives the WBC CoM task a real restoring force; the balance
  equality still moves c* off the anchor when the load geometry (e.g. a
  press pushback) demands it.
  """
  if minimize is None:
    raise ImportError("solve_mount_qp requires scipy.")
  cfg = cfg or MountQpCfg()
  num_k = len(units)
  w_vec = mass * np.asarray(gravity_w, dtype=np.float64)
  f_ext = (
    np.zeros(3)
    if extra_force_w is None
    else np.asarray(extra_force_w, dtype=np.float64)
  )
  m_ext = (
    np.zeros(3)
    if extra_moment_w is None
    else np.asarray(extra_moment_w, dtype=np.float64)
  )
  w_norm = float(np.linalg.norm(w_vec))
  com_meas = np.asarray(com_meas, dtype=np.float64)
  com_anchor = (
    com_meas
    if com_anchor is None
    else np.asarray(com_anchor, dtype=np.float64)
  )

  if num_k == 0 or w_norm < 1.0e-9:
    return MountQpSolution(
      lambda_local=np.zeros((num_k, 3)),
      force_on_robot_w=np.zeros((num_k, 3)),
      com_target=com_meas.copy(),
      utilization=np.zeros(num_k),
      margin=1.0,
      feasible=num_k > 0,
      residual=w_norm,
      unit_names=tuple(u.name for u in units),
      info={"num_units": num_k},
    )

  g_hat = w_vec / w_norm
  is_wang = np.array([u.kind == "wang" for u in units], dtype=bool)
  r = np.stack([np.asarray(u.pos_w, dtype=np.float64) for u in units])
  frames = np.stack([np.asarray(u.frame_w, dtype=np.float64) for u in units])
  # Force ON the robot per unit column-frame: wang reaction is −R λ (λ is
  # the load the robot puts on the wall), ground is +R λ directly.
  sign = np.where(is_wang, -1.0, 1.0)
  rot_eff = frames * sign[:, None, None]

  caps = np.array(
    [
      cfg.gamma_target * cfg.f0_min * float(np.clip(u.cap_scale, 0.05, 1.0))
      if u.kind == "wang"
      else w_norm
      for u in units
    ]
  )
  full_caps = np.where(is_wang, caps / max(cfg.gamma_target, 1.0e-6), w_norm)
  slack_cap = max(cfg.pad_torque_cap * num_k, 1.0e-6)
  w_target = -w_vec - f_ext
  # Balance: Σ F_on_robot + m g⃗ + F_ext = 0  →  Σ F_k = −(m g⃗ + F_ext).
  # (gravity_w already points down, so w_vec = m g⃗ has negative z.)

  mu_hat = cfg.ground_friction / math.sqrt(2.0)

  n_lam = 3 * num_k
  caps_n = caps / w_norm
  lever = 0.5 * w_norm
  n_var = n_lam + 6

  def _split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
      x[:n_lam].reshape(num_k, 3),
      x[n_lam : n_lam + 3],
      x[n_lam + 3 :],
    )

  # Constant equality jacobian (exact — SLSQP treats the problem as the QP
  # it is; numerical differencing false-converged in the parent).
  eq_jac = np.zeros((6, n_var))
  for k in range(num_k):
    eq_jac[0:3, 3 * k : 3 * k + 3] = rot_eff[k]
    eq_jac[3:6, 3 * k : 3 * k + 3] = _skew(r[k]) @ rot_eff[k] * (
      w_norm / lever
    )
  # moment_res carries +cross(c, w_vec) = −skew(w_vec)·c, hence the minus.
  eq_jac[3:6, n_lam : n_lam + 3] = -_skew(w_vec) / lever
  eq_jac[3:6, n_lam + 3 :] = -np.eye(3) * (slack_cap / lever)

  def _forces_w(lam_n: np.ndarray) -> np.ndarray:
    return np.einsum("kij,kj->ki", rot_eff, lam_n)  # weight units

  def _equality(x: np.ndarray) -> np.ndarray:
    lam_n, c, slack_n = _split(x)
    forces = _forces_w(lam_n)
    force_res = forces.sum(axis=0) - w_target / w_norm
    moment = np.cross(r, forces).sum(axis=0) * w_norm  # N·m about origin
    # Moment balance about the origin: Σ r×F + c×(m g⃗) + M_ext = s
    moment_res = (
      moment + np.cross(c, w_vec) + m_ext - slack_n * slack_cap
    ) / lever
    return np.concatenate([force_res, moment_res])

  wang_w = np.where(is_wang, 1.0 / (caps_n * caps_n), cfg.ground_reg)

  def _objective(x: np.ndarray) -> float:
    lam_n, c, slack_n = _split(x)
    util = (lam_n * lam_n).sum(axis=1) * wang_w
    dc = c - com_anchor
    return (
      float(util.sum())
      + cfg.com_reg * float(dc @ dc)
      + cfg.com_gravity_pin * float(dc @ g_hat) ** 2
      + cfg.slack_weight * float(slack_n @ slack_n)
    )

  def _objective_grad(x: np.ndarray) -> np.ndarray:
    lam_n, c, slack_n = _split(x)
    grad = np.zeros(n_var)
    grad[:n_lam] = (2.0 * lam_n * wang_w[:, None]).reshape(-1)
    dc = c - com_anchor
    grad[n_lam : n_lam + 3] = 2.0 * cfg.com_reg * dc + (
      2.0 * cfg.com_gravity_pin * float(dc @ g_hat)
    ) * g_hat
    grad[n_lam + 3 :] = 2.0 * cfg.slack_weight * slack_n
    return grad

  share = np.array([u.share_cap for u in units], dtype=np.float64)
  share_idx = np.flatnonzero(np.isfinite(share))
  wang_idx = np.flatnonzero(is_wang)
  gnd_idx = np.flatnonzero(~is_wang)

  # Cone rows, all ≥ 0 feasible, in normalized units.
  def _cone_rows(x: np.ndarray) -> np.ndarray:
    lam_n, c, _ = _split(x)
    rows: list[np.ndarray] = []
    if len(wang_idx):
      lw = lam_n[wang_idx] * w_norm
      f_t, f_l, f_n = lw[:, 0], lw[:, 1], lw[:, 2]
      tension = _tension_plus(f_n, cfg.softplus_eps)
      cw = caps[wang_idx]
      rows += [
        (f_t - cfg.f_t_min) / w_norm,
        (cfg.tension_ratio * f_t - f_n) / w_norm,
        (cfg.lateral_ratio * f_t - f_l) / w_norm,
        (cfg.lateral_ratio * f_t + f_l) / w_norm,
        (cw * cw - (f_t * f_t + f_l * f_l + tension * tension))
        / (w_norm * w_norm),
      ]
    if len(gnd_idx):
      lg = lam_n[gnd_idx]  # weight units suffice — the pyramid is homogeneous
      f_x, f_y, f_z = lg[:, 0], lg[:, 1], lg[:, 2]
      rows += [
        f_z,
        mu_hat * f_z - f_x,
        mu_hat * f_z + f_x,
        mu_hat * f_z - f_y,
        mu_hat * f_z + f_y,
      ]
    rows.append(np.array([cfg.com_x_max - c[0]]))
    if len(share_idx):
      mag2 = (lam_n[share_idx] * lam_n[share_idx]).sum(axis=1)
      rows.append(share[share_idx] * share[share_idx] - mag2)
    return np.concatenate(rows)

  def _cone_jac(x: np.ndarray) -> np.ndarray:
    lam_n, _, _ = _split(x)
    n_rows = (
      5 * len(wang_idx) + 5 * len(gnd_idx) + 1 + len(share_idx)
    )
    jac = np.zeros((n_rows, n_var))
    row = 0
    if len(wang_idx):
      nw = len(wang_idx)
      f_n = lam_n[wang_idx, 2] * w_norm
      tension = _tension_plus(f_n, cfg.softplus_eps)
      t_grad = _tension_plus_grad(f_n, cfg.softplus_eps)
      for j, k in enumerate(wang_idx):
        col = 3 * k
        jac[row + j, col] = 1.0  # f_t floor
        jac[row + nw + j, col] = cfg.tension_ratio
        jac[row + nw + j, col + 2] = -1.0
        jac[row + 2 * nw + j, col] = cfg.lateral_ratio
        jac[row + 2 * nw + j, col + 1] = -1.0
        jac[row + 3 * nw + j, col] = cfg.lateral_ratio
        jac[row + 3 * nw + j, col + 1] = 1.0
        rr = row + 4 * nw + j  # capacity
        jac[rr, col] = -2.0 * lam_n[k, 0]
        jac[rr, col + 1] = -2.0 * lam_n[k, 1]
        jac[rr, col + 2] = -2.0 * tension[j] * t_grad[j] / w_norm
      row += 5 * nw
    if len(gnd_idx):
      ng = len(gnd_idx)
      for j, k in enumerate(gnd_idx):
        col = 3 * k
        jac[row + j, col + 2] = 1.0  # f_z ≥ 0
        jac[row + ng + j, col] = -1.0
        jac[row + ng + j, col + 2] = mu_hat
        jac[row + 2 * ng + j, col] = 1.0
        jac[row + 2 * ng + j, col + 2] = mu_hat
        jac[row + 3 * ng + j, col + 1] = -1.0
        jac[row + 3 * ng + j, col + 2] = mu_hat
        jac[row + 4 * ng + j, col + 1] = 1.0
        jac[row + 4 * ng + j, col + 2] = mu_hat
      row += 5 * ng
    jac[row, n_lam] = -1.0  # com_x ceiling
    row += 1
    for j, k in enumerate(share_idx):
      jac[row + j, 3 * k : 3 * k + 3] = -2.0 * lam_n[k]
    return jac

  ineqs = [{"type": "ineq", "fun": _cone_rows, "jac": _cone_jac}]
  bounds = [(None, None)] * (n_lam + 3) + [(-1.0, 1.0)] * 3

  def _fresh_seed() -> np.ndarray:
    lam0 = np.zeros((num_k, 3))
    for k in range(num_k):
      # Even split of the support target through each unit's effective frame.
      lam0[k] = rot_eff[k].T @ (w_target / num_k) / w_norm
    if len(wang_idx):
      lam0[wang_idx, 0] = np.maximum(
        lam0[wang_idx, 0], (cfg.f_t_min + 1.0) / w_norm
      )
    if len(gnd_idx):
      lam0[gnd_idx, 2] = np.maximum(lam0[gnd_idx, 2], 0.05)
    return np.concatenate([lam0.reshape(-1), com_meas.copy(), np.zeros(3)])

  names = tuple(u.name for u in units)
  # Warm-start the loads only, and only when the support set is unchanged —
  # the mount's set churns (feet lift, hands latch) and index-matched reuse
  # across different sets is silent garbage. c re-seeds at the measurement
  # every solve (the parent observed it drifting 0.5 m up the gravity line
  # when chained).
  if (
    warm_start is not None
    and warm_start.feasible
    and warm_start.unit_names == names
  ):
    x0 = np.concatenate(
      [
        warm_start.lambda_local.reshape(-1) / w_norm,
        com_meas.copy(),
        np.zeros(3),
      ]
    )
  else:
    x0 = _fresh_seed()

  eq_cons = {"type": "eq", "fun": _equality, "jac": lambda x: eq_jac}
  res = minimize(
    _objective,
    x0,
    jac=_objective_grad,
    method="SLSQP",
    bounds=bounds,
    constraints=[eq_cons] + ineqs,
    options={"maxiter": 200, "ftol": 1.0e-9},
  )
  feasible = bool(res.success) and float(
    np.linalg.norm(_equality(res.x)[:3])
  ) < 0.02  # 2% of body weight
  x_sol = res.x
  status = int(res.status)

  if not feasible:
    res_retry = minimize(
      _objective,
      _fresh_seed(),
      jac=_objective_grad,
      method="SLSQP",
      bounds=bounds,
      constraints=[eq_cons] + ineqs,
      options={"maxiter": 200, "ftol": 1.0e-9},
    )
    if res_retry.success and float(
      np.linalg.norm(_equality(res_retry.x)[:3])
    ) < 0.02:
      feasible = True
      x_sol = res_retry.x
      status = int(res_retry.status)
    else:

      def _penalized(x: np.ndarray) -> float:
        eq = _equality(x)
        return _objective(x) + cfg.penalty_weight * float(eq @ eq)

      def _penalized_grad(x: np.ndarray) -> np.ndarray:
        eq = _equality(x)
        return _objective_grad(x) + 2.0 * cfg.penalty_weight * (eq_jac.T @ eq)

      res2 = minimize(
        _penalized,
        _fresh_seed(),
        jac=_penalized_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=ineqs,
        options={"maxiter": 200, "ftol": 1.0e-9},
      )
      if np.isfinite(res2.fun):
        x_sol = res2.x
      status = int(res2.status)

  lam_sol_n, c_sol, slack_sol_n = _split(x_sol)
  lam_sol = lam_sol_n * w_norm
  forces_w = np.einsum("kij,kj->ki", rot_eff, lam_sol)
  util = np.zeros(num_k)
  if len(wang_idx):
    lw = lam_sol[wang_idx]
    tension = _tension_plus(lw[:, 2], cfg.softplus_eps)
    load_mag = np.sqrt(lw[:, 0] ** 2 + lw[:, 1] ** 2 + tension**2)
    util[wang_idx] = load_mag / np.maximum(full_caps[wang_idx], 1.0e-9)
  if len(gnd_idx):
    util[gnd_idx] = (
      np.linalg.norm(lam_sol[gnd_idx], axis=-1) / w_norm
    )
  eq_res = _equality(x_sol)
  wang_util = util[wang_idx] if len(wang_idx) else np.zeros(0)
  return MountQpSolution(
    lambda_local=lam_sol,
    force_on_robot_w=forces_w,
    com_target=c_sol.copy(),
    utilization=util,
    margin=float(1.0 - wang_util.max()) if len(wang_idx) else 1.0,
    feasible=feasible,
    residual=float(np.linalg.norm(eq_res) * w_norm),
    unit_names=names,
    info={
      "num_units": num_k,
      "slsqp_status": status,
      "objective": float(res.fun) if np.isfinite(res.fun) else float("inf"),
      "moment_slack": slack_sol_n * slack_cap,
    },
  )
