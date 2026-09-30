"""Closed 3D microspine limit surface after Wang, Jiang & Cutkosky (IJRR 2017).

Reference: "Design and modeling of linearly-constrained compliant spines for
human-scale locomotion on rocky surfaces" (docs/wang-et-al-2017-*.pdf),
Sections 2.2-3.2 (eqs 1-4, 9-12). The model predicts the adhesion capacity of
a unidirectional spine array as a function of load *direction*; the resulting
surface is closed: capacity peaks along the rake (pure engaged shear), tension
capacity exists only alongside simultaneous shear (the forward lobe closes at
the origin — pure normal pull-off holds ~nothing), lateral capacity is a
fraction of the rake capacity (eq 12), and against-rake capacity collapses to
the weak side-contact residual (Fig 4(c); the non-convex x<0 region of Fig 8).

Load direction in the engage frame (t̂ = rake, l̂ = lateral, n̂ = outward
normal, matching ``MicrospineAffordance._engage_frame``):

  theta = atan2(|f_l|, f_t)        azimuth in the wall plane, 0..180 deg
  phi   = atan2(max(f_n,0), |f_in-plane|)   elevation toward tension, 0..90 deg

Capacity along that direction is ``F0 * max(R(theta, phi), r_floor)`` with F0
the pure-rake-shear capacity (per-env DR sample) and R the normalized surface
value tabulated here. Compression (f_n < 0) is clamped out of phi: pressing in
is carried by the ordinary sphere-wall contacts, whose friction realizes the
paper's Sec 3.4 substrate-friction extension (eq 13-14) physically.

Model ingredients and convention choices (the paper's eq-11 angle conventions
are internally inconsistent with its Sec 3.1 worked example; conventions here
are resolved to reproduce that example and the Fig 8 anchor ratios):

  * Slip window (eqs 1, 11): a tip engaged on an asperity of slope psi holds
    until the load leaves the local friction cone, i.e. usable slopes are
    psi >= psi_c + phi with psi_c = arctan(1/mu_tip). Raising phi shrinks the
    usable population from below; the upper end is psi_c + delta_psi.
  * Population (eq 9): linearly decaying slope density over
    [psi_c, psi_c + delta_psi]; its integral from psi_c + x is (1 - x/dpsi)^2.
  * Strength (eqs 2-3): tip engagement (strength c_i, dimensionless >> 1)
    exists for slopes psi <= beta_eff + 90 - alpha/2; steeper contacts ride
    at that effective slope (eq 3) with unit side/shaft strength (Fig 4(c))
    and slip once it falls below the slip bound. As beta_eff goes negative on
    the against-rake half, the tip-eligible window narrows continuously and
    empties near theta = 180 deg — this produces both the smooth azimuth
    falloff and the small non-convex against-rake residual of Fig 8 without
    any extra gating. mu_tip must satisfy arctan(1/mu_tip) close to
    -beta + 90 - alpha/2 for that residual to land near the paper's ~0.05
    (larger mu opens the reverse window, smaller slams it to zero).
  * Channel slip (eqs 4, 19): factor sin(beta_eff + 90 - phi - atan(mu_s)) *
    (1 + mu_s^2), unity inside the no-slip band phi <= beta_eff + atan(mu_s),
    clamped to [0, 1].
  * Azimuth (eq 12): beta_eff(theta) = arcsin(sin(beta) * cos(theta)),
    negative on the against-rake half.

Default parameters are fit to the paper's single-tile 3D surface (Fig 8),
normalized anchors: peak tension ratio ~0.12 at ~15 deg elevation (paper
~0.14 @ ~16 deg), pure lateral 0.57 (~0.56), pure reverse 0.043 (~0.05),
forward lobe closing near 30 deg (~35-40 deg), azimuth profile smooth
(max grid step < 0.01, no knife edges for the release check to chatter on).
``scripts/plot_wang_limit_surface.py`` regenerates these numbers and the
cross-section figure.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from scr_phybot_mjab.microspine.config import MicrospineLimitSurfaceCfg

_DEG = math.pi / 180.0


def _population_integral(lo: torch.Tensor, hi: torch.Tensor, base_lo: float, delta_psi: float) -> torch.Tensor:
  """Integral of the eq-9 linear slope density over [lo, hi] (degrees)."""

  def cdf(x: torch.Tensor) -> torch.Tensor:
    u = ((x - base_lo) / delta_psi).clamp(0.0, 1.0)
    return 1.0 - (1.0 - u) ** 2

  return (cdf(hi) - cdf(lo)).clamp_min(0.0)


def _unnormalized_capacity(
  phi_deg: torch.Tensor,
  beta_eff_deg: torch.Tensor,
  *,
  mu_tip: float,
  delta_psi_deg: float,
  alpha_deg: float,
  mu_channel: float,
  c_i: float,
) -> torch.Tensor:
  """N(phi, beta_eff) of the module docstring, broadcast over inputs."""
  psi_c = math.degrees(math.atan(1.0 / mu_tip))
  base_hi = psi_c + delta_psi_deg
  psi_thr = beta_eff_deg + 90.0 - alpha_deg / 2.0
  slip_lo = psi_c + phi_deg

  # Tip-engaged slopes [slip_lo, min(psi_thr, base_hi)], strength c_i. The
  # window narrows continuously as beta_eff drops (against-rake azimuths)
  # and empties near theta = 180 deg — no explicit direction gate.
  tip_hi = torch.minimum(psi_thr, torch.full_like(psi_thr, base_hi))
  w_tip = _population_integral(slip_lo, tip_hi, psi_c, delta_psi_deg)

  # Side/shaft contacts above psi_thr ride at effective slope psi_thr (eq 3)
  # with unit strength; they all slip once psi_thr drops below the slip bound.
  shaft_lo = torch.maximum(slip_lo, psi_thr)
  w_shaft = _population_integral(shaft_lo, torch.full_like(shaft_lo, base_hi), psi_c, delta_psi_deg)
  w_shaft = torch.where(psi_thr >= slip_lo, w_shaft, torch.zeros_like(w_shaft))

  strength = c_i * w_tip + w_shaft

  # Channel slip factor (eqs 4, 19).
  atan_mu_s = math.degrees(math.atan(mu_channel))
  cc_raw = torch.sin((beta_eff_deg + 90.0 - phi_deg - atan_mu_s) * _DEG) * (1.0 + mu_channel**2)
  cc = torch.where(
    phi_deg <= beta_eff_deg + atan_mu_s,
    torch.ones_like(cc_raw),
    cc_raw.clamp(0.0, 1.0),
  )
  return cc * strength


def build_capacity_lut(cfg: MicrospineLimitSurfaceCfg, device: torch.device | str) -> torch.Tensor:
  """Normalized capacity table R[theta_bin, phi_bin] in [0, 1], R[0, 0] = 1.

  Rows span azimuth theta in [0, 180] deg, columns elevation phi in [0, 90]
  deg, both uniformly. Built once at startup; ~65 kB at the default 181x91.
  """
  kwargs = dict(
    mu_tip=cfg.wang_mu_tip,
    delta_psi_deg=cfg.wang_delta_psi_deg,
    alpha_deg=cfg.wang_alpha_deg,
    mu_channel=cfg.wang_mu_channel,
    c_i=cfg.wang_c_i,
  )
  theta = torch.linspace(0.0, 180.0, cfg.wang_lut_theta_bins, dtype=torch.float64)
  phi = torch.linspace(0.0, 90.0, cfg.wang_lut_phi_bins, dtype=torch.float64)
  beta_eff = torch.rad2deg(torch.arcsin(math.sin(cfg.wang_beta_deg * _DEG) * torch.cos(theta * _DEG)))
  n = _unnormalized_capacity(phi.unsqueeze(0), beta_eff.unsqueeze(1), **kwargs)
  n0 = _unnormalized_capacity(torch.zeros(()), torch.full((), float(cfg.wang_beta_deg)), **kwargs)
  return (n / n0).clamp(0.0, 1.0).to(device=device, dtype=torch.float32)


def capacity_ratio(
  lut: torch.Tensor,
  f_t: torch.Tensor,
  f_l: torch.Tensor,
  f_n: torch.Tensor,
) -> torch.Tensor:
  """Bilinear LUT sample of R along the direction of an engage-frame load.

  Compression (f_n < 0) is clamped to the phi = 0 row: the anchor envelope
  only judges the tension component, pressing in is carried by contacts.
  Zero load maps to (theta, phi) = (0, 0), i.e. R = 1 (maximum margin).
  """
  tension = f_n.clamp_min(0.0)
  in_plane = torch.sqrt(f_t**2 + f_l**2)
  theta = torch.atan2(f_l.abs(), f_t) / _DEG  # 0..180 deg
  phi = torch.atan2(tension, in_plane) / _DEG  # 0..90 deg

  n_theta, n_phi = lut.shape
  ti = (theta / 180.0 * (n_theta - 1)).clamp(0.0, float(n_theta - 1))
  pi_ = (phi / 90.0 * (n_phi - 1)).clamp(0.0, float(n_phi - 1))
  t0 = ti.floor().long().clamp(max=n_theta - 2)
  p0 = pi_.floor().long().clamp(max=n_phi - 2)
  tw = ti - t0.float()
  pw = pi_ - p0.float()
  r00 = lut[t0, p0]
  r01 = lut[t0, p0 + 1]
  r10 = lut[t0 + 1, p0]
  r11 = lut[t0 + 1, p0 + 1]
  return r00 * (1 - tw) * (1 - pw) + r01 * (1 - tw) * pw + r10 * tw * (1 - pw) + r11 * tw * pw


def effective_capacity(
  lut: torch.Tensor,
  cfg: MicrospineLimitSurfaceCfg,
  f0: torch.Tensor,
  f_t: torch.Tensor,
  f_l: torch.Tensor,
  f_n: torch.Tensor,
) -> torch.Tensor:
  """Capacity magnitude [N] along the load direction: F0 * max(R, r_floor).

  ``wang_r_floor`` models the stochastic residual adhesion the deterministic
  surface predicts as zero (Fig 6/8 scatter): a small isotropic scrap that
  also keeps release margins well-conditioned where R -> 0.
  """
  r = capacity_ratio(lut, f_t, f_l, f_n)
  return f0 * r.clamp_min(cfg.wang_r_floor)
