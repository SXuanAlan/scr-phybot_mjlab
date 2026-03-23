"""Microspine adhesion physics based on Jiang et al. 2018.

Pure functions implementing the stochastic spine-asperity interaction model.
All functions are vectorized over batch dimensions for GPU efficiency.

Reference equations from the paper:
- Eq. 2: Two-phase asperity spatial distribution p(α, λ; x)
- Eq. 3: Cumulative asperity strength probability Pr(F)
- Eq. 4: Expected shear stress E[σ]
- Eq. 5: Single spine force F(x) = K·(d_stroke - x)
- Eq. 6-8: Opposed array force balance
"""

from __future__ import annotations

import torch


def compute_spine_force(
    travel: torch.Tensor,
    K: float,
    d_stroke: float,
    F_max: float,
) -> torch.Tensor:
    """Compute force from a single spine given its travel distance.

    F(x) = K · (d_stroke - x), clamped to [0, F_max].  (Eq. 5)

    Args:
        travel: Travel distance x for each spine. Shape: (...).
        K: Spine stiffness [N/m].
        d_stroke: Maximum travel [m].
        F_max: Maximum spine force [N].

    Returns:
        Force per spine, same shape as travel.
    """
    force = K * (d_stroke - travel)
    return force.clamp(0.0, F_max)


def compute_engagement_probability(
    force: torch.Tensor,
    eta: torch.Tensor,
) -> torch.Tensor:
    """Probability that an asperity can sustain a given spine force.

    Piecewise cumulative probability (Eq. 3):
      Pr(F) = 1.0           if F <= 0.5 N
      Pr(F) = η + (1-η)·(3-F)/2.5  if 0.5 < F < 3.0 N  (linear interpolation)
      Pr(F) = η             if F >= 3.0 N

    Args:
        force: Spine force values [N]. Shape: (...).
        eta: Probability of strong asperities. Shape broadcastable to force.

    Returns:
        Hold probability per spine, same shape as force.
    """
    # Linear interpolation region: maps F from [0.5, 3.0] to probability [1.0, eta]
    interp = eta + (1.0 - eta) * (3.0 - force) / 2.5
    prob = torch.where(force <= 0.5, torch.ones_like(force), interp)
    prob = torch.where(force >= 3.0, eta.expand_as(force), prob)
    return prob.clamp(0.0, 1.0)


def compute_expected_shear_stress(
    alpha: torch.Tensor,
    lambda_: torch.Tensor,
    eta: torch.Tensor,
    K: float,
    d_stroke: float,
    F_max: float,
    tile_area_factor: float,
    n_points: int = 32,
) -> torch.Tensor:
    """Compute expected shear stress E[σ] via numerical integration.

    Implements Eq. 4:
      E[σ] = ∫₀^d_stroke p(α,λ;x) · F(x) · Pr(F(x)) dx / tile_area_factor

    Uses trapezoidal rule with fixed quadrature points (GPU-friendly).

    Args:
        alpha: Asperity spatial parameter. Shape: (batch,).
        lambda_: Asperity density [per mm]. Shape: (batch,).
        eta: Strong asperity probability. Shape: (batch,).
        K: Spine stiffness [N/m].
        d_stroke: Maximum travel [m].
        F_max: Maximum spine force [N].
        tile_area_factor: (d_unit + d_flexure + d_stroke) * w [m²].
        n_points: Number of quadrature points.

    Returns:
        Expected shear stress [Pa]. Shape: (batch,).
    """
    # Quadrature points: x ∈ [0, d_stroke], shape (n_points,)
    x = torch.linspace(0, d_stroke, n_points, device=alpha.device, dtype=alpha.dtype)
    dx = d_stroke / (n_points - 1)

    # Expand for broadcasting: alpha (batch, 1), x (1, n_points)
    a = alpha.unsqueeze(-1)        # (batch, 1)
    lam = lambda_.unsqueeze(-1)    # (batch, 1)
    e = eta.unsqueeze(-1)          # (batch, 1)
    x_exp = x.unsqueeze(0)         # (1, n_points)

    # Asperity spatial PDF (continuous part of Eq. 2): p_exp(x) = α·λ·e^(-λx)
    # The delta function term (1-α)·δ(x) is handled separately as the x=0 contribution.
    # Convert lambda from per-mm to per-m for consistent units with x in meters.
    lam_m = lam * 1000.0  # per mm → per m
    p_exp = a * lam_m * torch.exp(-lam_m * x_exp)  # (batch, n_points)

    # Add delta contribution at x=0: (1-α)·δ(x) → concentrated at first point
    # In discrete integration, δ(x)·dx ≈ 1 at x=0, so we add (1-α)/dx to p at x=0
    # But more cleanly: the initial term contributes (1-α) * F(0) * Pr(F(0))
    # separately from the integral.

    # Single spine force F(x) at each quadrature point
    force = compute_spine_force(x_exp, K, d_stroke, F_max)  # (1, n_points)

    # Asperity strength probability Pr(F(x))
    pr = compute_engagement_probability(force, e)  # (batch, n_points)

    # Integrand: p(x) · F(x) · Pr(F(x))
    integrand = p_exp * force * pr  # (batch, n_points)

    # Trapezoidal integration
    integral = torch.trapezoid(integrand, dx=dx, dim=-1)  # (batch,)

    # Add delta function contribution: (1-α) · F(0) · Pr(F(0))
    F_0 = compute_spine_force(torch.zeros(1, device=alpha.device), K, d_stroke, F_max)
    Pr_0 = compute_engagement_probability(F_0, eta)
    delta_contrib = (1.0 - alpha) * F_0.squeeze() * Pr_0.squeeze()  # (batch,)

    # Total expected stress
    stress = (integral + delta_contrib) / tile_area_factor  # (batch,)
    return stress


def compute_adhesion_activation(
    n_engaged: torch.Tensor,
    n_spines: int,
    alpha: torch.Tensor,
    lambda_: torch.Tensor,
    eta: torch.Tensor,
) -> torch.Tensor:
    """Map spine engagement state to adhesion actuator ctrl value [0, 1].

    The activation combines the engagement ratio with a surface quality
    factor derived from the asperity parameters.

    Args:
        n_engaged: Number of engaged spines per EE. Shape: (batch, n_ee).
        n_spines: Total spines per tile.
        alpha: Surface alpha parameter. Shape: (batch,) or (batch, 1).
        lambda_: Surface lambda parameter. Shape: (batch,) or (batch, 1).
        eta: Surface eta parameter. Shape: (batch,) or (batch, 1).

    Returns:
        Adhesion activation in [0, 1]. Shape: (batch, n_ee).
    """
    engaged_ratio = n_engaged.float() / n_spines  # (batch, n_ee)

    # Surface quality: higher eta and lambda → better adhesion
    # Normalize to roughly [0, 1] range based on parameter ranges
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(-1)
        lambda_ = lambda_.unsqueeze(-1)
        eta = eta.unsqueeze(-1)

    surface_quality = (eta * lambda_ / 0.49).clamp(0.0, 1.0)  # (batch, 1)

    activation = engaged_ratio * surface_quality  # (batch, n_ee)
    return activation.clamp(0.0, 1.0)
