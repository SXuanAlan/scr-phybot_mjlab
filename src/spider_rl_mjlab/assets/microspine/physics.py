"""Microspine adhesion physics based on Jiang et al. 2018.

Pure functions implementing the stochastic spine-asperity interaction model.
All functions are vectorized over batch dimensions for GPU efficiency.

Reference equations from the paper:
- Eq. 2: Two-phase asperity spatial distribution p(α, λ; x)
- Eq. 3: Cumulative asperity strength probability Pr(F)
- Eq. 4: Expected shear stress E[σ]
- Eq. 5: Single spine force F(x) = K·(d_stroke - x)
- Eq. 6-8: Opposed array force balance
- Eq. 1/8: Loading angle limit θ_th = arctan(h_mis / d_unit); spine fails when
           arctan(F_n / F_s) ≥ θ_th (pull-out condition)
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
    x = torch.linspace(0, d_stroke, n_points, device=alpha.device, dtype=alpha.dtype)
    dx = d_stroke / (n_points - 1)

    a = alpha.unsqueeze(-1)
    lam = lambda_.unsqueeze(-1)
    e = eta.unsqueeze(-1)
    x_exp = x.unsqueeze(0)

    lam_m = lam * 1000.0
    p_exp = a * lam_m * torch.exp(-lam_m * x_exp)

    force = compute_spine_force(x_exp, K, d_stroke, F_max)
    pr = compute_engagement_probability(force, e)

    integrand = p_exp * force * pr
    integral = torch.trapezoid(integrand, dx=dx, dim=-1)

    F_0 = compute_spine_force(torch.zeros(1, device=alpha.device), K, d_stroke, F_max)
    Pr_0 = compute_engagement_probability(F_0, eta)
    delta_contrib = (1.0 - alpha) * F_0.squeeze() * Pr_0.squeeze()

    stress = (integral + delta_contrib) / tile_area_factor
    return stress


def compute_loading_angle_failure(
    v_pullout: torch.Tensor,
    v_shear: torch.Tensor,
    theta_th: float,
) -> torch.Tensor:
    """Check whether the loading angle exceeds the pull-out threshold (Eq. 8).

    A spine/asperity contact fails when the force direction tilts too far away
    from the surface plane:

        arctan(F_n / F_s) ≥ θ_th    (Eq. 8)

    Force direction is proxied by EE velocity direction.

    Args:
        v_pullout: Wall-normal pull-out speed per EE [m/s], clamped ≥ 0.
                   Shape: (N, E).
        v_shear:   Wall-parallel speed magnitude per EE [m/s]. Shape: (N, E).
        theta_th:  Loading angle limit [rad] = arctan(h_mis / d_unit).

    Returns:
        Boolean tensor, True where spines should disengage. Shape: (N, E).
    """
    loading_angle = torch.atan2(v_pullout.clamp(min=0.0), v_shear + 1e-6)
    return loading_angle >= theta_th


def compute_total_shear_force(
    travel: torch.Tensor,
    engaged: torch.Tensor,
    K: float,
    d_stroke: float,
    F_max: float,
) -> torch.Tensor:
    """Total shear force per EE from all engaged spines.

    Args:
        travel: Spine travel distances. Shape: (N, E, S).
        engaged: Boolean mask of engaged spines. Shape: (N, E, S).
        K: Spine stiffness [N/m].
        d_stroke: Maximum travel [m].
        F_max: Maximum force per spine [N].

    Returns:
        Total shear force per EE [N]. Shape: (N, E).
    """
    force_per_spine = compute_spine_force(travel, K, d_stroke, F_max)
    return (force_per_spine * engaged.float()).sum(dim=-1)
