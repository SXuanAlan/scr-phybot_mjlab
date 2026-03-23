"""Configuration for the microspine adhesion model.

Based on Jiang et al. 2018, "Stochastic models of compliant spine arrays
for rough surface grasping."
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MicrospineCfg:
    """Microspine adhesion model parameters.

    Combines spine mechanics (Jiang et al. Eq. 4-5), surface asperity
    properties, and opposed-array configuration (Section 4).
    """

    # -- End-effector bodies --------------------------------------------------
    ee_body_names: tuple[str, ...] = (
        "left_ankle_roll",
        "right_ankle_roll",
        "left_elbow_pitch",
        "right_elbow_pitch",
    )

    # -- Spine mechanics (Jiang et al.) ----------------------------------------
    n_spines: int = 20
    """Number of spines per tile."""

    d_unit: float = 0.008
    """Spine body length [m]. (8 mm)"""

    d_stroke: float = 0.006
    """Maximum spine travel before hard stop [m]. (6 mm)"""

    d_flexure: float = 0.010
    """Original flexure length [m]. (10 mm)"""

    F_max: float = 2.5
    """Maximum force per spine at full stroke [N]."""

    w: float = 0.010
    """Inter-row spacing / spine width [m]. (10 mm)"""

    # -- MuJoCo adhesion actuator ---------------------------------------------
    max_adhesion_gain: float = 50.0
    """Adhesion actuator gain [N]. Equals n_spines * F_max at full engagement."""

    # -- Surface asperity parameters (domain-randomizable) --------------------
    alpha: float = 0.25
    """Probability of non-zero travel before engagement (1 - alpha = prob of
    immediate engagement). See Eq. 2."""

    alpha_range: tuple[float, float] = (0.20, 0.31)
    """Uniform range for domain randomization of alpha."""

    lambda_: float = 0.40
    """Average number of asperities per mm. Controls exponential term in
    the two-phase asperity distribution. See Eq. 2."""

    lambda_range: tuple[float, float] = (0.33, 0.49)
    """Uniform range for domain randomization of lambda."""

    eta: float = 0.20
    """Probability that an asperity can sustain >= 3 N (strong asperity).
    See Fig. 4 and Eq. 3."""

    eta_range: tuple[float, float] = (0.027, 0.53)
    """Uniform range for domain randomization of eta."""

    # -- Opposed array (Section 4) --------------------------------------------
    preload_force: float = 1.0
    """Preload spring initial force [N]."""

    theta_threshold: float = 0.8
    """Loading angle limit [rad]. Beyond this angle, spines slip."""

    # -- Engagement geometry --------------------------------------------------
    contact_threshold: float = 0.05
    """Maximum distance to wall surface for spine engagement [m]."""

    # -- Wall reference -------------------------------------------------------
    wall_x_offset: float = 1.5
    """Wall inner face x-position relative to each env origin [m]."""

    # -- Randomization --------------------------------------------------------
    randomize_surface: bool = True
    """Whether to randomize surface parameters (alpha, lambda, eta) per env
    on reset. Useful for sim-to-real domain randomization."""

    # -- Derived properties ---------------------------------------------------
    @property
    def K(self) -> float:
        """Spine stiffness [N/m]. K = F_max / d_stroke."""
        return self.F_max / self.d_stroke

    @property
    def tile_area_factor(self) -> float:
        """Denominator for stress calculation: (d_unit + d_flexure + d_stroke) * w."""
        return (self.d_unit + self.d_flexure + self.d_stroke) * self.w
