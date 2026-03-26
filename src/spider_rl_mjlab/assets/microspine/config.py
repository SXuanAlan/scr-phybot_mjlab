"""Configuration for the microspine adhesion model.

Based on Jiang et al. 2018, "Stochastic models of compliant spine arrays
for rough surface grasping."
"""

from __future__ import annotations

import math
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
    n_spines: int = 81*12
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

    # -- Spine/asperity geometry ----------------------------------------------
    h_mis: float = 0.001
    """Average asperity height above the spine contact point [m]. (1 mm)
    Measured from the excursion of the spine tip in the normal direction
    during engagement. See Eq. 1 and p. 676 of Jiang et al. 2018."""

    # -- Opposed array (Section 4) --------------------------------------------
    preload_force: float = 1.0
    """Preload spring initial force [N]."""

    # -- Engagement geometry --------------------------------------------------
    contact_threshold: float = 0.05
    """Maximum distance to wall surface for spine engagement [m]."""

    contact_hysteresis: float = 0.02
    """Extra distance beyond contact_threshold before losing contact [m].
    Prevents rapid toggling at the boundary. Disengage at
    contact_threshold + contact_hysteresis."""

    # -- Anchor PID constraint model -------------------------------------------
    anchor_kp: float = 100000.0
    """Proportional gain [N/m]. Immediate spring response.
    100N load → 1mm displacement."""

    anchor_ki: float = 50000.0
    """Integral gain [N/(m·s)]. Builds up holding force over time for
    smooth steady-state load bearing. Eliminates steady-state error."""

    anchor_kd: float = 500.0
    """Derivative gain [N·s/m]. Damping to suppress oscillation.
    Ramps up over 0.5s after contact to avoid initial impulse."""

    anchor_integral_limit: float = 0.01
    """Integral windup limit [m·s]. Prevents excessive force buildup."""

    # -- Randomization --------------------------------------------------------
    randomize_surface: bool = True
    """Whether to randomize surface parameters (alpha, lambda, eta) per env
    on reset. Useful for sim-to-real domain randomization."""

    # -- Derived properties ---------------------------------------------------
    @property
    def theta_threshold(self) -> float:
        """Loading angle limit [rad].

        From Eq. 1: tan(θ_m) = h_mis / d_unit, so θ_th = arctan(h_mis / d_unit).
        A spine disengages when arctan(F_n / F_s) ≥ θ_th (Eq. 8).
        With h_mis=1 mm, d_unit=8 mm → θ_th ≈ 0.124 rad (7.1°).
        """
        return math.atan(self.h_mis / self.d_unit)

    @property
    def K(self) -> float:
        """Spine stiffness [N/m]. K = F_max / d_stroke."""
        return self.F_max / self.d_stroke

    @property
    def tile_area_factor(self) -> float:
        """Denominator for stress calculation: (d_unit + d_flexure + d_stroke) * w."""
        return (self.d_unit + self.d_flexure + self.d_stroke) * self.w
