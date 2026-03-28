"""Configuration for the microspine adhesion model.

Based on Jiang et al. 2018, "Stochastic models of compliant spine arrays
for rough surface grasping."
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MicrospineCfg:
    """Microspine adhesion model parameters.

    Combines spine mechanics (Jiang et al. Eq. 4-5), surface asperity
    properties, and contact friction configuration.
    """

    # -- End-effector bodies --------------------------------------------------
    ee_body_names: tuple[str, ...] = (
        "left_hand",
        "right_hand",
    )

    # -- Spine mechanics (Jiang et al.) ----------------------------------------
    n_spines: int = 81 * 12
    """Number of spines per tile (972)."""

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
    """Adhesion actuator gain [N]."""

    preload_force: float = 15.0
    """Minimum normal contact load [N] before spine engagement is allowed."""

    # -- Contact friction (anisotropic via contactfilter) ----------------------
    engaged_friction: float = 1000.0
    """Friction coefficient along wall-up (tangent1) when spines are engaged."""

    lateral_friction: float = 0.1
    """Friction coefficient along lateral direction (tangent2) when engaged."""

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
    h_mis: float = 0.005
    """Average asperity height above the spine contact point [m]. (5 mm)"""

    # -- Engagement geometry --------------------------------------------------
    contact_threshold: float = 0.05
    """Maximum distance to wall surface for spine engagement [m]."""

    contact_hysteresis: float = 0.02
    """Extra distance beyond contact_threshold before losing contact [m]."""

    max_penetration: float = 0.002
    """Maximum allowed surface penetration [m] before forcing disengagement."""

    penetration_hysteresis: float = 0.001
    """Extra penetration margin [m] before releasing an already-engaged contact."""

    # -- Randomization --------------------------------------------------------
    randomize_surface: bool = True
    """Whether to randomize surface parameters per env on reset."""

    # -- Derived properties ---------------------------------------------------
    @property
    def theta_threshold(self) -> float:
        """Loading angle limit [rad]. From Eq. 1: θ_th = arctan(h_mis / d_unit)."""
        return math.atan(self.h_mis / self.d_unit)

    @property
    def K(self) -> float:
        """Spine stiffness [N/m]. K = F_max / d_stroke."""
        return self.F_max / self.d_stroke

    @property
    def tile_area_factor(self) -> float:
        """Denominator for stress calculation: (d_unit + d_flexure + d_stroke) * w."""
        return (self.d_unit + self.d_flexure + self.d_stroke) * self.w
