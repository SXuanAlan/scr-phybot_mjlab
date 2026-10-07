"""Why a latched pad released (``MicrospineAffordance.release_reason``)."""

from enum import IntEnum


class ReleaseReason(IntEnum):
  """Release codes the affordance writes on the step a latch drops.

  Values 1, 3-6 are unassigned; they stay free so older logs still decode.
  """

  NONE = 0
  # The binding unit's normal load exceeded its empirical strength draw.
  NORMAL = 2
  # The binding unit's load left the measured shear/lateral domain.
  EMPIRICAL_DOMAIN = 7
  # The binding unit failed early in loading (its loading-quantile draw).
  EMPIRICAL_EARLY = 8
  # A moving grasp surface no longer covers the pad.
  OFF_SURFACE = 9
