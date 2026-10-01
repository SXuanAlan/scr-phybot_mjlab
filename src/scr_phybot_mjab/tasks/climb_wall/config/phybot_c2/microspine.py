"""Microspine pad physics of the climbing task."""

from __future__ import annotations

from scr_phybot_mjab.microspine.config import MicrospineCfg, MicrospineLimitSurfaceCfg

from .stance import CLIMB_EE_BODY_NAMES, PAD_PROXY_COLS, PAD_PROXY_ROWS


def climb_microspine_cfg(unit_layout: str) -> MicrospineCfg:
  """Seating and unit preload engage a pad; the hold judge alone releases it.

  ``unit_layout`` selects the active CAD unit sites: "all" twelve, or the
  four "corners" of the hardware variants. The grip event takes a shallow
  copy of this config, so it must be complete before that event is built.
  """
  return MicrospineCfg(
    ee_body_names=CLIMB_EE_BODY_NAMES,
    spheres_per_ee_rows=PAD_PROXY_ROWS,
    spheres_per_ee_cols=PAD_PROXY_COLS,
    # Fallback reset latches only: the motion command's RSI overwrites them
    # per env with the sampled clip frame's schedule (see seed_engaged).
    initial_engaged_ee_names=CLIMB_EE_BODY_NAMES,
    # Ordinary contact handles compression and seating only; tangential
    # support comes from the anchors.
    contact_friction=(0.2, 0.2, 0.005, 1.0e-4, 1.0e-4),
    # A 0.20 s cooldown and 0.06 s debounce; a longer pair leaves otherwise
    # valid contacts one step short of every re-engagement.
    reengage_cooldown_steps=10,
    engage_consecutive_steps=3,
    # The pad self-aligns through the passive wrist gimbal, so the weld is
    # rotationally rigid. Within the +-3 deg free travel only spring torque
    # is transmitted; at the stops structural reaction also enters the weld
    # moment (the stops' PLA strength is unmeasured).
    anchor_torquescale=1.0,
    # Latch only a fully seated pad with every active unit preloaded.
    engage_min_spine_fraction=1.0,
    engage_unit_preload_n=10.0,
    limit_surface=MicrospineLimitSurfaceCfg(unit_layout=unit_layout),
  )
