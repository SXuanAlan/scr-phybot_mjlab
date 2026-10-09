"""Training schedules of the 80-degree transition task, and values several sections share.

A schedule is a list of {"iter", "value"} stages over training iterations:
its term starts at stage 0 and a curriculum term anneals it (curricula.py).
Stages at iteration 50000 and later lie past the training horizon; those
tightenings are advanced by hand.
"""

from __future__ import annotations


def stage0(stages: list[dict]) -> float:
  return float(sorted(stages, key=lambda s: s["iter"])[0]["value"])


def stage_final(stages: list[dict]) -> float:
  return float(sorted(stages, key=lambda s: s["iter"])[-1]["value"])


# Rip and engage-cone violations within 0.12 s of a birth are exempt, the
# span of the seed judge guard; later losses are the policy's to prevent.
SEED_SETTLE_EXEMPT_S = 0.12

# Bill per ripped scheduled pad per second until it is re-welded, kept under
# the per-pad tracking income so a rip is never worth dying over.
RIP_PENALTY_STAGES = [
  {"iter": 0, "value": 0.5},
  {"iter": 1000, "value": 1.5},
  {"iter": 2500, "value": 3.0},
]

# Pad gimbal stop travel (deg): opened wide, then held at each step on the
# way to the real +-3 deg stop.
GIMBAL_TRAVEL_STAGES = [
  {"iter": 0, "value": 15.0},
  {"iter": 7900, "value": 15.0},
  {"iter": 8100, "value": 8.0},
  {"iter": 10000, "value": 8.0},
  {"iter": 10200, "value": 5.0},
  {"iter": 16000, "value": 5.0},
  {"iter": 16200, "value": 3.0},
]

# Maximum down-wall pad drift while loading (m/s): opened to the observed
# touches, then annealed back to the specification.
ANTIRAKE_LADDER = [
  {"iter": 0, "value": 0.05},
  {"iter": 50000, "value": 0.02},
]

# The contact law: a scheduled contact missing for the grace ends the
# episode, and a weld within the radius satisfies the schedule. The
# engage-progress potential uses the same radius.
CONTACT_GRACE_STAGES = [
  {"iter": 0, "value": 1.0},
]
ANCHOR_RADIUS_STAGES = [
  {"iter": 0, "value": 0.15},
  {"iter": 50000, "value": 0.10},
  {"iter": 52000, "value": 0.08},
]

# Tracking-deviation thresholds (m) of the anchor and the pad faces.
ANCHOR_POS_STAGES = [{"iter": 0, "value": 0.35}, {"iter": 50000, "value": 0.25}]
PAD_POS_STAGES = [{"iter": 0, "value": 0.30}, {"iter": 50000, "value": 0.20}]

# Engage-cone limits: pad tilt (deg), grind and slam speed (m/s), force (N).
TILT_STAGES = [
  {"iter": 0, "value": 15.0},
  {"iter": 50000, "value": 10.0},
  {"iter": 52000, "value": 6.0},
]
GRIND_STAGES = [
  {"iter": 0, "value": 0.30},
  {"iter": 50000, "value": 0.20},
  {"iter": 52000, "value": 0.12},
]
SLAM_STAGES = [
  {"iter": 0, "value": 0.60},
  {"iter": 50000, "value": 0.45},
  {"iter": 52000, "value": 0.30},
]
FORCE_STAGES = [
  {"iter": 0, "value": 80.0},
  {"iter": 50000, "value": 60.0},
  {"iter": 52000, "value": 40.0},
]
# The cone's termination probability starts near zero, so early touches
# survive and get paid before any press-and-hold skill exists.
SOFT_P_STAGES = [
  {"iter": 0, "value": 0.05},
  {"iter": 50000, "value": 0.5},
  {"iter": 52000, "value": 1.0},
]
