"""The numbers of the mount choreography; ``MountRecipe()`` is the committed a80 mount.

Lengths are in meters, angles in radians unless the name ends in ``_deg``,
durations in seconds, and heights are measured along the wall. +y is the
robot's left.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MountRecipe:
  """Every number the a80 mount choreography uses."""

  # -- Sampling, wall and stance ------------------------------------------------
  fps: float = 50.0
  wall_angle_deg: float = 80.0
  """Wall angle from horizontal. The clip is solved in the tilted world and
  stored in the canonical vertical-wall frame."""
  standoff: float = 0.19
  """Standing distance from the wall's base line; toe clearance at the base binds it."""
  stance_width: float = 0.16
  """IK re-plants the standing feet at +-stance_width/2 (the walk keyframe stands at 0.296)."""
  stand_y: float = 0.0
  """Lateral station of the standing footprint off the anchor lane."""

  # -- Rungs ------------------------------------------------------------------------
  hand_z: float = 1.64
  """Final hand anchor height: the climb cycle's first frame, shifted along the wall."""
  hand_z_low: float = 1.34
  """The left hand's first rung, inside the standing reach."""
  low_rung_y: float = 0.0
  """Lateral offset of the left hand's first rung off its final lane."""
  hand_z_mid: float = 1.53
  """The right hand's first rung, reachable flat-footed."""
  mid_rung_y: float = 0.0
  """Lateral offset of the right hand's first rung."""
  hand_up_left_mid: float = 1.57
  """The middle rung of the left hand's two-step walk-up; each landing keeps
  elbow flexion in reserve."""
  hand_up_left_mid_y: float = 0.0
  """Lateral offset of that middle rung off the final lane."""

  # -- Segment durations ----------------------------------------------------------
  stand_s: float = 1.0
  """ground_stand."""
  raise_s: float = 1.6
  """hand_raise and hand_raise_right."""
  place_s: float = 1.8
  """hand_place (the left hand's first rung)."""
  set_left_s: float = 0.8
  """hand_set_left and hand_set_right."""
  place_right_s: float = 1.8
  """hand_place_right."""
  set_hold_s: float = 0.8
  """hand_set_hold."""
  foot_swing_right_s: float = 2.4
  """right_foot_mount, the braced step."""
  switch_hold_s: float = 0.5
  """foot_switch_hold."""
  foot_swing_s: float = 2.2
  """left_foot_mount, the toe pivot."""
  brace_hold_s: float = 0.6
  """brace_hold."""
  hand_up_s: float = 2.6
  """Each hand-up step."""
  up_hold_s: float = 0.4
  """hand_up_hold."""
  up_left_mid_hold_s: float = 0.4
  """hand_up_left_hold, the re-seat dwell between the left hand's two steps."""
  handoff_hold_s: float = 0.5
  """handoff_hold: the all-four dwell inside the playable region, a full
  engage run away from the last edge."""
  end_hold_s: float = 0.6
  """end_hold."""
  hand_load_transfer_s: float = 0.1
  """The all-four load-transfer dwell that opens each hand-up step."""

  # -- Arm raises ---------------------------------------------------------------------
  raise_roll: float = 0.8
  """Outboard shoulder roll held through the left arm's raise (0 = a pure front raise)."""
  raise_pitch_end: float = -1.8
  """Shoulder pitch the left arm's raise ends at (-1.8 holds the hand in front of the face)."""
  raise_yaw: float = 0.8
  """Mirrored shoulder-yaw excursion during each arm fold."""
  raise_roll_right: float = 1.2
  """Outboard shoulder roll of the right arm's raise."""
  raise_pitch_end_right: float = -2.4
  """Shoulder pitch the right arm's raise ends at."""
  raise_pitch_comp_sign: float = 1.0
  """Scale of the trunk-pitch compensation added to the right raise's end key."""

  # -- Hand placement ------------------------------------------------------------------
  place_bulge: float = 0.0
  """Outward bulge of the left hand's placement arc."""
  place_lean: float = 0.02
  """Forward base drift allowed during the left hand's placement."""
  place_bow: float = 0.0
  """Nose-down trunk pitch the base bows to while the left hand lands (0 = upright)."""
  touch_below: float = 0.0
  """How far below the low rung the left pad first touches before sliding up."""
  touch_frac: float = 0.5
  """Share of the placement arc spent reaching that touch point."""
  place_right_pitch: float = 0.1
  """Trunk pitch toward the wall reached over the right hand's placement."""
  place_right_base_dz: float = 0.02
  """Along-wall base rise over the right hand's placement (reach relief for its rung)."""
  place_right_base_dx: float = 0.0
  """Base shift toward the wall over the right hand's placement."""
  place_right_left_arm_w: float = 4.0
  """Posture weight of the welded left arm during the right hand's placement."""
  set_hold_press: float = 0.10
  """Base shift toward the wall during hand_set_hold, so the pivot starts closer."""
  set_hold_rise: float = 0.0
  """Along-wall base rise during hand_set_hold."""

  # -- Rakes: the seating stroke down the wall before a pad engages ----------------------
  rake_s: float = 0.5
  rake_dist: float = 0.004
  """Hand rakes."""
  rake_right_s: float = 0.4
  """The right foot's rake, longer to cover the weak window between toe-off and landing."""
  foot_rake_s: float = 0.3
  """The left foot's rake."""
  foot_rake_dist: float = 0.002
  """The stroke of both foot rakes."""

  # -- Right foot mount (the braced step) ------------------------------------------------
  foot_bulge: float = 0.04
  foot_lift: float = 0.1
  """Outward and upward offsets of the step arc's control point (ground clearance)."""
  step_base_w: tuple[float, float, float] = (100.0, 40.0, 60.0)
  """Base position task weights through the step."""
  step_press: float = 0.0
  """Mid-swing base press toward the wall through the step."""
  step_hips_lateral: float = 0.05
  """Lateral hip shift over the standing left foot."""
  step_hips_in: float = 0.06
  """Hip travel toward the wall base, which gives the welded hands slack."""
  step_rise: float = 0.01
  """Along-wall base rise through the step."""
  switch_lateral: float = -0.05
  """Lateral base shift during foot_switch_hold."""

  # -- Left foot mount (the toe pivot on three welds) ---------------------------------------
  pivot_out: float = -0.03
  """Base station change across the pivot, positive away from the wall."""
  pivot_rise: float = 0.0
  """Along-wall base rise through the pivot."""
  pivot_smear_gap: float = 0.008
  """The toe smear stops this far short of the wall base."""
  pivot_toe_release: float = 0.49
  """Pivot progress at which the planted toe pin starts fading (gone 0.17 later)."""
  pivot_ctl_up: float = 0.08
  pivot_ctl_out: float = 0.03
  """Lift and outward offset of the pivot arc's control point."""
  pivot_fold_u0: float = 0.15
  """Pivot progress at which the guided leg fold toward the climb stance starts."""
  swing_press: float = 0.04
  """Base press toward the wall through the pivot; shrinks the anchored hands'
  pull-out in the weak mid-swing window."""
  lateral_weight_shift: float = 0.03
  """Lateral base shift during the pivot."""

  # -- Hand-up steps ---------------------------------------------------------------------------
  swing_bulge: float = 0.04
  """Outward bulge of a hand-up arc."""
  reach_arm_pin: float = 6.0
  """Posture weight on the swinging arm, which pins its solution branch."""

  # -- Inverse kinematics and output -------------------------------------------------------------
  limit_margin_frac: float = 0.10
  limit_weight: float = 3.0
  """Joint-limit margin (share of the range) and its penalty weight."""
  weld_weight_scale: float = 4.0
  """Scale on every support-weld task weight (position 1000, rotation 500):
  above 1 trades base and press realization for less engaged-pad drift."""
  smooth_sigma: float = 1.5
  """Zero-phase Gaussian smoothing of the solved frames (in frames, 0 = off);
  the exact handoff frames pad the right edge, so the closure still lands."""
