"""Generate a PHYBot C2 wall-climb regrip cycle clip via whole-body IK.

The reference is produced directly on the robot's own kinematics (no mocap /
retargeting), like ``gen_squat_clip.py`` / ``gen_stand_clip.py``. One cycle
uses a four-beat crawl that remains feasible with a finite pad-moment limit:

  state 1   climb-task reset stance ("climb zero"): all four microspine pads
            engaged on the wall, arms overhead, legs in the frog splay.
  1 -> 2    LEFT-FOOT REGRIP: only the left foot releases and travels one
            step upward on an off-wall arc while the TOE of the articulated
            foot (planted against the wall during the preceding dwell)
            braces the wall as an ordinary compression contact; both hands
            and the right foot remain anchored. The released spine pad
            never touches the wall mid-swing — dragging unloaded spine tips
            across asperities damages them — so the toe supplies the
            support the sliding pad used to.
  2 -> 3    RIGHT-FOOT REGRIP: mirror of 1 -> 2 (the intervening four-pad
            dwell retracts the left toe and plants the right one). The base
            finishes rising one step.
  3 -> 4    LEFT-HAND REGRIP: the left hand releases, swings along an off-wall
            arc to one step height above its old anchor, and engages.
  4 -> 5    RIGHT-HAND REGRIP: mirror of 3 -> 4. State 5 equals state 1
            translated up by exactly one step height (enforced by a closure
            blend over the tail of the last swing), so cycles tile vertically
            (``--cycles``) and the clip loops as a gait reference. All dwells
            sit on four engaged anchors; only the swings move a limb.

During every free swing exactly one pad is released (the hand swings begin
with a short all-four load-transfer dwell). The remaining three always
include at least one hand and one foot, preserving the meter-scale
upper-tension/lower-compression couple that balances gravity's wall-normal
moment. The braced toe (a pair-only sphere at the ankle_roll foot tip, see
``PHYBOT_C2_FOOT_TOES``) stays planted at one world point through its whole
swing — the ankle absorbs the 30 mm shin rise — so it adds compression
margin with zero sliding anywhere. The old two-foot hop removed the large
couple and demanded that two nearly level hand pads carry the full
49--60 N*m as local bending.

Every landing ends with a pre-engage RAKE: the pad first touches the wall
``--rake-dist`` (default 2 mm) above its scheduled anchor and then drags that
distance straight down the wall surface over the last ``--rake-s`` of the
swing. Microspines need a small downward displacement under compression to
load their tips into asperities, so the reference bakes that seating stroke in
before the engaged flag flips at the segment boundary.

Engaged pads are 6-DoF weld tasks at their anchors (the sim's engaged
microspine is a weld: ``anchor_lock_rotation=True``); the 4-DoF arms cannot
satisfy two hand welds from a prescribed base pose, so the base is a soft task
and tilts/sinks the way the hanging robot physically would. Swing pads track
position exactly and orientation with a weight that ramps up toward landing
(a wristless arm has no spare DoF mid-swing).

Output extends the AMP/motion ``.npy`` schema (``joints_list``,
``joint_positions``, ``root_position``, ``root_quaternion`` xyzw,
``foot_sites_list``, ``foot_positions``, ``fps`` — accepted by
``scripts/view_motion.py`` and ``scripts/validation/check_amp_dataset.py``)
with climb-specific keys:

    ee_list               list[str]         4 spine bodies, microspine order
                                            (left/right hand, left/right foot)
    ee_positions          (T, 4, 3) float32 spine body origins, world
    ee_quaternions        (T, 4, 4) float32 spine body orientation, xyzw
    pad_positions         (T, 4, 3) float32 pad centers (contact-sphere grid
                                            center), world
    ee_engaged            (T, 4)    uint8   1 while the pad is anchored
    ee_anchor_positions   (T, 4, 3) float32 pad-center anchor the weld holds
                                            (engaged) or the upcoming target
                                            anchor (swinging)
    toe_braced            (T, 2)    uint8   1 inside a foot swing's full-press
                                            toe-brace window (toe_list order)
    toe_list              list[str]         foot EEs owning a bracing toe
    phase_segment         (T,)      uint8   index into segment_names
    segment_names         list[str]
    cycle_index           (T,)      uint16
    step_height           float32           per-cycle rise of every anchor
    wall_surface_x        float32           wall plane
    pad_touch_x           float32           wall coordinate of contact center C

Validate/preview:
    python scripts/view_motion.py \
      src/scr_phybot_mjab/assets/motions/climb_cycle_moment_safe.npy
    python scripts/validation/check_amp_dataset.py --data-dir <output dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import mjlab  # noqa: F401  # Initialize task plugins before project imports.

from scr_phybot_mjab.tools.climbing_reference.cycle import generate_clip
from scr_phybot_mjab.tools.paths import REPOSITORY_ROOT as _REPO


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output",
    type=Path,
    default=(_REPO / "src/scr_phybot_mjab/assets/motions/climb_cycle_moment_safe.npy"),
  )
  parser.add_argument(
    "--step-height",
    type=float,
    default=None,
    help="per-cycle anchor rise (m); default: the climb task's step_height",
  )
  parser.add_argument("--cycles", type=int, default=1, help="number of stacked cycles")
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument(
    "--foot-swing-s",
    type=float,
    default=0.9,
    help="duration of each single-foot regrip; the other three pads stay engaged",
  )
  parser.add_argument("--swing-s", type=float, default=1.2, help="single hand regrip duration")
  parser.add_argument("--hold-start-s", type=float, default=0.5)
  parser.add_argument(
    "--foot-transfer-hold-s",
    type=float,
    default=0.4,
    help="four-pad dwell between the foot regrips; retracts the left toe "
    "and plants the right one, so keep the ankle sweep under ~4 rad/s",
  )
  parser.add_argument("--seat-hold-s", type=float, default=0.3)
  parser.add_argument("--regrip-hold-s", type=float, default=0.3)
  parser.add_argument("--end-hold-s", type=float, default=0.5)
  parser.add_argument(
    "--swing-bulge",
    type=float,
    default=0.04,
    help="hand swing off-wall Bezier control offset (m); peak clearance is half",
  )
  parser.add_argument(
    "--swing-lateral",
    type=float,
    default=0.0,
    help="hand swing outward Bezier control offset (m); peak offset is half "
    "(default 0: lateral excursion buys nothing and abducts the shoulder)",
  )
  parser.add_argument(
    "--foot-bulge",
    type=float,
    default=0.03,
    help="foot swing off-wall Bezier control offset (m; peak clearance is "
    "half): the released spine pad must clear the wall — the planted toe, "
    "not a sliding pad, braces the wall during foot regrips",
  )
  parser.add_argument(
    "--body-press",
    type=float,
    default=0.03,
    help="base shift toward the wall during the three-pad transfer beats (m)",
  )
  parser.add_argument(
    "--lateral-weight-shift",
    type=float,
    default=0.025,
    help="base shift toward the opposite-side support before each regrip (m)",
  )
  parser.add_argument(
    "--hand-load-transfer-s",
    type=float,
    default=0.1,
    help="all-four-anchor preload dwell at the start of each hand transfer (s)",
  )
  parser.add_argument(
    "--rake-s",
    type=float,
    default=0.1,
    help="pre-engage rake duration at the end of every swing (s)",
  )
  parser.add_argument(
    "--rake-dist",
    type=float,
    default=0.002,
    help="downward on-wall seating stroke before engage (m)",
  )
  parser.add_argument(
    "--limit-margin-frac",
    type=float,
    default=0.10,
    help="joint-limit avoidance zone as a fraction of each joint's range; "
    "inside it a spring pulls the joint back toward the interior",
  )
  parser.add_argument(
    "--limit-weight",
    type=float,
    default=3.0,
    help="IK weight of the limit-avoidance spring (posture weight is 0.3)",
  )
  args = parser.parse_args()

  generate_clip(
    output=args.output,
    step_height=args.step_height,
    cycles=args.cycles,
    fps=args.fps,
    foot_swing_s=args.foot_swing_s,
    swing_s=args.swing_s,
    hold_start_s=args.hold_start_s,
    foot_transfer_hold_s=args.foot_transfer_hold_s,
    seat_hold_s=args.seat_hold_s,
    regrip_hold_s=args.regrip_hold_s,
    end_hold_s=args.end_hold_s,
    swing_bulge=args.swing_bulge,
    swing_lateral=args.swing_lateral,
    foot_bulge=args.foot_bulge,
    body_press=args.body_press,
    lateral_weight_shift=args.lateral_weight_shift,
    hand_load_transfer_s=args.hand_load_transfer_s,
    rake_s=args.rake_s,
    rake_dist=args.rake_dist,
    limit_margin_frac=args.limit_margin_frac,
    limit_weight=args.limit_weight,
  )


if __name__ == "__main__":
  main()
