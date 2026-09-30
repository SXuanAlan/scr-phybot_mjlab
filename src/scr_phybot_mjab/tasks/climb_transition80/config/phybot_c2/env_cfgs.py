"""PHYBot C2 80-degree transition environment (BeyondMimic-minimal, termination-first).

Built by calling the climb builder and mutating the result — scene, sim,
s2r DR package and actions are inherited verbatim. Everything else is
REPLACED wholesale and lives in this package (nothing is imported from
climb_transition):

  * PD table written out explicitly = the real robot's motor-firmware table
    (legs 100/10, ankles 50/5, waist 20/10, arms 50/5);
  * observation contract written out explicitly (v2, 2026-08-21): actor 238
    = climb proprio + BeyondMimic reference lanes + pad faces + wall face
    corners, critic 419 = actor + mjlab tracking body lanes + privileged
    microspine lanes (``TRANSITION80_ACTOR_TERMS`` / ``_CRITIC_TAIL``);
  * rewards = BeyondMimic's exp-kernel tracking terms (+ a pad-only position
    term) + ONE staged engage-progress potential + two regularisers;
  * terminations = tracking deviation (with persistence), the LadderMan
    contact law, scheduled-anchor rip, body-through-wall, and the engage
    constraint cone in Constraints-as-Terminations mode (5-step persistence,
    seed-settle exemption);
  * curricula = wall pinned at 80 deg, the weld-shear ignition ladder, and
    threshold anneals on the terminations — births are sampled adaptively.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import functools
from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_ACTUATOR_ANKLE_PITCH,
  PHYBOT_C2_ACTUATOR_ANKLE_ROLL,
  PHYBOT_C2_ACTUATOR_ELBOW_PITCH,
  PHYBOT_C2_ACTUATOR_HIP_PITCH,
  PHYBOT_C2_ACTUATOR_HIP_ROLL,
  PHYBOT_C2_ACTUATOR_HIP_YAW,
  PHYBOT_C2_ACTUATOR_KNEE,
  PHYBOT_C2_ACTUATOR_NECK,
  PHYBOT_C2_ACTUATOR_SHOULDER_PITCH,
  PHYBOT_C2_ACTUATOR_SHOULDER_ROLL,
  PHYBOT_C2_ACTUATOR_SHOULDER_YAW,
  PHYBOT_C2_ACTUATOR_WAIST_YAW,
  PHYBOT_C2_STANDING_JOINT_POS,
  configure_phybot_c2_floor_geom,
)
from scr_phybot_mjab.tasks.climb_transition80 import mdp
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  _policy_joint_cfg,
  phybot_c2_climb_wall_env_cfg,
)

_STEPS_PER_ITER = 96  # climb PPO num_steps_per_env (test-guarded)
_MOTIONS_DIR = Path(__file__).resolve().parents[4] / "assets" / "motions"

TRANSITION80_WALL_ANGLE_DEG = 80.0


def mount_clip_path(angle_deg: float) -> Path:
  """Native-angle mount prologue asset (gen_mount_clip --wall-angle-deg)."""
  return _MOTIONS_DIR / f"climb_mount_prologue_a{int(round(angle_deg))}.npy"


TRANSITION80_MOTION_FILE: Path = mount_clip_path(TRANSITION80_WALL_ANGLE_DEG)

# Blind-delivery spawn scatter (walk hands the robot over with tape-mark
# accuracy): +-wall-distance, +-lateral (m), +-yaw (rad, 5 deg). Standing
# frames only; the reference and its anchors stay canonical.
TRANSITION80_SPAWN_XY_NOISE = (0.05, 0.03)
TRANSITION80_SPAWN_YAW_NOISE = 0.0873
# Cosmetic standing base x for the a80 clip (standoff 0.19 from the base
# line at x = 1.5; RSI overwrites on every reset).
TRANSITION80_STANDING_X = 1.31

# v57 seed settle guard: the 1 g birth transient rips young seeded welds
# within the stock 2-step readout guard; no gravity ladder here, so the
# guard alone must outwait the settle (0.12 s). Seeded RSI welds only.
# v3 (2026-08-22): SEEDED-BIRTH FREEZE — the reference clock holds on the
# birth frame for 25 steps (0.5 s) so the settle transient is absorbed with
# the reference standing still (user design; passively validated with a
# matching judge guard: seed loss at the pivot births 100 % -> 14 %, 790
# 96 % -> 6 %). See TRANSITION80_SEED_GUARD_STEPS for why the judge guard
# does NOT span the freeze in training.
TRANSITION80_BIRTH_FREEZE_STEPS = 25
# v3b (2026-08-22 02:0x): the judge guard is DECOUPLED from the freeze again.
# With the guard spanning the whole 25-step freeze the resumed policy leaned
# on unbreakable seeds (scheduled_hold_margin -4..-5.5) and every seed died
# on the first judged step (616/616 rips at step 26, rip share 91 %, even
# births that never ripped in v2). The freeze keeps the reference still; the
# seeds are judged from step 7 as in v2 (the 0.2 s settle exemption covers
# the transient rips), so overloading a seed costs the policy from step 11.
TRANSITION80_SEED_GUARD_STEPS = 6
# v3 TENABLE-BIRTH MASK: frame ranges (end exclusive) where a passive hold
# loses >= 20 % of its seeds even after the guard (seed_diag map, cross v3,
# 2026-08-22): right_foot_mount 270-370 (LH pulled into tension/lateral while
# the right foot lifts) and the end of hand_place_right 500-540 (RF toe in
# reverse shear + LH ultimate during the RH reach). RSI never births there;
# the phases stay reachable by playing through. REGENERATE after any clip
# re-choreography (scripts/transition80_birth_mask.py).
# cross v3r6 (2026-08-22 14:0x, place-right-s 3.2 -> the reach ends 20 frames
# later; hand rakes 0.5 s): right_foot_mount 270-379, the end of
# hand_place_right 510-569, and one mid-hand_up_left frame 790-799 (LF
# lateral under a passive hold, 95 %; the compressed swing arc). Map:
# logs/clip_drafts/validator_logs/a80_cross_v3r6_passive_map.txt.
# v4b (2026-08-22 13:xx, user decision): MASK OFF. The passively untenable
# births (right_foot_mount 270-379, the tiptoe reach 510-569, frame 790) are
# the policy's to learn with active loading (v1g: rip 18 % -> 3 % in ~100
# iter; v3b: 31 % -> 7 %). The mask also killed adaptive bin 6 outright and
# two of the five ignition-classroom rake windows ((355,380), (540,565):
# pool 125 -> 75 frames). Mechanism kept (birth_mask_from_ranges); re-arm at
# launch with the v3r6 map
#   --env.commands.climb-motion.birth-exclude-frames 270 380 510 570 790 800
# (the flag with no values is ()).
TRANSITION80_BIRTH_EXCLUDE_FRAMES: tuple[tuple[int, int], ...] = ()
# v4 IGNITION CLASSROOM (user decision 2026-08-22): this share of births lands
# in the pre-landing rake windows [edge - rake_frames, edge) of every engage
# rising edge — pad at the wall, reference dragging it down, weld one
# press-and-hold away. The v3b-v3f probes put the hands 3 cm off the wall on
# 98.7 % of window steps; adaptive RSI (1 s bins, failure-weighted) never
# targets the 0.3-0.5 s rakes. Rake frames = round(rake_s * fps): cross v3
# 0.3 s -> 15; cross v4 0.5 s -> 25 (update with the clip).
# v6e (2026-08-30, user decision): 0.3 -> 0.45, and the classroom windows are
# EXPLICIT now -- every ignition rake window (the old auto set, recomputed
# from the clip at config-build time) PLUS the first 30 frames of each
# closing hand step, so hand-up births start with >= 1.2 s of runway before
# the landing deadline instead of dying at the window's tail.
TRANSITION80_CLASSROOM_FRACTION = 0.45
TRANSITION80_CLASSROOM_HANDUP_WINDOW_FRAMES = 30


def _transition80_classroom_windows(motion_file: str) -> tuple[tuple[int, int], ...]:
  data = np.load(motion_file, allow_pickle=True).item()
  windows = list(
    mdp.rake_windows_from_engaged(
      np.asarray(data["ee_engaged"]), TRANSITION80_CLASSROOM_RAKE_FRAMES
    )
  )
  names = list(data["segment_names"])
  labels = np.asarray(data["phase_segment"])
  segs = ["hand_up_right", "hand_up_left"]
  if "hand_up_left_mid" in names and np.any(labels == names.index("hand_up_left_mid")):
    segs.append("hand_up_left_mid")  # v6g two-step walk-up
  for seg in segs:
    frames = np.flatnonzero(labels == names.index(seg))
    windows.append(
      (int(frames[0]), int(frames[0]) + TRANSITION80_CLASSROOM_HANDUP_WINDOW_FRAMES)
    )
  return tuple(sorted(windows))
# v5e (2026-08-23, user decision: tighten the adaptive sampler, guard stays
# 6): the failure-weighted start bins fed on the closing hand-up windows --
# births there die in 0.2-0.5 s (seeds torn at the first judged step, head
# into the fence), each death re-feeds the histogram, and the replica probe
# on v5d model_4025 found 55 % of all births in bins 17-20 (f 825-971).
# Uniform floor 0.1 -> 0.5 and per-bin cap 0.35 -> 0.08 (~1.7x uniform over
# 21 bins): the four hard bins can take at most ~32 % between them.
TRANSITION80_ADAPTIVE_UNIFORM_RATIO = 0.5
TRANSITION80_ADAPTIVE_MAX_BIN_PROB = 0.08
TRANSITION80_CLASSROOM_RAKE_FRAMES = 25  # cross v3r6: hand rakes 0.5 s (feet 0.3-0.4 s -> window starts in the approach)
# Birth-settle exemption shared by the rip law and the engage cone. 0.2 s
# (10 steps) covers the step-2 landing impulse and the 6-step seed guard;
# the 0.5 s window of v1c-v1f was a loophole — model_1550 window probes
# showed the policy tearing every seeded anchor off at 0.15-0.3 s for free
# and pantomiming the rest of the choreography without contact. Rips from
# step 11 on (the hold-still probe's compliance-driven 5-19 step rips) are
# now the policy's responsibility (v57: learnable with active loading).
# v5b (2026-08-23, user decision): 0.2 -> 0.12 s = the 6-step judge guard.
# The 4 judged-but-exempt steps (7-10) were a FREE window: the replica probe
# on v5 model_1575 found 83 % of seeds shed before step 9 (median step 5),
# feet occupancy 0.02, self-weld rate decaying 0.06 -> 0.005 -- the policy
# dumps its seeds for free and keeps one hand. With the rip now a bill (not
# a death) there is no gradient desert to protect against: a seed lost at
# step 7+ pays rent until it is re-welded.
TRANSITION80_SEED_SETTLE_EXEMPT_S = 0.12
# v5 RIP BILL (user decision 2026-08-23, climb's rule): a scheduled anchor
# torn off is a per-step PENALTY until re-welded, not a death (the
# ``anchor_rip`` termination is retired; the contact law's grace stays as
# the backstop). Per flagged pad, per second; piecewise-linear ramp. Sized
# under the per-pad tracking income (~8.5/s total) so the bill never makes
# the early death the optimum (08-13 lesson).
TRANSITION80_RIP_PENALTY_STAGES = [
  {"iter": 0, "value": 0.5},
  {"iter": 1000, "value": 1.5},
  {"iter": 2500, "value": 3.0},
]
# v5 GIMBAL TRAVEL LADDER (user design 2026-08-23): the pad gimbals' stops
# open at +-15 deg and anneal piecewise-linearly to the +-3 deg real stop
# (PHYBOT_C2_PAD_GIMBAL_TRAVEL_RAD) -- angle compliance as a curriculum,
# climb's tilt-peel anneal re-done on the real joint. Written into the
# per-world jnt_range by an event (startup + every reset).
# v5d (2026-08-23, user decision): STAGED slow anneal, the 3-deg tightening
# POSTPONED -- "learn at 15 first". The v5c ladder (linear 300->3000) hit the
# 3-deg stop while the hand-up windows were still passively untenable there
# (the 29/80 gate list): every birth in 800-885/905-995 died on the wall
# fence within 0.2-0.4 s (head-first pitch after the seeds tore, fence
# forensics: neck 52 %). Steps 15 -> 8 -> 3 with short 500-iter ramps
# between holds; push the boundaries later if 15 is not yet learned.
# v6b (2026-08-24, user decision): the 15-deg hold runs to iter 10000 (the
# run's whole max-iterations budget) -- learn everything at 15 deg first;
# the tightening stages beyond are placeholders past the horizon.
TRANSITION80_GIMBAL_TRAVEL_STAGES = [
  # v6d (2026-08-25, user decision after the 8-deg record run and the first
  # full from-stand chain): 8 -> 5 now, 6000 iters at 5, then 3 for 10000
  # (run ends at 26000). 200-iter ramps at each boundary; every OTHER
  # parked tightening stays beyond the horizon (30000/32000) -- those
  # advance by hand on graduation evidence, only the gimbal moves here.
  {"iter": 0, "value": 15.0},
  {"iter": 7900, "value": 15.0},
  {"iter": 8100, "value": 8.0},
  {"iter": 10000, "value": 8.0},
  {"iter": 10200, "value": 5.0},
  {"iter": 16000, "value": 5.0},
  {"iter": 16200, "value": 3.0},
]

# Full-body wall fence: every link that never legitimately reaches the wall
# (the pads, forearms/elbows and feet/toes stay out). Geom SURFACES are
# judged, not link origins (08-13 lesson). Roster = the 08-13/08-14
# measured reference envelope (body links >= +13.3 cm, upper arms +8.4 cm
# through stand/raise/place; a80 cross v3 keeps >= +10.8 mm on every frame).
# v6 dare (2026-08-24): the four shoulder/upper-arm geoms LEFT the fence --
# arms get ORDINARY physical wall contact (TRANSITION80_ARM_WALL_GEOMS pairs)
# and may lean on the wall; head/torso/base/hips/knees stay lethal.
TRANSITION80_FENCE_GEOMS = (
  "neck_pitch_collision",
  "torso_collision",
  "base_link_collision",
  "left_hip_roll_collision",
  "right_hip_roll_collision",
  "left_hip_yaw_collision",
  "right_hip_yaw_collision",
  "left_knee_collision",
  "right_knee_collision",
)
TRANSITION80_ARM_WALL_GEOMS = (
  "left_shoulder_roll_collision",
  "right_shoulder_roll_collision",
  "left_shoulder_yaw_collision",
  "right_shoulder_yaw_collision",
  "left_elbow_pitch_collision",
  "right_elbow_pitch_collision",
)

# Touch+drag weld ignition ladder (the only plant schedule kept): easy ->
# honest spec. Iters are ceilings; drive by the engage lane.
# v3d (2026-08-22 04:0x): first rung 4 -> 2 N. The v3b/v3c ignition probes
# (model_1150/1350) found the hands' rare touches carrying 0-symmetric shear
# (p50 0 N, 13-15 % of touching steps >= 4 N) with at most 2 consecutive
# qualifying steps: the 4 N + 2 cm/s ignition was out of reach of the
# contact style the policy has; the ladder still anneals to the honest spec.
# v4 (2026-08-22): every second/third stage moved to 4500/6500 so the loosened
# ignition regime (cone p 0.05, 2 N, 0.05 m/s, 3 s grace) outlives the first
# self-welds after the warm start at iter 1175 (the v3f window closed at 3000
# before the feather touches turned into loads).
# v6f (2026-08-31): parked stages pushed 30000/32000 -> 50000/52000 after
# they AUTO-FIRED mid-v6e (the budget grew past the old horizon and nobody
# moved the parking lot -- the third stage-cliff incident; rolled back to
# model_29975 on the user's call). RULE: whenever max-iterations grows,
# re-check every parked stage sits beyond the new horizon.
# v5g (2026-08-24, user decision): LEARN THE LENIENT REGIME FIRST -- 2 N
# ignition, 8-deg gimbal, 3 s grace hold until the skills are demonstrably
# learned; the tightening stages are parked at 14000/15500 as placeholders
# and will be advanced BY HAND on graduation evidence (eng_hz sustained,
# closs falling, the from-start chain welding), not by the clock. The v5f
# 7500/9500 stages re-killed ignition (7-10 N with eng_hz 0.0002) and cut
# the grace to 1-2 s, turning training into hold-until-executed.
# v5f (2026-08-23): every 4500/6500 stage shifted +3000 (-> 7500/9500). The
# v4 clock assumed self-welds by iter 4500; at 4512 the run had eng_hz 0.005
# and the simultaneous switch (cone p 0.05 -> 0.5, ignition 2 -> 7 N,
# grace 3 -> 2 s, radius 0.15 -> 0.10, deviation executioners tightened)
# halved eplen (137 -> 80) and made the cone the top killer (4 -> 35 %).
TRANSITION80_SHEAR_LADDER = [
  {"iter": 0, "value": 2.0},
  {"iter": 50000, "value": 7.0},
  {"iter": 52000, "value": 10.0},
]
# v3d: anti-rake ignition ladder — the weld engine requires the pad to drift
# down-wall no faster than this while loading (m/s). 2 cm/s was met on only
# 58 % of touching steps (touch v_tan p50 0.24-0.31 m/s); 5 cm/s opens the
# gate to the current touches, then anneals back to the spec with the shear
# ladder's second rung.
TRANSITION80_ANTIRAKE_LADDER = [
  {"iter": 0, "value": 0.05},
  {"iter": 50000, "value": 0.02},
]
# Contact law (LadderMan executioner): grace after a scheduled edge, and the
# radius within which a weld satisfies the schedule. Phi's anchor radius
# anneals in lockstep.
# v6 dare (2026-08-24, user decision): 1 s FLAT. LadderMan's discipline (its
# Table 5 uses 30 sim steps ~ 0.6 s): a scheduled contact missing for one
# second ends the episode -- hovering out the old 3 s for tracking pay is no
# longer a strategy. Forensics on v5g model_9975: hover kept 98 % of the two
# big position terms and the approach was dithered out over 88-106 steps.
TRANSITION80_CONTACT_GRACE_STAGES = [
  {"iter": 0, "value": 1.0},
]
TRANSITION80_ANCHOR_RADIUS_STAGES = [
  {"iter": 0, "value": 0.15},
  {"iter": 50000, "value": 0.10},
  {"iter": 52000, "value": 0.08},
]
# Tracking-deviation executioners (persist 10 steps = 0.2 s).
TRANSITION80_ANCHOR_POS_STAGES = [{"iter": 0, "value": 0.35}, {"iter": 50000, "value": 0.25}]
TRANSITION80_PAD_POS_STAGES = [{"iter": 0, "value": 0.30}, {"iter": 50000, "value": 0.20}]
# Engage constraint cone limits (start -> graduation) and soft probabilities.
TRANSITION80_TILT_STAGES = [
  {"iter": 0, "value": 15.0},
  {"iter": 50000, "value": 10.0},
  {"iter": 52000, "value": 6.0},
]
TRANSITION80_GRIND_STAGES = [
  {"iter": 0, "value": 0.30},
  {"iter": 50000, "value": 0.20},
  {"iter": 52000, "value": 0.12},
]
TRANSITION80_SLAM_STAGES = [
  {"iter": 0, "value": 0.60},
  {"iter": 50000, "value": 0.45},
  {"iter": 52000, "value": 0.30},
]
TRANSITION80_FORCE_STAGES = [
  {"iter": 0, "value": 80.0},
  {"iter": 50000, "value": 60.0},
  {"iter": 52000, "value": 40.0},
]
# v3f (2026-08-22 05:4x): the cone's termination PROBABILITY now starts near
# zero and anneals up. Four probes (v3b-v3e) showed a learned avoidance: the
# hands park 3 cm off the wall on 98.7 % of their window steps because every
# early touch was lethal (CaT p 0.5 from iteration 0) before any press/hold
# skill existed; reward-side fixes (Phi x4, ignition ladder, approach+press
# lanes) did not move the touch rate and sigma climbed 0.59 -> 0.77 (no new
# gradient). With the cone nearly off inside the windows, noise-driven
# touches survive and get paid; the cone tightens once ignition exists.
TRANSITION80_SOFT_P_STAGES = [
  {"iter": 0, "value": 0.05},
  {"iter": 50000, "value": 0.5},
  {"iter": 52000, "value": 1.0},
]
# action_rate_l2 weight for the ignition period (climb parity is -0.1).
# v5b (2026-08-23): -0.02 -> -0.05 (halfway to climb's -0.1) -- the model_2500
# window videos jitter and the replica probe reads action_saturation 0.53;
# the chain now trains (eplen 190 steps), so the smoothness gradient can
# be raised off its ignition-period floor.
TRANSITION80_ACTION_RATE_WEIGHT = -0.05
# Climb's schedule-adherence discount on pose income (v1e, user call): pose
# kernels pay ``floor`` of their value while any pad's latch state disagrees
# with the chart, ramped over ``grace_fraction`` of the violated window near
# its edges. Velocity kernels and Phi are not gated (climb parity).
# v6 dare: 0.25 -> 0.10 (hover keeps one tenth of the gated tracking pay).
TRANSITION80_ENGAGE_GATE_FLOOR = 0.10
# v1g: a FIXED 0.5 s ramp. The climb-parity grace_fraction 0.3 of a 10 s
# window was a 3 s ramp — the gate still read 0.83 when the contact law
# executed at edge + 3 s (model_1550 trace); the discount never bit.
TRANSITION80_ENGAGE_GATE_GRACE_S = 0.5
# Phi reach radius: the hand parked 46 cm from its anchor, outside the 0.25 m
# reach stage (Phi = 0 there); 0.5 m gives a slope from where it stands.
TRANSITION80_PHI_REACH_RADIUS_M = 0.5
# v3c (2026-08-22 03:0x): Phi weight 0.5 -> 2.0 (queued candidate, user-seen).
# v3b model_1150 ignition probe = a HOVER TRAP: in its windows the LH pad
# parks 3-4 cm off the wall (98.6 % of in-window steps never touch; cone
# grind/slam/force bills all zero) and eats the pose income until the 3 s
# contact-law grace executes it (closs 87 %). Phi is the only term that
# prices touch (0.5+) over hover (<= 0.3) over weld (0.9+); at 0.5 the touch
# premium was ~2 % of the per-step income, at 2.0 it is 8-20 %.
# v6 dare: 2.0 -> 3.0, and the share mix moves toward approach/press (the
# transition itself was underpaid: forensics diff only +17 % vs welded).
TRANSITION80_PHI_WEIGHT = 3.0
# v3e (2026-08-22 04:4x): Phi shape for the hover trap — pay the last 10 cm of
# wall-normal approach and the press itself (see mdp.rewards.engage_progress).
# reach 0.3 + approach 0.15 < touch 0.5; touch 0.5 + load 0.2 + press 0.2 = weld 0.9.
TRANSITION80_PHI_APPROACH_SHARE = 0.19
TRANSITION80_PHI_REACH_SHARE = 0.25  # v6: 0.30 -> 0.25 (reach+approach 0.44 < touch_level)
TRANSITION80_PHI_TOUCH_LEVEL = 0.45  # v6: 0.50 -> 0.45 (touch+load+press = 0.90 = weld_level)
TRANSITION80_PHI_APPROACH_RADIUS_M = 0.10
TRANSITION80_PHI_PRESS_SHARE = 0.35
TRANSITION80_PHI_PRESS_REF_N = 20.0
TRANSITION80_PHI_LOAD_SHARE = 0.10
# v3 hold-margin dwell wage (user call 2026-08-22, "方案 A"): scheduled AND
# welded pads are paid clip(hold_margin / 0.3, 0, 1) averaged over the
# scheduled pads. Weight 0.5 (x dt -> <= 0.01/step, ~10 % of the tracking
# income): a load-sharing preference, never rent, never a reason not to weld.
TRANSITION80_HOLD_MARGIN_WAGE_WEIGHT = 0.5
TRANSITION80_HOLD_MARGIN_FULL = 0.3

# Joint PD table = the real robot's motor-firmware table
# (deploy/phybot_deploy_c2/MotorList/config/phybot_mini_{1,2}.yaml PDKp/PDKd;
# identical to the shipped walk-AMP and squat tasks). Differences from the
# climb pd100 table: waist 100 -> 20 and the eight arm joints 100/10 -> 50/5
# (user call 2026-08-21: the 08-01 A0 hang bench showed deploy gains hold the
# hang with 4.9 mm sag, and kp=100 arms turn σ=0.6 exploration noise into
# ~1 m/s pad grinds and tens-of-N slams that the engage cone then kills).
# Every entry is a fresh dataclasses.replace copy: the actuator cfgs are
# module singletons shared with walk/stand/climb and must never be mutated
# in place (07-12 rule).
TRANSITION80_ARM_PD = (50.0, 5.0)
TRANSITION80_WAIST_PD = (20.0, 10.0)
TRANSITION80_LEG_PD = (100.0, 10.0)
TRANSITION80_ANKLE_PD = (50.0, 5.0)


def transition80_actuators(
  arm_pd: tuple[float, float] = TRANSITION80_ARM_PD,
  waist_pd: tuple[float, float] = TRANSITION80_WAIST_PD,
  leg_pd: tuple[float, float] = TRANSITION80_LEG_PD,
  ankle_pd: tuple[float, float] = TRANSITION80_ANKLE_PD,
) -> tuple:
  """The explicit 12-actuator table (fresh copies, fixed asset order)."""

  def _pd(actuator, pd):
    return dataclasses.replace(actuator, stiffness=pd[0], damping=pd[1])

  return (
    _pd(PHYBOT_C2_ACTUATOR_HIP_PITCH, leg_pd),
    _pd(PHYBOT_C2_ACTUATOR_HIP_ROLL, leg_pd),
    _pd(PHYBOT_C2_ACTUATOR_HIP_YAW, leg_pd),
    _pd(PHYBOT_C2_ACTUATOR_KNEE, leg_pd),
    _pd(PHYBOT_C2_ACTUATOR_ANKLE_PITCH, ankle_pd),
    _pd(PHYBOT_C2_ACTUATOR_ANKLE_ROLL, ankle_pd),
    _pd(PHYBOT_C2_ACTUATOR_WAIST_YAW, waist_pd),
    _pd(PHYBOT_C2_ACTUATOR_SHOULDER_PITCH, arm_pd),
    _pd(PHYBOT_C2_ACTUATOR_SHOULDER_ROLL, arm_pd),
    _pd(PHYBOT_C2_ACTUATOR_SHOULDER_YAW, arm_pd),
    _pd(PHYBOT_C2_ACTUATOR_ELBOW_PITCH, arm_pd),
    PHYBOT_C2_ACTUATOR_NECK,  # non-policy joint, asset default 20/2
  )


def _stage0(stages: list[dict]) -> float:
  return float(sorted(stages, key=lambda s: s["iter"])[0]["value"])


def _stage_final(stages: list[dict]) -> float:
  return float(sorted(stages, key=lambda s: s["iter"])[-1]["value"])


def _floor_spec_fn(spec: mujoco.MjSpec, base_fn) -> None:
  """The climb scene spec plus a contact-enabled floor (the climb builder's
  spec_fn disables the terrain plane; the later write wins)."""
  base_fn(spec)
  configure_phybot_c2_floor_geom(spec, enable_contact=True)


def _termination_anneal(
  termination_name: str, param_name: str, stages: list[dict]
) -> CurriculumTermCfg:
  return CurriculumTermCfg(
    func=mdp.termination_param_schedule,
    params={
      "termination_name": termination_name,
      "param_name": param_name,
      "stages": [dict(s) for s in stages],
      "steps_per_iter": _STEPS_PER_ITER,
    },
  )

##
# Observation contract (v2, 2026-08-21) — written out in one place.
#
# Retired from the inherited climb contract (all audited dead or redundant on
# the a80 prologue at a pinned 80-degree wall): ``height_above_spawn`` (an
# affine readout of root_pose_world), ``velocity_command`` (constant zero —
# the climb_velocity command is dropped with it), ``microspine_disengage_time``
# (a saturating copy of not-engaged), ``microspine_cooldown`` (non-zero only
# for 0.2 s after a rare non-terminating forced release), ``wall_base_position``
# (a constant; the face corners replace it), and six constant lanes of the
# 16-lane motion_command chart (now 10, see mdp.motion).
##
TRANSITION80_REMOVED_OBS = (
  "height_above_spawn",
  "velocity_command",
  "microspine_disengage_time",
  "microspine_cooldown",
  "wall_base_position",
)
TRANSITION80_ACTOR_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "projected_gravity",
  "joint_pos_rel",
  "joint_vel_rel",
  "last_action",
  "prev_action",
  "prev_prev_action",
  "microspine_engaged_state",
  "microspine_contact_fraction",
  "motion_command",
  "motion_ref_joint_pos_rel",
  "motion_ref_joint_vel",
  "motion_anchor_pos_b",
  "motion_anchor_ori_b",
  "motion_pad_center_relative",
  "motion_anchor_relative",
  "microspine_pad_center_world",
  "root_pose_world",
  "wall_face_corners",
)
TRANSITION80_CRITIC_TAIL = (
  "robot_body_pos_b",
  "robot_body_ori_b",
  "microspine_hold_margin",
  "microspine_anchor_load",
  "microspine_forced_release",
  "microspine_pad_normal_force",
  "microspine_pad_shear_down",
)
TRANSITION80_OBS_DIMS = {
  "base_lin_vel": 3,
  "base_ang_vel": 3,
  "projected_gravity": 3,
  "joint_pos_rel": 21,
  "joint_vel_rel": 21,
  "last_action": 21,
  "prev_action": 21,
  "prev_prev_action": 21,
  "microspine_engaged_state": 4,
  "microspine_contact_fraction": 4,
  "motion_command": 10,
  "motion_ref_joint_pos_rel": 21,
  "motion_ref_joint_vel": 21,
  "motion_anchor_pos_b": 3,
  "motion_anchor_ori_b": 6,
  "motion_pad_center_relative": 12,
  "motion_anchor_relative": 12,
  "microspine_pad_center_world": 12,
  "root_pose_world": 7,
  "wall_face_corners": 12,
  "robot_body_pos_b": 51,
  "robot_body_ori_b": 102,
  "microspine_hold_margin": 4,
  "microspine_anchor_load": 12,
  "microspine_forced_release": 4,
  "microspine_pad_normal_force": 4,
  "microspine_pad_shear_down": 4,
}
TRANSITION80_ACTOR_DIM = sum(TRANSITION80_OBS_DIMS[n] for n in TRANSITION80_ACTOR_TERMS)  # 238
TRANSITION80_CRITIC_DIM = TRANSITION80_ACTOR_DIM + sum(
  TRANSITION80_OBS_DIMS[n] for n in TRANSITION80_CRITIC_TAIL
)  # 419

_INHERITED_ACTOR_TERMS = TRANSITION80_ACTOR_TERMS[:11]  # .. motion_command
_INHERITED_CRITIC_TERMS = (
  "microspine_hold_margin",
  "microspine_anchor_load",
  "microspine_forced_release",
)


def transition80_observation_groups(
  cfg: ManagerBasedRlEnvCfg,
  *,
  motion_command: str = "climb_motion",
  affordance_command: str = "microspine_affordance",
) -> None:
  """Rebuild the actor and critic groups of a climb-built cfg into the v2
  contract, in ``TRANSITION80_ACTOR_TERMS`` (+ ``_CRITIC_TAIL``) order.

  The inherited lanes are the climb builder's own term objects (their DR
  ``Unoise`` and params come along); every new lane is a fresh term with no
  noise. The actor group is edited IN PLACE so its ``enable_corruption``
  (True under DR, False in play) survives; the critic group is replaced by
  noise-free copies of the actor lanes plus the tail."""
  actor_group = cfg.observations["actor"]
  old_actor = actor_group.terms
  old_critic = cfg.observations["critic"].terms
  actor: dict[str, ObservationTermCfg] = {
    name: old_actor[name] for name in _INHERITED_ACTOR_TERMS
  }
  motion_lanes = (
    ("motion_ref_joint_pos_rel", mdp.motion_ref_joint_pos_rel),
    ("motion_ref_joint_vel", mdp.motion_ref_joint_vel),
    ("motion_anchor_pos_b", mdp.motion_anchor_pos_b),
    ("motion_anchor_ori_b", mdp.motion_anchor_ori_b),
    ("motion_pad_center_relative", mdp.motion_pad_center_relative),
    ("motion_anchor_relative", mdp.motion_anchor_relative),
  )
  for name, func in motion_lanes:
    actor[name] = ObservationTermCfg(func=func, params={"command_name": motion_command})
  actor["microspine_pad_center_world"] = ObservationTermCfg(
    func=mdp.microspine_pad_center_world,
    params={"command_name": affordance_command},
  )
  actor["root_pose_world"] = ObservationTermCfg(func=mdp.root_pose_world)
  actor["wall_face_corners"] = ObservationTermCfg(func=mdp.wall_face_corners)
  assert tuple(actor) == TRANSITION80_ACTOR_TERMS
  actor_group.terms = actor

  critic: dict[str, ObservationTermCfg] = {
    name: dataclasses.replace(term, noise=None) for name, term in actor.items()
  }
  critic["robot_body_pos_b"] = ObservationTermCfg(
    func=mdp.robot_body_pos_b, params={"command_name": motion_command}
  )
  critic["robot_body_ori_b"] = ObservationTermCfg(
    func=mdp.robot_body_ori_b, params={"command_name": motion_command}
  )
  for name in _INHERITED_CRITIC_TERMS:
    critic[name] = old_critic[name]
  critic["microspine_pad_normal_force"] = ObservationTermCfg(
    func=mdp.microspine_pad_normal_force,
    params={"command_name": affordance_command},
    scale=0.01,
  )
  critic["microspine_pad_shear_down"] = ObservationTermCfg(
    func=mdp.microspine_pad_shear_down,
    params={"command_name": affordance_command},
    scale=0.1,
  )
  assert tuple(critic) == TRANSITION80_ACTOR_TERMS + TRANSITION80_CRITIC_TAIL
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=critic,
    concatenate_terms=True,
    enable_corruption=False,
  )


def phybot_c2_climb_transition80_env_cfg(
  play: bool = False,
  domain_rand: bool = True,
  student_obs: bool = False,
  wall_angle_deg: float = TRANSITION80_WALL_ANGLE_DEG,
  constraint_mode: str = "cat",
  relative_frame: str = "world",
  asym_arm_tracking: bool = False,
  arm_pd: tuple[float, float] = TRANSITION80_ARM_PD,
  waist_pd: tuple[float, float] = TRANSITION80_WAIST_PD,
  engage_gate_floor: float | None = TRANSITION80_ENGAGE_GATE_FLOOR,
) -> ManagerBasedRlEnvCfg:
  """80-degree walk->climb transition: stand at the wall, latch all four, hold.

  Args:
    play: deterministic single-env evaluation (start at frame 0, no scatter,
      constraint cone off, curricula pinned).
    domain_rand: the climb s2r DR package (obs noise, encoder bias,
      friction/mass) — climb parity.
    student_obs: not supported (no transition80 student contract exists;
      the v2 actor contract IS the deployable one) — raises.
    wall_angle_deg: wall slope from horizontal; the clip asset must be the
      matching native-angle prologue (``climb_mount_prologue_a<deg>``).
    constraint_mode: engage cone mode, ``"cat"`` (soft) | ``"hard"``.
    relative_frame: body tracking frame, ``"world"`` | ``"anchor_yaw"``.
    asym_arm_tracking: LadderMan Eq.(1) split — arm links at half weight /
      double std; legs, torso and pads strict.
    arm_pd / waist_pd: (stiffness, damping) overrides of the explicit table.
    engage_gate_floor: climb's mismatch discount floor on the pose kernels
      (None = ungated).
  """
  if constraint_mode not in ("cat", "hard"):
    raise ValueError("constraint_mode must be 'cat' or 'hard'.")
  if student_obs:
    raise ValueError("transition80 has no student observation contract.")
  cfg = phybot_c2_climb_wall_env_cfg(
    contact_model="wang2017",  # preserve the separate legacy spring-field recipe
    play=play,
    domain_rand=domain_rand,
    student_obs=student_obs,
    robustness_release=False,
  )

  # Floor contact ON.
  base_spec_fn = cfg.scene.spec_fn
  cfg.scene.spec_fn = functools.partial(_floor_spec_fn, base_fn=base_spec_fn)

  ##
  # Explicit PD table (real-robot firmware values) + climb's 0.98 soft limit.
  ##
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.articulation = dataclasses.replace(
    robot_cfg.articulation,
    soft_joint_pos_limit_factor=0.98,
    actuators=transition80_actuators(arm_pd=arm_pd, waist_pd=waist_pd),
  )

  ##
  # Command — the prologue with FK body tables and adaptive RSI (assigned IN
  # PLACE to keep the test-enforced command order).
  ##
  cycle_motion = cfg.commands["climb_motion"]
  cfg.commands["climb_motion"] = mdp.TransitionMotionCommandCfg(
    entity_name=cycle_motion.entity_name,
    motion_file=str(mount_clip_path(wall_angle_deg)),
    affordance_command_name=cycle_motion.affordance_command_name,
    ee_body_names=cycle_motion.ee_body_names,
    sampling_mode="start" if play else "adaptive",
    joint_pos_noise=0.0 if play else cycle_motion.joint_pos_noise,
    spawn_xy_noise=(0.0, 0.0) if play else TRANSITION80_SPAWN_XY_NOISE,
    spawn_yaw_noise=0.0 if play else TRANSITION80_SPAWN_YAW_NOISE,
    relative_frame=relative_frame,  # type: ignore[arg-type]
    birth_freeze_steps=TRANSITION80_BIRTH_FREEZE_STEPS,
    birth_exclude_frames=() if play else TRANSITION80_BIRTH_EXCLUDE_FRAMES,
    spawn_early_fraction=0.0 if play else TRANSITION80_CLASSROOM_FRACTION,
    spawn_bias_frame_windows=() if play else _transition80_classroom_windows(str(mount_clip_path(wall_angle_deg))),
    spawn_bias_auto_rake_frames=0,
    adaptive_uniform_ratio=TRANSITION80_ADAPTIVE_UNIFORM_RATIO,
    adaptive_max_bin_prob=TRANSITION80_ADAPTIVE_MAX_BIN_PROB,
  )
  # A mount has no climb rate: the climb velocity command (constant zero,
  # no consumer in this task) is dropped outright.
  cfg.commands.pop("climb_velocity")

  ##
  # Observations — the explicit v2 contract (actor 238 / critic 419).
  ##
  transition80_observation_groups(cfg)

  ##
  # Microspine plant — spring field (v3 plant), seed settle guard.
  ##
  microspine_cfg = cfg.commands["microspine_affordance"].microspine_cfg
  # v6 dare: the arms become ordinary wall contacts (pairs in the spec fn).
  microspine_cfg.arm_wall_contact_geoms = TRANSITION80_ARM_WALL_GEOMS
  microspine_cfg.spring_field_enabled = True
  microspine_cfg.limit_surface.seed_hold_judge_guard_steps = TRANSITION80_SEED_GUARD_STEPS

  ##
  # Rewards — replaced wholesale.
  ##
  tracking_bodies = None
  if asym_arm_tracking:
    tracking_bodies = tuple(
      n for n in mdp.DEFAULT_TRACKING_BODY_NAMES if n not in mdp.ARM_TRACKING_BODY_NAMES
    )
  pad_bodies = tuple(cycle_motion.ee_body_names)
  gate = {
    "engage_gate_floor": engage_gate_floor,
    "engage_gate_grace_s": TRANSITION80_ENGAGE_GATE_GRACE_S,
    "engage_gate_grace_fraction": None,
  }
  rewards: dict[str, RewardTermCfg] = {
    "motion_anchor_pos": RewardTermCfg(
      func=mdp.motion_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "climb_motion", "std": 0.3, **gate},
    ),
    "motion_anchor_ori": RewardTermCfg(
      func=mdp.motion_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "climb_motion", "std": 0.4, **gate},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_body_position_error_exp,
      weight=1.0,
      params={"command_name": "climb_motion", "std": 0.3, "body_names": tracking_bodies, **gate},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "climb_motion", "std": 0.4, "body_names": tracking_bodies, **gate},
    ),
    # Pad FACE centers, UNGATED (v1g): the one pose income that must stay
    # full while unwelded — it is the path to compliance.
    # v6 dare: std 0.2 -> 0.08. At 0.2 a pad hovering 3 cm off the wall kept
    # 98 % of this pay (forensics: welded 48.0 vs hover 47.2 per s) -- the
    # last centimeters to contact were free to skip.
    "motion_pad_pos": RewardTermCfg(
      func=mdp.motion_pad_center_error_exp,
      weight=1.0,
      params={"command_name": "climb_motion", "std": 0.08},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "climb_motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "climb_motion", "std": 3.14},
    ),
    "engage_progress": RewardTermCfg(
      func=mdp.engage_progress,
      weight=TRANSITION80_PHI_WEIGHT,
      params={
        "command_name": "climb_motion",
        "affordance_command_name": "microspine_affordance",
        "edge_lead_time_s": 2.0,
        "reach_radius_m": TRANSITION80_PHI_REACH_RADIUS_M,
        "touch_radius_m": 0.15,
        "reach_share": TRANSITION80_PHI_REACH_SHARE,
        "touch_level": TRANSITION80_PHI_TOUCH_LEVEL,
        "load_share": TRANSITION80_PHI_LOAD_SHARE,
        "approach_share": TRANSITION80_PHI_APPROACH_SHARE,
        "approach_radius_m": TRANSITION80_PHI_APPROACH_RADIUS_M,
        "press_share": TRANSITION80_PHI_PRESS_SHARE,
        "press_ref_n": TRANSITION80_PHI_PRESS_REF_N,
        "anchor_radius_m": _stage0(TRANSITION80_ANCHOR_RADIUS_STAGES),
      },
    ),
    # Ignition-period value (user call 2026-08-21, v1d): at -0.1 the term is a
    # pure variance tax (E[sum (da)^2] ~ 2*21*sigma^2) that collapsed sigma
    # 0.60 -> 0.40 within ~35 iterations in three runs regardless of cone or
    # PD changes, before any skill could form. -0.02 keeps the smoothness
    # gradient at a fifth of the tax; raise back toward climb's -0.1 once the
    # chain trains (a reward_weight schedule is the intended path).
    "hold_margin_wage": RewardTermCfg(
      func=mdp.scheduled_hold_margin_wage,
      weight=TRANSITION80_HOLD_MARGIN_WAGE_WEIGHT,
      params={
        "command_name": "climb_motion",
        "affordance_command_name": "microspine_affordance",
        "full_margin": TRANSITION80_HOLD_MARGIN_FULL,
      },
    ),
    # v5 rip bill (see TRANSITION80_RIP_PENALTY_STAGES).
    "anchor_rip_penalty": RewardTermCfg(
      func=mdp.scheduled_anchor_rip_penalty(),
      weight=-1.0,
      params={
        "command_name": "climb_motion",
        "affordance_command_name": "microspine_affordance",
        "weight_per_pad": (
          _stage_final(TRANSITION80_RIP_PENALTY_STAGES)
          if play
          else _stage0(TRANSITION80_RIP_PENALTY_STAGES)
        ),
        "release_lead_time_s": 0.5,
        "seed_settle_exempt_s": TRANSITION80_SEED_SETTLE_EXEMPT_S,
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=TRANSITION80_ACTION_RATE_WEIGHT),
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": _policy_joint_cfg()},
    ),
  }
  if asym_arm_tracking:
    rewards["motion_body_pos_arms"] = RewardTermCfg(
      func=mdp.motion_body_position_error_exp,
      weight=0.5,
      params={"command_name": "climb_motion", "std": 0.6, "body_names": mdp.ARM_TRACKING_BODY_NAMES},
    )
    rewards["motion_body_ori_arms"] = RewardTermCfg(
      func=mdp.motion_body_orientation_error_exp,
      weight=0.5,
      params={"command_name": "climb_motion", "std": 0.8, "body_names": mdp.ARM_TRACKING_BODY_NAMES},
    )
  cfg.rewards = rewards

  ##
  # Terminations — replaced wholesale.
  ##
  robot_scene_cfg = SceneEntityCfg("robot")
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_height": TerminationTermCfg(
      func=mdp.root_below_wall_altitude,
      params={"minimum_altitude": 0.3, "motion_command_name": "climb_motion"},
    ),
    "bad_state": TerminationTermCfg(
      func=mdp.bad_state,
      params={"asset_cfg": robot_scene_cfg, "max_velocity": 20.0},
    ),
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_termination(),
      params={
        "command_name": "climb_motion",
        "threshold": _stage0(TRANSITION80_ANCHOR_POS_STAGES),
        "persist_steps": 10,
      },
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori_termination(),
      params={"command_name": "climb_motion", "threshold": 0.8, "persist_steps": 10},
    ),
    # Pad FACE centers (v1g), key kept as body_pos for lane continuity.
    "body_pos": TerminationTermCfg(
      func=mdp.bad_pad_center_termination(),
      params={
        "command_name": "climb_motion",
        "threshold": _stage0(TRANSITION80_PAD_POS_STAGES),
        "persist_steps": 10,
      },
    ),
    "contact_loss": TerminationTermCfg(
      func=mdp.scheduled_contact_loss_termination(),
      params={
        "command_name": "climb_motion",
        "affordance_command_name": "microspine_affordance",
        "grace_s": _stage0(TRANSITION80_CONTACT_GRACE_STAGES),
        "anchor_radius_m": _stage0(TRANSITION80_ANCHOR_RADIUS_STAGES),
      },
    ),
    # v5: the ``anchor_rip`` death is retired -- a rip is billed by the
    # ``anchor_rip_penalty`` reward (TRANSITION80_RIP_PENALTY_STAGES) and a
    # never-re-seated pad still meets the contact law below.
    "wall_contact": TerminationTermCfg(
      func=mdp.illegal_wall_contact_termination(),
      params={
        "command_name": "climb_motion",
        "geom_names": TRANSITION80_FENCE_GEOMS,
        "threshold_m": 0.0,
        "gate": "none",
      },
    ),
    "engage_constraint": TerminationTermCfg(
      func=mdp.engage_constraint_termination(),
      params={
        "command_name": "climb_motion",
        "affordance_command_name": "microspine_affordance",
        "mode": "off" if play else constraint_mode,
        "tilt_limit_deg": _stage0(TRANSITION80_TILT_STAGES),
        "grind_limit_mps": _stage0(TRANSITION80_GRIND_STAGES),
        "slam_limit_mps": _stage0(TRANSITION80_SLAM_STAGES),
        "force_limit_n": _stage0(TRANSITION80_FORCE_STAGES),
        "touch_force_n": 3.0,
        "edge_lead_time_s": 2.0,
        "p_tilt": _stage0(TRANSITION80_SOFT_P_STAGES),
        "p_vel": _stage0(TRANSITION80_SOFT_P_STAGES),
        "p_force": _stage0(TRANSITION80_SOFT_P_STAGES),  # v3f: annealed with p_tilt/p_vel
        "p_unscheduled": 1.0,
        "ee_indices": (0, 1, 2, 3),
        # v1b: sustained-violation persistence (0.1 s = the weld engine's own
        # debounce) — noise spikes do not bill, behaviour does.
        "persist_steps": 5,
        "seed_settle_exempt_s": TRANSITION80_SEED_SETTLE_EXEMPT_S,
      },
    ),
  }

  ##
  # Curriculum — replaced wholesale: wall pinned, shear ladder, anneals.
  ##
  curriculum: dict[str, CurriculumTermCfg] = {
    "wall_angle": CurriculumTermCfg(
      func=mdp.wall_angle,
      params={
        "angle_stages": [{"iter": 0, "angle_deg": float(wall_angle_deg)}],
        "steps_per_iter": _STEPS_PER_ITER,
        "angle_jitter_deg": 0.0,
      },
    ),
    "engage_shear_ladder": CurriculumTermCfg(
      func=mdp.microspine_param_schedule,
      params={
        "param_name": "engage_shear_threshold_n",
        "stages": (
          [{"iter": 0, "value": _stage_final(TRANSITION80_SHEAR_LADDER)}]
          if play
          else [dict(s) for s in TRANSITION80_SHEAR_LADDER]
        ),
        "command_name": "microspine_affordance",
        "steps_per_iter": _STEPS_PER_ITER,
      },
    ),
    "engage_antirake_ladder": CurriculumTermCfg(
      func=mdp.microspine_param_schedule,
      params={
        "param_name": "engage_max_antirake_vel",
        "stages": (
          [{"iter": 0, "value": _stage_final(TRANSITION80_ANTIRAKE_LADDER)}]
          if play
          else [dict(s) for s in TRANSITION80_ANTIRAKE_LADDER]
        ),
        "command_name": "microspine_affordance",
        "steps_per_iter": _STEPS_PER_ITER,
      },
    ),
  }
  if not play:
    curriculum["contact_law_grace"] = _termination_anneal(
      "contact_loss", "grace_s", TRANSITION80_CONTACT_GRACE_STAGES
    )
    curriculum["contact_law_radius"] = _termination_anneal(
      "contact_loss", "anchor_radius_m", TRANSITION80_ANCHOR_RADIUS_STAGES
    )
    curriculum["rip_bill"] = CurriculumTermCfg(
      func=mdp.reward_param_schedule,
      params={
        "reward_name": "anchor_rip_penalty",
        "param_name": "weight_per_pad",
        "stages": [dict(s) for s in TRANSITION80_RIP_PENALTY_STAGES],
        "steps_per_iter": _STEPS_PER_ITER,
        "interpolate": True,
      },
    )
    curriculum["engage_progress_radius"] = CurriculumTermCfg(
      func=mdp.reward_param_schedule,
      params={
        "reward_name": "engage_progress",
        "param_name": "anchor_radius_m",
        "stages": [dict(s) for s in TRANSITION80_ANCHOR_RADIUS_STAGES],
        "steps_per_iter": _STEPS_PER_ITER,
      },
    )
    curriculum["anchor_pos_anneal"] = _termination_anneal(
      "anchor_pos", "threshold", TRANSITION80_ANCHOR_POS_STAGES
    )
    curriculum["body_pos_anneal"] = _termination_anneal(
      "body_pos", "threshold", TRANSITION80_PAD_POS_STAGES
    )
    curriculum["constraint_tilt"] = _termination_anneal(
      "engage_constraint", "tilt_limit_deg", TRANSITION80_TILT_STAGES
    )
    curriculum["constraint_grind"] = _termination_anneal(
      "engage_constraint", "grind_limit_mps", TRANSITION80_GRIND_STAGES
    )
    curriculum["constraint_slam"] = _termination_anneal(
      "engage_constraint", "slam_limit_mps", TRANSITION80_SLAM_STAGES
    )
    curriculum["constraint_force"] = _termination_anneal(
      "engage_constraint", "force_limit_n", TRANSITION80_FORCE_STAGES
    )
    curriculum["constraint_p_tilt"] = _termination_anneal(
      "engage_constraint", "p_tilt", TRANSITION80_SOFT_P_STAGES
    )
    curriculum["constraint_p_vel"] = _termination_anneal(
      "engage_constraint", "p_vel", TRANSITION80_SOFT_P_STAGES
    )
    curriculum["constraint_p_force"] = _termination_anneal(
      "engage_constraint", "p_force", TRANSITION80_SOFT_P_STAGES
    )
  if not play:
    # v6e: cover the 22.6 s clip + settle with margin (base 30 s; play 1e9).
    cfg.episode_length_s = 35.0

  cfg.curriculum = curriculum

  ##
  # v5 gimbal travel ladder (events write the per-world jnt_range).
  ##
  gimbal_stages = (
    [{"iter": 0, "value": _stage_final(TRANSITION80_GIMBAL_TRAVEL_STAGES)}]
    if play
    else [dict(s) for s in TRANSITION80_GIMBAL_TRAVEL_STAGES]
  )
  for name, mode in (("pad_gimbal_travel_init", "startup"), ("pad_gimbal_travel", "reset")):
    cfg.events[name] = EventTermCfg(
      mode=mode,
      func=mdp.pad_gimbal_travel_schedule,
      params={
        "stages": gimbal_stages,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_gimbal_[xyz]",)),
        "steps_per_iter": _STEPS_PER_ITER,
      },
    )

  ##
  # Cosmetic init pose (RSI overwrites on every reset).
  ##
  robot_cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(TRANSITION80_STANDING_X, 0.0, 0.726),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos=dict(PHYBOT_C2_STANDING_JOINT_POS),
    joint_vel={".*": 0.0},
  )

  ##
  # Metrics — climb's + handoff readiness + constraint lanes.
  ##
  cfg.metrics["mount_handoff_ready"] = MetricsTermCfg(
    func=mdp.mount_handoff_ready,
    params={"command_name": "climb_motion", "affordance_command_name": "microspine_affordance"},
  )
  cfg.metrics["anchor_rips"] = MetricsTermCfg(
    func=mdp.scheduled_anchor_rips, params={"reward_name": "anchor_rip_penalty"}
  )
  cfg.metrics["anchor_ripped_pads"] = MetricsTermCfg(
    func=mdp.scheduled_anchor_ripped_pads, params={"reward_name": "anchor_rip_penalty"}
  )
  cfg.metrics["scheduled_hold_margin"] = MetricsTermCfg(
    func=mdp.scheduled_hold_margin,
    params={"command_name": "climb_motion", "affordance_command_name": "microspine_affordance"},
  )
  cfg.metrics["rh_hold_margin"] = MetricsTermCfg(
    func=mdp.scheduled_hold_margin,
    params={
      "command_name": "climb_motion",
      "affordance_command_name": "microspine_affordance",
      "ee_index": 1,
    },
  )
  for key in ("tilt", "grind", "slam", "force", "unscheduled"):
    cfg.metrics[f"constraint_{key}_violation"] = MetricsTermCfg(
      func=mdp.engage_constraint_violation,
      params={"termination_name": "engage_constraint", "key": key},
    )
  cfg.metrics["constraint_delta"] = MetricsTermCfg(
    func=mdp.engage_constraint_delta,
    params={"termination_name": "engage_constraint"},
  )

  return cfg
