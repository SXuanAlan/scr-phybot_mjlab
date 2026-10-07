# transition80: configuration history

The reasoning and experiment history that used to live in the comments of
`src/scr_phybot_mjab/tasks/climb_transition80/config/phybot_c2/env_cfgs.py`
(as of commit aaa388c, before the factory was split into section modules),
grouped by the code they annotated. The code now carries one-line present-tense
reasons; the run names, dates and probe results below explain how each value was
reached. Kept verbatim.

Where the values live now (same directory, `TRANSITION80_` prefix dropped):

| Old name | Now |
|---|---|
| `*_STAGES`, `ANTIRAKE_LADDER`, `SEED_SETTLE_EXEMPT_S` | `schedules.py` |
| `SPAWN_*_NOISE`, `BIRTH_FREEZE_STEPS`, `BIRTH_EXCLUDE_FRAMES`, `CLASSROOM_*`, `ADAPTIVE_*`, `MOTION_FILE` | `commands.py` (inline in the motion command; the clip path is `assets.motions.CLIMB_MOUNT80_MOTION`) |
| `SEED_GUARD_STEPS`, `ARM_WALL_GEOMS` | `microspine.py` |
| `*_PD`, `STANDING_X` | `robot.py` |
| `FENCE_GEOMS` | `terminations.py` |
| `ENGAGE_GATE_*`, `PHI_*`, `HOLD_MARGIN_*`, `ACTION_RATE_WEIGHT` | `rewards.py` (inline) |
| `REMOVED_OBS`, `ACTOR_TERMS`, `CRITIC_TAIL`, `OBS_DIMS` | `observations.py` |
| `WALL_ANGLE_DEG` | `curricula.py` (inline) |

## `TRANSITION80_SPAWN_XY_NOISE = (0.05, 0.03)`

Blind-delivery spawn scatter (walk hands the robot over with tape-mark
accuracy): +-wall-distance, +-lateral (m), +-yaw (rad, 5 deg). Standing
frames only; the reference and its anchors stay canonical.

## `TRANSITION80_STANDING_X = 1.31`

Cosmetic standing base x for the a80 clip (standoff 0.19 from the base
line at x = 1.5; RSI overwrites on every reset).

## `TRANSITION80_BIRTH_FREEZE_STEPS = 25`

v57 seed settle guard: the 1 g birth transient rips young seeded welds
within the stock 2-step readout guard; no gravity ladder here, so the
guard alone must outwait the settle (0.12 s). Seeded RSI welds only.
v3 (2026-08-22): SEEDED-BIRTH FREEZE — the reference clock holds on the
birth frame for 25 steps (0.5 s) so the settle transient is absorbed with
the reference standing still (user design; passively validated with a
matching judge guard: seed loss at the pivot births 100 % -> 14 %, 790
96 % -> 6 %). See TRANSITION80_SEED_GUARD_STEPS for why the judge guard
does NOT span the freeze in training.

## `TRANSITION80_SEED_GUARD_STEPS = 6`

v3b (2026-08-22 02:0x): the judge guard is DECOUPLED from the freeze again.
With the guard spanning the whole 25-step freeze the resumed policy leaned
on unbreakable seeds (scheduled_hold_margin -4..-5.5) and every seed died
on the first judged step (616/616 rips at step 26, rip share 91 %, even
births that never ripped in v2). The freeze keeps the reference still; the
seeds are judged from step 7 as in v2 (the 0.2 s settle exemption covers
the transient rips), so overloading a seed costs the policy from step 11.

## `TRANSITION80_BIRTH_EXCLUDE_FRAMES: tuple[tuple[int, int], ...] = ()`

v3 TENABLE-BIRTH MASK: frame ranges (end exclusive) where a passive hold
loses >= 20 % of its seeds even after the guard (seed_diag map, cross v3,
2026-08-22): right_foot_mount 270-370 (LH pulled into tension/lateral while
the right foot lifts) and the end of hand_place_right 500-540 (RF toe in
reverse shear + LH ultimate during the RH reach). RSI never births there;
the phases stay reachable by playing through. REGENERATE after any clip
re-choreography (scripts/transition80_birth_mask.py in archive/pre-cleanup-2026-09).
cross v3r6 (2026-08-22 14:0x, place-right-s 3.2 -> the reach ends 20 frames
later; hand rakes 0.5 s): right_foot_mount 270-379, the end of
hand_place_right 510-569, and one mid-hand_up_left frame 790-799 (LF
lateral under a passive hold, 95 %; the compressed swing arc). Map:
logs/clip_drafts/validator_logs/a80_cross_v3r6_passive_map.txt.
v4b (2026-08-22 13:xx, user decision): MASK OFF. The passively untenable
births (right_foot_mount 270-379, the tiptoe reach 510-569, frame 790) are
the policy's to learn with active loading (v1g: rip 18 % -> 3 % in ~100
iter; v3b: 31 % -> 7 %). The mask also killed adaptive bin 6 outright and
two of the five ignition-classroom rake windows ((355,380), (540,565):
pool 125 -> 75 frames). Mechanism kept (birth_mask_from_ranges); re-arm at
launch with the v3r6 map
  --env.commands.climb-motion.birth-exclude-frames 270 380 510 570 790 800
(the flag with no values is ()).

## `TRANSITION80_CLASSROOM_FRACTION = 0.45`

v4 IGNITION CLASSROOM (user decision 2026-08-22): this share of births lands
in the pre-landing rake windows [edge - rake_frames, edge) of every engage
rising edge — pad at the wall, reference dragging it down, weld one
press-and-hold away. The v3b-v3f probes put the hands 3 cm off the wall on
98.7 % of window steps; adaptive RSI (1 s bins, failure-weighted) never
targets the 0.3-0.5 s rakes. Rake frames = round(rake_s * fps): cross v3
0.3 s -> 15; cross v4 0.5 s -> 25 (update with the clip).
v6e (2026-08-30, user decision): 0.3 -> 0.45, and the classroom windows are
EXPLICIT now -- every ignition rake window (the old auto set, recomputed
from the clip at config-build time) PLUS the first 30 frames of each
closing hand step, so hand-up births start with >= 1.2 s of runway before
the landing deadline instead of dying at the window's tail.

## `TRANSITION80_ADAPTIVE_UNIFORM_RATIO = 0.5`

v5e (2026-08-23, user decision: tighten the adaptive sampler, guard stays
6): the failure-weighted start bins fed on the closing hand-up windows --
births there die in 0.2-0.5 s (seeds torn at the first judged step, head
into the fence), each death re-feeds the histogram, and the replica probe
on v5d model_4025 found 55 % of all births in bins 17-20 (f 825-971).
Uniform floor 0.1 -> 0.5 and per-bin cap 0.35 -> 0.08 (~1.7x uniform over
21 bins): the four hard bins can take at most ~32 % between them.

## `TRANSITION80_SEED_SETTLE_EXEMPT_S = 0.12`

Birth-settle exemption shared by the rip law and the engage cone. 0.2 s
(10 steps) covers the step-2 landing impulse and the 6-step seed guard;
the 0.5 s window of v1c-v1f was a loophole — model_1550 window probes
showed the policy tearing every seeded anchor off at 0.15-0.3 s for free
and pantomiming the rest of the choreography without contact. Rips from
step 11 on (the hold-still probe's compliance-driven 5-19 step rips) are
now the policy's responsibility (v57: learnable with active loading).
v5b (2026-08-23, user decision): 0.2 -> 0.12 s = the 6-step judge guard.
The 4 judged-but-exempt steps (7-10) were a FREE window: the replica probe
on v5 model_1575 found 83 % of seeds shed before step 9 (median step 5),
feet occupancy 0.02, self-weld rate decaying 0.06 -> 0.005 -- the policy
dumps its seeds for free and keeps one hand. With the rip now a bill (not
a death) there is no gradient desert to protect against: a seed lost at
step 7+ pays rent until it is re-welded.

## `TRANSITION80_RIP_PENALTY_STAGES = [`

v5 RIP BILL (user decision 2026-08-23, climb's rule): a scheduled anchor
torn off is a per-step PENALTY until re-welded, not a death (the
``anchor_rip`` termination is retired; the contact law's grace stays as
the backstop). Per flagged pad, per second; piecewise-linear ramp. Sized
under the per-pad tracking income (~8.5/s total) so the bill never makes
the early death the optimum (08-13 lesson).

## `TRANSITION80_GIMBAL_TRAVEL_STAGES = [`

v5 GIMBAL TRAVEL LADDER (user design 2026-08-23): the pad gimbals' stops
open at +-15 deg and anneal piecewise-linearly to the +-3 deg real stop
(PHYBOT_C2_PAD_GIMBAL_TRAVEL_RAD) -- angle compliance as a curriculum,
climb's tilt-peel anneal re-done on the real joint. Written into the
per-world jnt_range by an event (startup + every reset).
v5d (2026-08-23, user decision): STAGED slow anneal, the 3-deg tightening
POSTPONED -- "learn at 15 first". The v5c ladder (linear 300->3000) hit the
3-deg stop while the hand-up windows were still passively untenable there
(the 29/80 gate list): every birth in 800-885/905-995 died on the wall
fence within 0.2-0.4 s (head-first pitch after the seeds tore, fence
forensics: neck 52 %). Steps 15 -> 8 -> 3 with short 500-iter ramps
between holds; push the boundaries later if 15 is not yet learned.
v6b (2026-08-24, user decision): the 15-deg hold runs to iter 10000 (the
run's whole max-iterations budget) -- learn everything at 15 deg first;
the tightening stages beyond are placeholders past the horizon.

## `{"iter": 0, "value": 15.0},`

v6d (2026-08-25, user decision after the 8-deg record run and the first
full from-stand chain): 8 -> 5 now, 6000 iters at 5, then 3 for 10000
(run ends at 26000). 200-iter ramps at each boundary; every OTHER
parked tightening stays beyond the horizon (30000/32000) -- those
advance by hand on graduation evidence, only the gimbal moves here.

## `TRANSITION80_FENCE_GEOMS = (`

Full-body wall fence: every link that never legitimately reaches the wall
(the pads, forearms/elbows and feet/toes stay out). Geom SURFACES are
judged, not link origins (08-13 lesson). Roster = the 08-13/08-14
measured reference envelope (body links >= +13.3 cm, upper arms +8.4 cm
through stand/raise/place; a80 cross v3 keeps >= +10.8 mm on every frame).
v6 dare (2026-08-24): the four shoulder/upper-arm geoms LEFT the fence --
arms get ORDINARY physical wall contact (TRANSITION80_ARM_WALL_GEOMS pairs)
and may lean on the wall; head/torso/base/hips/knees stay lethal.

## `TRANSITION80_ANTIRAKE_LADDER = [`

v3d: anti-rake ignition ladder — the weld engine requires the pad to drift
down-wall no faster than this while loading (m/s). 2 cm/s was met on only
58 % of touching steps (touch v_tan p50 0.24-0.31 m/s); 5 cm/s opens the
gate to the current touches, then anneals back to the spec.

## `TRANSITION80_CONTACT_GRACE_STAGES = [`

Contact law (LadderMan executioner): grace after a scheduled edge, and the
radius within which a weld satisfies the schedule. Phi's anchor radius
anneals in lockstep.
v6 dare (2026-08-24, user decision): 1 s FLAT. LadderMan's discipline (its
Table 5 uses 30 sim steps ~ 0.6 s): a scheduled contact missing for one
second ends the episode -- hovering out the old 3 s for tracking pay is no
longer a strategy. Forensics on v5g model_9975: hover kept 98 % of the two
big position terms and the approach was dithered out over 88-106 steps.

## `TRANSITION80_ANCHOR_POS_STAGES = [{"iter": 0, "value": 0.35}, {"iter": 50000, "value": 0.25}]`

Tracking-deviation executioners (persist 10 steps = 0.2 s).

## `TRANSITION80_TILT_STAGES = [`

Engage constraint cone limits (start -> graduation) and soft probabilities.

## `TRANSITION80_SOFT_P_STAGES = [`

v3f (2026-08-22 05:4x): the cone's termination PROBABILITY now starts near
zero and anneals up. Four probes (v3b-v3e) showed a learned avoidance: the
hands park 3 cm off the wall on 98.7 % of their window steps because every
early touch was lethal (CaT p 0.5 from iteration 0) before any press/hold
skill existed; reward-side fixes (Phi x4, ignition ladder, approach+press
lanes) did not move the touch rate and sigma climbed 0.59 -> 0.77 (no new
gradient). With the cone nearly off inside the windows, noise-driven
touches survive and get paid; the cone tightens once ignition exists.

## `TRANSITION80_ACTION_RATE_WEIGHT = -0.05`

action_rate_l2 weight for the ignition period (climb parity is -0.1).
v5b (2026-08-23): -0.02 -> -0.05 (halfway to climb's -0.1) -- the model_2500
window videos jitter and the replica probe reads action_saturation 0.53;
the chain now trains (eplen 190 steps), so the smoothness gradient can
be raised off its ignition-period floor.

## `TRANSITION80_ENGAGE_GATE_FLOOR = 0.10`

Climb's schedule-adherence discount on pose income (v1e, user call): pose
kernels pay ``floor`` of their value while any pad's latch state disagrees
with the chart, ramped over ``grace_fraction`` of the violated window near
its edges. Velocity kernels and Phi are not gated (climb parity).
v6 dare: 0.25 -> 0.10 (hover keeps one tenth of the gated tracking pay).

## `TRANSITION80_ENGAGE_GATE_GRACE_S = 0.5`

v1g: a FIXED 0.5 s ramp. The climb-parity grace_fraction 0.3 of a 10 s
window was a 3 s ramp — the gate still read 0.83 when the contact law
executed at edge + 3 s (model_1550 trace); the discount never bit.

## `TRANSITION80_PHI_REACH_RADIUS_M = 0.5`

Phi reach radius: the hand parked 46 cm from its anchor, outside the 0.25 m
reach stage (Phi = 0 there); 0.5 m gives a slope from where it stands.

## `TRANSITION80_PHI_WEIGHT = 3.0`

v3c (2026-08-22 03:0x): Phi weight 0.5 -> 2.0 (queued candidate, user-seen).
v3b model_1150 ignition probe = a HOVER TRAP: in its windows the LH pad
parks 3-4 cm off the wall (98.6 % of in-window steps never touch; cone
grind/slam/force bills all zero) and eats the pose income until the 3 s
contact-law grace executes it (closs 87 %). Phi is the only term that
prices touch (0.5+) over hover (<= 0.3) over weld (0.9+); at 0.5 the touch
premium was ~2 % of the per-step income, at 2.0 it is 8-20 %.
v6 dare: 2.0 -> 3.0, and the share mix moves toward approach/press (the
transition itself was underpaid: forensics diff only +17 % vs welded).

## `TRANSITION80_PHI_APPROACH_SHARE = 0.19`

v3e (2026-08-22 04:4x): Phi shape for the hover trap — pay the last 10 cm of
wall-normal approach and the press itself (see mdp.engage_progress).
reach 0.3 + approach 0.15 < touch 0.5; touch 0.5 + load 0.2 + press 0.2 = weld 0.9.

## `TRANSITION80_HOLD_MARGIN_WAGE_WEIGHT = 0.5`

v3 hold-margin dwell wage (user call 2026-08-22, "方案 A"): scheduled AND
welded pads are paid clip(hold_margin / 0.3, 0, 1) averaged over the
scheduled pads. Weight 0.5 (x dt -> <= 0.01/step, ~10 % of the tracking
income): a load-sharing preference, never rent, never a reason not to weld.

## `TRANSITION80_ARM_PD = (50.0, 5.0)`

Joint PD table = the real robot's motor-firmware table
(deploy/phybot_deploy_c2/MotorList/config/phybot_mini_{1,2}.yaml PDKp/PDKd;
identical to the shipped walk-AMP and squat tasks). Differences from the
climb pd100 table: waist 100 -> 20 and the eight arm joints 100/10 -> 50/5
(user call 2026-08-21: the 08-01 A0 hang bench showed deploy gains hold the
hang with 4.9 mm sag, and kp=100 arms turn σ=0.6 exploration noise into
~1 m/s pad grinds and tens-of-N slams that the engage cone then kills).
Every entry is a fresh dataclasses.replace copy: the actuator cfgs are
module singletons shared with walk/stand/climb and must never be mutated
in place (07-12 rule).

## `TRANSITION80_REMOVED_OBS = (`


Observation contract (v2, 2026-08-21) — written out in one place.

Retired from the inherited climb contract (all audited dead or redundant on
the a80 prologue at a pinned 80-degree wall): ``height_above_spawn`` (an
affine readout of root_pose_world), ``velocity_command`` (constant zero —
the climb_velocity command is dropped with it), ``microspine_disengage_time``
(a saturating copy of not-engaged), ``microspine_cooldown`` (non-zero only
for 0.2 s after a rare non-terminating forced release), ``wall_base_position``
(a constant; the face corners replace it), and six constant lanes of the
16-lane motion_command chart (now 10, see mdp.motion).


## `base_spec_fn = cfg.scene.spec_fn`

Floor contact ON.

## `robot_cfg = cfg.scene.entities["robot"]`


Explicit PD table (real-robot firmware values) + climb's 0.98 soft limit.


## `cycle_motion = cfg.commands["climb_motion"]`


Command — the prologue with FK body tables and adaptive RSI (assigned IN
PLACE to keep the test-enforced command order).


## `cfg.commands.pop("climb_velocity")`

A mount has no climb rate: the climb velocity command (constant zero,
no consumer in this task) is dropped outright.

## `transition80_observation_groups(cfg)`


Observations — the explicit v2 contract (actor 238 / critic 419).


## `microspine_cfg = cfg.commands["microspine_affordance"].microspine_cfg`


Microspine plant — the climb task's empirical unit model, plus arm wall
contacts and the seed settle guard.


## `microspine_cfg.arm_wall_contact_geoms = TRANSITION80_ARM_WALL_GEOMS`

v6 dare: the arms become ordinary wall contacts (pairs in the spec fn).

## `gate = {`


Rewards — replaced wholesale.


## `"motion_pad_pos": RewardTermCfg(`

Pad FACE centers, UNGATED (v1g): the one pose income that must stay
full while unwelded — it is the path to compliance.
v6 dare: std 0.2 -> 0.08. At 0.2 a pad hovering 3 cm off the wall kept
98 % of this pay (forensics: welded 48.0 vs hover 47.2 per s) -- the
last centimeters to contact were free to skip.

## `"hold_margin_wage": RewardTermCfg(`

Ignition-period value (user call 2026-08-21, v1d): at -0.1 the term is a
pure variance tax (E[sum (da)^2] ~ 2*21*sigma^2) that collapsed sigma
0.60 -> 0.40 within ~35 iterations in three runs regardless of cone or
PD changes, before any skill could form. -0.02 keeps the smoothness
gradient at a fifth of the tax; raise back toward climb's -0.1 once the
chain trains (a reward_weight schedule is the intended path).

## `"anchor_rip_penalty": RewardTermCfg(`

v5 rip bill (see TRANSITION80_RIP_PENALTY_STAGES).

## `robot_scene_cfg = SceneEntityCfg("robot")`


Terminations — replaced wholesale.


## `"body_pos": TerminationTermCfg(`

Pad FACE centers (v1g), key kept as body_pos for lane continuity.

## `"wall_contact": TerminationTermCfg(`

v5: the ``anchor_rip`` death is retired -- a rip is billed by the
``anchor_rip_penalty`` reward (TRANSITION80_RIP_PENALTY_STAGES) and a
never-re-seated pad still meets the contact law below.

## `"persist_steps": 5,`

v1b: sustained-violation persistence (0.1 s = the weld engine's own
debounce) — noise spikes do not bill, behaviour does.

## `curriculum: dict[str, CurriculumTermCfg] = {`


Curriculum — replaced wholesale: wall pinned, anti-rake ladder, anneals.


## `cfg.episode_length_s = 35.0`

v6e: cover the 22.6 s clip + settle with margin (base 30 s; play 1e9).

## `gimbal_stages = (`


v5 gimbal travel ladder (events write the per-world jnt_range).


## `robot_cfg.init_state = EntityCfg.InitialStateCfg(`


Cosmetic init pose (RSI overwrites on every reset).


## `cfg.metrics["mount_handoff_ready"] = MetricsTermCfg(`


Metrics — climb's + handoff readiness + constraint lanes.
