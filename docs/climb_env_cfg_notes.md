# Climb env cfg — tuning-decision archive (phybot_c2)

Verbatim narrative comments moved out of
`src/scr_phybot_mjab/tasks/climb_wall/config/phybot_c2/env_cfgs.py` during the
2026-08-10 restructure (three-layer split: mdp machinery / base task structure /
c2 robot binding). Original line numbers reference git `84e5d18` (the
pre-refactor state of that file). The code keeps a 1-3 line "why this value"
summary per tuned value pointing here; directive/contract comments (student
APPEND-ONLY, command ordering, critic re-instantiation, actuator
replace-not-assign, microspine finalize-before-events) stay in code and are NOT
archived here.

Entry titles carry the term key / cfg field, so `grep -n "<term key>"` on this
file lands on the right block.

> **Representation note (2026-08-10, same day, later):** the gravity-tilt
> representation these comments were written under was RETIRED the same day.
> Wall angles are now realized by tilting the WALL about its base line
> (x = 1.5, z = 0) under real gravity (`mdp.wall_angle`, curriculum key
> `wall_angle`), with the motion reference rotating along; the terrain floor
> is contact-ENABLED (transition-world parity). At 90° the two
> representations are bit-identical — every calibration and exam number
> quoted below carries over unchanged; "gravity ladder/angle" in the archived
> text reads as "wall-angle ladder" today.

---

## Motion file (`_CLIMB_MOTION_FILE`) — paced clip lineage  [L95-101]

```
# Recruitment-paced toe-brace crawl (4.52 s / 30 mm cycle, 6.6 mm/s): the
# fast clip's poses with every dwell restored to >= 0.3 s and foot swings to
# 0.75 s — the fast timing put seat/regrip holds (0.2 s) UNDER the 0.25 s
# fresh-anchor recruitment ramp, so each transfer loaded an immature anchor
# (motionref16's abandoned right foot). Gates 2026-07-30: static audit
# 0.8773 worst-util PASS, schedule-timing audit PASS (worst maturity gap
# 0.28 s >= ramp), reference-hold PASS (1/68 transients, margin +0.058).
```

## Microspine cfg (`## Microspine physics` section)

### initial_engaged_ee_names — RSI fallback seeding  [L150-153]

```
# Fallback reset latches only: the motion command's RSI overwrites them
# per env with the sampled clip frame's engage schedule right after the
# affordance's own reset (see seed_engaged). All-four matches frame 0.
```

### contact_friction — tangential support belongs to the limit surface  [L159-161]

```
# Ordinary contact is only for compression and seating. Tangential support
# must come primarily from the microspine equality/limit surface.
```

### reengage_cooldown_steps / engage_consecutive_steps — task-rate timing  [L162-166]

```
# A released limb must be able to re-seat before a partially supported
# robot falls. Keep the physical gate (two spheres above 3 N) unchanged,
# but use task-rate timing: 0.20 s cooldown + 0.06 s debounce. The former
# 0.60 s/0.10 s pair left otherwise-valid contacts one step short of every
# re-engagement in a stochastic rollout.
```

### anchor_lock_rotation + rake_mode — pad locks the whole patch  [L170-173]

```
# Constraint grip engages from compression count + face alignment; the
# limit surface owns release.
# A real spine pad locks position and orientation across the contact patch.
# Capacity follows the pad-local rake direction.
```

### pad_span_* — billing spans derived from the robot XML  [L176-179]

```
# Moment billing measures the pad from the robot XML itself: the billing
# spans are the contact-sphere grid extents (74.7 mm rake x 96 mm lateral
# on this model). Regenerate nothing when the XML pad changes; this
# derives at cfg-build time.
```

### moment_billing_mode = "per_unit" (2026-08-04)  [L185-195]

```
# Per-unit billing (2026-08-04): distribute the weld's 6D wrench onto the
# paper's 3x4 spine-unit grid (bolt-group statics) and judge every unit's
# 3D load against its F0/12 share of the closed surface — weakest unit
# releases the pad. Supersedes the worst_edge fold whose arm = span/2 is
# exact only for the two-row point abstraction (the real spread array
# concentrates bending on its edge units ~1.5-1.8x harder per axis:
# span*(N+1)/(6*(N-1)) = s/3 rake, 0.278*s lateral). The arm-sweep bench
# (logs/eval/arm_sweep) showed clean climbing is insensitive to billing
# harshness — only worst-F0 forced-release recoveries feel it.
# moment_arm_fraction stays at its 0.25 default, used only when a bench
# (engage_raster arm=) explicitly flips back to worst_edge mode.
```

### anchor_torquescale = 0.01 — reference-hold calibration  [L197-205]

```
# Weld rotational-residual scale, calibrated with the reference-hold
# stress test (validate_microspine_constraint.py reference-hold): the old
# 1 m default routed whole-body pitch couples through ~6 N*m of local pad
# moment ("four enormous flanges"); 0.04 still trapped 0.7-1.8 N*m of
# PD-parasitic moment in the wristless hands and force-released 65/79
# reference frames. At 0.01 the rotational rows are compliant enough that
# parasitic couples shed into force redistribution (peak |M| <= 0.07 N*m,
# all frames hold, margin +0.077 at the F0 = 400 N DR floor) while the
# honest-arm billing above still guards genuine sustained prying.
```

### wang2017 model + f_t_max_range (400, 650)  [L207-214]

```
# Closed Wang-2017 hold envelope (paper-faithful): tension capacity exists
# only alongside engaged shear, the forward lobe closes at the origin, and
# against-rake capacity collapses to the side-contact residual. F0 range
# raised versus the wedge's f_t_max because tension now taxes the whole
# surface: the 90 deg frame-0 hang loads hands at ~70 N shear + 14 N
# tension (~12 deg elevation, capacity ~0.6 * F0) — validated by
# scripts/validate_microspine_constraint.py hang. The paper's 12-tile palm
# demonstrated ~700 N pure shear, so 400-650 N per pad is conservative.
```

### tilt_peel_max_deg_range (3.5, 4.5) — the "4-deg" calibration pass (2026-07-31)  [L217-224]

```
# Tilt-peel calibration (2026-07-31, "4-deg" pass): the spine suspensions
# conform force-free over le_max = 5 mm (Wang 2017 Sec 2.3); rotating the
# chassis about the loaded edge row exhausts that travel at 5/74.7 mm =
# ~3.8 deg, after which the tension-side row is loaded structurally and
# pulls out — forced release, reason 5. Per-grip lottery like F0. The
# reference's engaged rotations measure <= 1.1 deg (3x margin at the DR
# floor); the pre-calibration policy rode the unlimited softness to
# 12-22 deg — exactly the sim2real region this closes.
```

### disable_contacts_while_latched — the margin 'barcode' fix (2026-07-24)  [L226-229]

```
# Latched pads hand their whole load path to the bilateral anchor; the
# sphere contacts (a redundant second path that fights the anchor and made
# the measured loads oscillate — motionref6's 90-deg margin 'barcode')
# switch off per env while latched and return on release (2026-07-24).
```

## Commands (`## Commands` section)

### climb_velocity lin_vel_z pin  [L234-236]

```
# Command exactly the average climb rate one reference cycle delivers
# (step_height / cycle duration), so the height-progress normalization and
# the commanded-velocity observation agree with the motion being imitated.
```

### climb_motion sampling_mode  [L267]

```
sampling_mode="uniform",  # DeepMimic RSI: random cycle phase per episode.
```

## Base-factory slim (deleted pops)

### gait_clock / hand_engaged_state obs drop rationale  [L247-250]

```
# Drop the base factory's fixed-period gait clock and legacy hand-latch
# obs: the motion command below provides the phase clock (cos/sin, tied to
# the reference cycle) and the reference engage schedule, and the
# microspine_engaged_state term added later covers all four latches.
```

### reset_robot_joints / reset_robot_base event drop rationale  [L451-453]

```
# Reference-state initialization owns the robot's reset pose and velocities
# (the motion command writes them AFTER reset events run); the generic
# perturbation events would only be overwritten.
```

## Robot cfg (`## Robot entity` section)

### toe geoms live on the robot entity spec  [L272-275]

```
# Foot toes brace the wall during foot regrips so the released spine pad
# never slides on the surface. The geoms live on the ROBOT entity spec
# (sensors resolve geom names against the entity, not the composed scene);
# their wall contact pairs are scene-level, added in _spec_fn below.
```

### shared walk collision / analytic wall plane  [L287-293]

```
# One shared CollisionCfg is the source of truth for both tasks. Climb adds
# sphere-wall pairs and grip equalities, but does not alter robot geoms.
# Keep the robot collision model byte-for-byte equivalent to walk. The
# climb wall uses an analytic plane for contact while retaining its finite
# box as a non-colliding visual; this avoids convex CCD substitutions on the
# robot itself.
```

### climb-zero stance v3 derivation (2026-07-22)  [L296-316]

```
# Override init state at task level (do NOT mutate the shared keyframe
# singleton on robot_cfg.init_state — other c2 tasks reuse it).
# Climb-zero stance v3 (2026-07-22): v2 with the hand pads raised 1.5 cm
# (pad z 1.738 -> 1.753) per the higher-reach request. On the wristless
# 4-DoF arm the flat-pad wall point is a function of base standoff alone,
# so the raise is bought by standing closer: base 0.173 m from the wall
# (v2 was 0.196); the legs re-solve with the foot pads 1.0 cm lower
# (z=0.566). Pads stay flat on the wall (|gap| <= 0.1 mm, tilt <= 0.7
# deg). Elbow straightens to -0.54 — the regime of v2's rejected 0.176 m
# variant — but the swing-singularity blowup that killed that variant
# predates the current gen_climb_clip.py safeguards (mid-swing pad-rot
# floor 20, posture anchor 0.6, frozen other-arm reference): the v3 cycle
# regenerates with swing pad-rot residual <= 0.2 deg and no yaw whip.
# Measured cycle cost vs regenerating on v2 (left-hand landing 3.1 mm,
# roll margin 1.8%, swing-corner wall graze +1.5 mm — the left-swing miss
# is inherent: the still-low right anchor drags the base): v3 lands the
# left hand 4.4 mm off schedule, shoulder-roll peaks 1.3% from the stop,
# graze +2.7 mm. Knee-to-wall clearance holds at 1.73 cm (less knee flex
# cancels the closer stance). A +2.0 cm / 0.166 m variant was generated
# and rejected: it sits exactly on the reach frontier (4.9 mm miss, roll
# 1.1%).
```

### pd100 actuator table (2026-08-02 deploypd-v3 revival)  [L348-357]

```
# pd100 table (2026-08-02 user call: revive deploypd-v3 from model_3300).
# These are the gains v3 was trained with — byte-matched against
# logs/.../2026-08-01_20-45-59_deploypd-v3-pd100/params/env.yaml — and a
# resumed checkpoint must get the plant it learned on: full upper chain
# INCLUDING shoulder_yaw at 100/10 (kd=10 lifts the hanging-chain zeta
# 0.35 -> 0.5), legs 100/10 over the 100/5 module default, ankles 50/5,
# neck default. (The 18c-exact 100/5 table this replaces served the
# robust-ladder campaign; it lives in the jv run dirs' param dumps.)
# dataclasses.replace, not assignment: the actuator cfgs are shared
# module singletons and walk/squat use the same objects.
```

### soft_joint_pos_limit_factor = 0.98  [L380-386]

```
# Reserve 1% of the hard range at each end. The action term below clamps
# PD targets to these limits; the physical hard stop remains a final
# guard. Widened from 0.9 (2026-07-21): the reference cycle works close
# to some stops — shoulder_roll kinematically peaks 0.06 rad from its
# hard limit at state 4 (one hand high, other anchored low) — and
# tracking must never fight the target clamp. The clip keeps its own
# clearance elsewhere via the gen script's limit-avoidance springs.
```

### wall offset / tune_pose stale note  [L391-394]

```
# Wall: keep the default WALL_OFFSET = (2.0, 0.0, 7.5). NOTE the stance-v2
# base x (1.304) no longer matches what scripts/tune_pose.py visualized for
# the v1 stance; the stance is now designed by the IK stance search
# (2026-07-21) with pads pinned to the wall plane.
```

## Events & DR (`## Events` / `if domain_rand` / `if robustness_release`)

### v4 DR revival rationale (2026-08-02)  [L458-465]

```
# v4 revival (2026-08-02 user call): resume deploypd-v3/model_3300 with
# v3's FULL walk-parity DR, byte-matched to that run's params/env.yaml.
# From-scratch-with-noise beats resume-then-inject: the graded R1/R2
# ladder (see the jv run dirs, 2026-08-02 midday) measured a permanent
# ~0.015 clean-exam tax injecting joint_vel noise into the noise-naive
# 18c line, while v3 carried this whole set from birth and examined
# 0.98 at 75-85 deg. The one thing that killed v3 was the gravity
# bridge, not the DR — see the angle_stages note below.
```

### body_friction regex scope  [L480-484]

```
# Walk's regex: every collision geom except the feet. The spine
# plates match too but their contype is 0 (walk CollisionCfg), so
# randomizing them is inert. Spine-sphere and toe-wall contacts are
# pair-driven — geom friction does not reach them (pair friction
# overrides), so the microspine calibration stays untouched.
```

### base_mass ±2 kg, not walk's ±5  [L498-501]

```
# +-2 kg, not walk's +-5: the static climb audit's thinnest hold
# margin is 12.3%, and a +5 kg draw pushes the worst F0 lottery out
# of the Wang envelope — no-fault forced releases would pollute the
# training signal. Revisit after a hang-bench pass at +2 kg.
```

### encoder_bias v3-exact  [L508-514]

```
# v3-exact: +-0.01 rad on every joint, drawn once per episode-world.
# At 0.4 m reach this is a 4-8 mm unobservable EE offset against the
# ~5 mm seating travel — the single suspect for real-pad reseat
# misses, which is why it ships in the base DR set.
```

### random_release — R-series history (2026-08-01 → R4)  [L519-534]

```
# Unscheduled-release recovery drill, deliberately NOT part of the base
# DR package (2026-08-01 night, user call). Rationale: an unplanned
# detach every ~5.5 s teaches "losing a latch mid-cycle is survivable",
# which is exactly the experience that makes weld-to-survive cheap once
# the wall gets steep — the engage-mismatch bill drifted monotonically
# through the 75-deg rung while it was on. The recovery skill it trains
# is still wanted for the distilled student (which cannot observe latch
# state), so it comes back as a FINAL stage once exact-90 is mastered,
# on top of a policy whose schedule discipline is already fixed.
# R4 stage (2026-08-03): default ON now that exact-90 is mastered — the
# drill pairs with the reseat-hold clock freeze (affordance.reseat_hold)
# so recovery is a trained skill, not schedule truancy. Interval starts
# gentle (6-12 s); tighten to (3, 8) once the recovery exam passes. For
# a FROM-SCRATCH retrain launch with robustness_release=False until
# exact-90 is mastered (2026-08-01 rationale: unplanned-release-is-normal
# economics makes welding cheap before schedule discipline forms).
```

## Rewards (`## Rewards` section) — one entry per term key

### rewards banner + motion_joint_pos (engage gate floor / grace fraction)  [L542-563]

```
# ------------------------------------------------------------------------
# Rewards — DeepMimic split. ``motion_*`` exp kernels teach the regrip
# cycle (no global position tracking); height/velocity progress terms
# drive actual climbing; the microspine battery enforces the physics.
# ------------------------------------------------------------------------
# -- motion imitation (positive tracking income, BeyondMimic exp kernel).
# engage_gate_floor 0.25 (motionref10): pose income multiplied by the
# schedule-adherence gate — "looking right" pays in full only while
# "gripping right". The engagement-blind pose kernels were the welded
# statue's last income pillar (ref8/ref9 both parked a diagonal at
# match~0.5 and lived off pose tracking; the cycle's poses never leave
# stance far enough for the statue to lose the kernel). Floor keeps a
# quarter of the income ungated so birth pose learning never starves;
# the gate uses the same edge-distance grace ramp as the mismatch bill.
# engage_gate_grace_fraction 0.3 (2026-07-30, motionref17): the gate's
# edge grace is a fraction of the violated window's own length, not a
# fixed 0.25 s — the fixed grace silently discounted welding whenever
# the clip was retimed (fast clip: only the middle 0.1 s of a 0.6 s foot
# window billed full price). At 0.3 a weld-through always forfeits ~70%
# of the window's gate credit, clip timing notwithstanding.
```

### motion_ee_pos — weight 2.5 (was 1.5, motionref5)  [L579-583]

```
# Root-frame EE positions: shapes reach and swing arcs without pinning
# the base to the clip's world trajectory. 2.5 (was 1.5, motionref5):
# the reach toward the next hold is the first link of the regrip chain
# (reach -> press -> latch); it must out-earn holding a parked limb
# still, or the weld attractor keeps winning the swing phase.
```

### motion_engage_mismatch — weight -1.0 + grace_fraction 0.3  [L604-617]

```
# The clip's engage schedule replaces the old commanded-gait machinery:
# a latch state disagreeing with the reference phase costs per step.
# -1.0 (was -0.25, raised 2026-07-21): the motionref1 audit showed the
# policy ignoring the schedule wholesale (match 0.44, LF parked at 0.03
# occupancy) because a wrong latch state cost pocket change next to the
# height income. ~2 mismatched pads must read as -2/s against a budgeted
# ~3.1/s task income, not -0.5/s against 22/s.
# grace_fraction 0.3 (2026-07-30, was grace_time_s 0.25): the bill ramps
# linearly over the outer 30% of the violated window to a scheduled flip
# (debounce 0.06 s + cooldown 0.20 s make instant compliance impossible)
# — the same countdown signal the policy sees in motion_command. As a
# fraction it survives clip retiming: a weld-through always pays ~0.7 of
# the window at full price (the fixed 0.25 s grace had cut the fast
# clip's foot-window weld bill nearly in half).
```

### motion_scheduled_press — seat-phase step credit (motionref5)  [L623-630]

```
# Seat-phase step credit (2026-07-22 evening, motionref5): every run so
# far welded one limb pair because a regrip's first motions earn nothing
# until the latch closes (motionref4 audit: feet occ 1.00 / hands 0.04 at
# iter 1450 — the 7th occurrence). Pays pad press inside the scheduled
# regrip window only (ref_engaged, or regrip countdown <= 0.3 s — the
# observed channel), stops at latch (engaged anchors carry tension), and
# the +0.4 ceiling stays below the -1.0 mismatch bill so pressing
# without latching is strictly net-negative.
```

### motion_toe_brace (2026-07-29)  [L636-642]

```
# Foot-toe brace shaping (2026-07-29, toe-brace reference): the released
# foot's support during its regrip is the TOE pressing the wall — the
# spine pad now travels clear of the surface (sliding an unloaded pad
# destroys spine tips). Pays saturated toe normal force inside the clip's
# scheduled brace windows only; pressing outside the window (e.g. while
# this foot's own pad is latched) bills 0.3x of the same scale — that
# reaction loads the fresh anchor toward its cheap release directions.
```

### motion_ref_height — task drive AS imitation (motionref8) + std 0.025  [L652-676]

```
# -- task drive AS an imitation term (2026-07-24, motionref8): the climb
# income follows the ghost. The free-standing height rewards ("any rise
# pays") funded every reference-ignoring gait — scramble, arms-only,
# legs-only — and each takeoff bought climb income with falling schedule
# adherence; three rounds of caps (per-step, flat budget, ref-profile
# ceiling) were patches on that split incentive. Now the payout is an exp
# kernel on (robot climb - reference climb) since settling: full on the
# profile (where RSI places every episode), decaying symmetrically when
# falling behind OR running ahead. std 0.05 ~ one step height: a one-step
# lag still earns, a runaway scramble earns nothing.
# (std param:)
# 0.83 step heights, the ratio the term was designed at (std 0.05
# against the old 60 mm step). Left at 0.05 for the 30 mm clip, one
# step moved the kernel only 0.6 sigma and the per-step climb
# gradient halved — measured iter-1400 basin: schedule match 0.98
# with net climb at ~5% of the ghost (ref12 hit 0.18 m/episode by
# iter 300 on the original ratio).
```

### motion_anchor_landing — weight 40, once per window (2026-07-30)  [L678-684]

```
# Landing accuracy on the anchor ladder (2026-07-30): the shape terms
# are root-relative and the engage gate checks timing only, so cycling
# in place collected nearly full imitation pay. This pays each scheduled
# latch once by where it LANDS (the reference anchors climb one step per
# cycle). Ceiling ~0.75/s at perfect landings (4 per 4.24 s cycle x
# 40 x dt) — meaningful against the ~4.5/s tracking ceiling without
# dominating; in-window relatch farming pays zero (once per window).
```

### motion_swing_approach (motionref17)  [L691-697]

```
# Dense approach shaping toward the landing rung (2026-07-30,
# motionref17): the landing bonus alone is a chicken-and-egg — reaching
# a rung one step UP needs the whole swing aimed there, and exploration
# never samples it (motionref16 landing income: 3-5% of ceiling). Pays
# the swing's progress toward ``ref_anchor_pos_w`` while the EE is
# scheduled OFF and released; ceiling 0.5/s per swinging EE, well under
# the mismatch bill so hovering released outside swing windows never
# profits.
```

### wall_distance_band_l2 — target_distance 0.173  [L713-714]

```
# Base standoff of the climb-zero v3 stance (1.5 - 1.327).
```

### physics-contract banner + ee_face_parallel_error  [L719-721]

```
# -- microspine physics contract. Two, three, and four anchors are all
# acceptable; only insufficient support is penalized. A reverse-shear
# unhook is free while safe; structural failures carry one-shot costs.
```

### engaged_tilt — tax threshold 3.0 / scale 5.0 (2026-07-31)  [L733-736]

```
# Engaged-tilt tax (recalibrated 2026-07-31, "4-deg" pass): capture-
# relative rotation priced from 3 deg so the tilt-peel cliff (DR
# 3.5-4.5 deg, reason 5) is felt as a gradient before it fires. The
# reference uses <= 1.1 deg — schedule-true execution pays nothing.
```

### microspine_support_loss — weight -4.0  [L748-750]

```
# -4.0 (was -3.0 pre-DeepMimic): the tracking terms raised the ceiling
# of positive income to ~4.5/s, and a one-anchor dangle still earning
# ~2.4-3.5/s of partial tracking must stay strictly net-negative.
```

### unsafe_microspine_release — weight -2.5 + scheduled-reverse story  [L757-779]

```
# -2.5 (was -1.0, raised 2026-07-21): motionref1's scramble gait ripped
# anchors at 3.5 Hz, 40% of them genuine tension pull-offs, and the
# per-event bill (~-0.4/s) never registered. Reverse-shear unhooks stay
# free by design — this prices only the failure modes.
# (params:)
# Reverse-shear unhooks stay free AT scheduled releases (the edge
# ramp zeroes the bill there) but cost 0.3x while the clip says
# ENGAGED: motionref16b farmed the free direction as a release
# valve — 43-45 latch-jiggle-unhook cycles per 30 s on the
# abandoned right foot, none of them priced (2026-07-30).
# Tilt-peel (reason 5) prices between tension (0.25) and ultimate
# (1.0): rotating a live patch off its grip is a structural failure
# the policy can always avoid (the reference needs <= 1.1 deg).
```

### torque_headroom — spike tax (motionref17b)  [L802-808]

```
# Spike tax above 70% of the deploy torque caps (2026-07-30,
# motionref17b): unpriced spike torque let the policy trade the
# weight-shift unhook for a near-saturation yank (0.45 -> 1.00 of the
# 58 Nm shoulder cap between model_2175 and model_4300, +7% climb).
# Threshold form leaves support duty and the gentle unhook untaxed; -8
# makes one saturated joint cost 8/s — a 1-2 step yank bills ~0.3-0.6,
# comparable to the timing income it shortcuts.
```

### action_rate_l2 — -0.1 (was -0.01)  [L823-828]

```
# -0.1 (was -0.01): at -0.01 chatter was economically free — model_3000's
# hovering hands buzzed the policy MEAN at 3-4 Hz, +-4-5 cm, ~40 mrad
# target flips per step (deterministic rollout, torque duty far from
# limits), which also blows through the 60 ms engage debounce window.
# Walk prices the same term at -0.25/-0.3; climbing keeps more headroom
# for the legitimately fast regrip reaches.
```

### dof_pos_limits / termination_penalty  [L830-831, L837]

```
# The target clamp is the safety guard; this term gives the policy a
# learning signal to stay away from the guard and from physical joint stops.
# RewardManager scales by dt, so -500 is a -10 one-shot failure cost.
```

## Terminations

### no_progress executioner obituary (2026-07-30, motionref17)  [L841-848]

```
# no_progress executioner REMOVED (2026-07-30, motionref17). Under the
# motionref16 economics it truncated every learning episode at ~8-13 s and
# priced failed regrip experiments at -10, which dismantled the very
# learning ladder motionref15 had climbed (dance -> choreography -> climb).
# The climb pressure is now carried by motion_ref_height +
# motion_anchor_landing + motion_swing_approach; if a dance-forever basin
# returns, reintroduce a curriculum-gated stall bar instead of a flat one
# (see mdp.no_progress_stall, kept for that purpose).
```

### bad_tracking — threshold 0.7  [L849-851]

```
# DeepMimic early termination: end rollouts whose pose left the reference
# entirely (fallen/tangled). Threshold is generous — a clean hang at the
# wrong phase scores ~0.1-0.2 rad mean error and survives.
```

## Metrics

### reseat-hold telemetry (R4)  [L862-864]

```
# R4 reseat-hold telemetry: clock-freeze duty cycle and how many EEs are
# pending. hold_fraction >> interval-implied duty means reseats are slow
# or timing out; ~0 with random_release on means holds resolve instantly.
```

## Curriculum — training-schedule doctrine (verbatim, incl. table)  [L919-965]

```
# ------------------------------------------------------------------------
# Training schedule — gravity ladder x tilt-peel anneal, one timeline
# (from-scratch rewrite 2026-08-02 evening, user call: the config must
# fully retrain from iter 0; the resume-era pinned variants live in the
# deploypd-v4-* run dirs and git history).
#
#    iter | gravity pool      | tilt-peel band (tax thresh/scale)
#   ------+-------------------+----------------------------------
#       0 | 45 (global +-5)   | 12-15 deg (10/8)  wide birth band
#    1200 |                   | 8-10 deg  (6.5/6)
#    1500 | 75 +-12 (63-87)   |
#    2400 |                   | 5-7 deg   (4/5)
#    3000 |                   | 3.5-4.5   (3/5)   calibrated, final
#    3550 | 85 +-3  (82-88)   |
#    4200 | 88 +-2  (86-90)   |
#    4800 | 90 exact          |
#
# The iters transcribe the ONE validated trajectory: v3 ran 0->3300 on
# exactly this 45/75/tilt prefix (from birth, full DR + pd100), and the
# v4 bridge campaign took the same lineage 3300->5050 through
# 85/88/exact-90 to the ship gate (90-deg cal-band exams
# 0.962/0.961/0.963 across three consecutive panels, zero falls).
# Design rules baked into the numbers:
#   - gravity is a function of training time alone (2026-07-21:
#     graduation-by-height once promoted a 20x-overspeed scramble), and
#     every hop overlaps its neighbor's pool (63-87 / 82-88 / 86-90 /
#     90) so zero-compression arrives in-distribution — v3 died on the
#     unbridged 75+-12 -> 88+-2 cliff (weld-hands basin: at 90 a hand
#     swing leaves 3 anchors on pure shear+tension, 12.3% margin at the
#     F0=400 worst draw). The final stage runs exactly 90; jitter there
#     would mix in >90 overhangs, a different task.
#   - the tilt anneal reaches the honest physics (3000) before the wall
#     gets steep (3550): every steep rung trains AND examines on the
#     calibrated band. Birth needs the wide band — cold starts on
#     3.5-4.5 peel apart within ~1 s (motionref18: zero learning).
#   - anneal steps and rung boundaries stay >=300 iters apart (fresh2b:
#     never stack two shocks inside a consolidation window).
#   - the iters are CEILINGS: drive by mastery gates (precise-angle exam
#     >=0.95 x2 zero falls; per-EE occ_pol-occ_ref < +0.04 is the weld
#     early warning) via kill/edit/resume, and when pulling a rung IN,
#     push the later rungs OUT (the 88 rung auto-fired at 4200 mid-soak
#     once; it survived only by luck of near-readiness).
#   - examine ALL rungs every cycle (periodic_eval.sh probes
#     75/85/88/90): steep-only pools erode shallow angles — accepted
#     under the ckpt-portfolio doctrine (bank each rung's graduate),
#     but the erosion must stay measured, never silent.
# steps_per_iter must match rl_cfg num_steps_per_env.
```

## Observations

### motion_command obs — 16 dims  [L1037-1040]

```
# Cycle phase clock (cos/sin) + the clip's engage schedule chart: current
# per-EE engage bits plus each EE's countdown to its next release and next
# regrip (2026-07-22), plus the per-foot toe-brace flags (2026-07-29).
# 16 dims total.
```

### critic group — asymmetric actor-critic design  [L1046-1050]

```
# Asymmetric actor-critic: the critic group is the actor group plus
# privileged microspine state (hold margin, anchor loads, forced releases)
# that a real robot cannot sense. Term cfgs are re-instantiated so the two
# groups never share mutable cfg objects. Built after all actor-term edits
# above so the copies pick them up.
```

### student group — deployable contract design  [L1074-1087]

```
# Deployable observation contract for the distilled student. Every term
# is hardware-measurable: IMU gyro + gravity, joint encoders, the
# policy's own action, and the motion command — a pure 50 Hz clock
# lookup exported alongside the policy. Deliberately absent: base_lin_vel
# and height_above_spawn (no estimator on the robot), the microspine
# latch state (no engagement sensing; the reference schedule inside
# motion_command plus proprioceptive history carries it), the constant
# velocity_command, and prev/prev_prev_action (the history covers them).
# 89 dims per frame x history 10 = 890; layout is term-major with frames
# oldest->newest (the export sidecar documents the C++ contract).
# pad_normal_force (2026-08-03) is the one non-proprioceptive channel:
# the gimbal-mounted load cell the sensing co-design provides — it gives
# the student a physical engage/loss estimator instead of pure schedule
# faith, and it is the deploy-side trigger for the R4 clock pause.
```

### student history_length = 50 (Route-B step 0 follow-up)  [L1134-1138]

```
# 50 frames = 1.0 s (2026-08-03, Route-B step 0 follow-up): the 10-frame
# window (0.2 s) cannot span the 0.6 s reengage cooldown, so latch
# timers were uninferable; 1 s also covers debounce + typical reseat
# and gives the height/slow-factor estimators a usable integration
# window. 89 dims/frame x 50 = 4450 — the packer contract.
```

## Play / Distill / Polish

### play overrides — full original notes  [L1143-1177]

```
# The viewer shows one world. Keeping the training batch of 4096 here
# simulates thousands of invisible robots and is the main cause of
# sub-real-time playback unless callers remember ``--num-envs 1``.
# Evaluation and diagnostic rollouts are deterministic physics probes:
# drop the DR events entirely (neutral ranges would still overwrite XML
# frictions under operation="abs").
# Debug transition logging forces CPU-GPU synchronization in the physics
# step. Keep normal play fast; it can still be enabled explicitly on a
# locally constructed config when diagnosing the state machine.
# Evaluation uses the final task directly. play.py's gravity resolver
# (CLI override or checkpoint-iteration lookup) can still replace this
# single stage. Jitter off: evaluation and diagnostic rollouts (engage
# raster) must be deterministic at the pinned angle.
# Evaluation always runs the CALIBRATED tilt-peel band (the annealing
# ladder is a training aid, not physics).
# Evaluation starts at the cycle's canonical stance instead of a random
# RSI phase, so rollouts read as one continuous climb from frame 0.
```

### distill — no-drill rationale (2026-08-03)  [L1199-1203]

```
# No release drills here (2026-08-03): DAgger/estimator rollouts want the
# undisturbed schedule — drills + the reseat freeze add label variance for
# the student and, in RL settings, subsidize welding (the polish-v1
# post-mortem). The recovery SKILL still transfers through imitation of
# the press-on-schedule behavior (model_250 already passed 2/2 drills).
```
