"""PHYBot C2 walk->climb transition (mount) environment.

Parallel to the climb task, not a fork of it: the config is BUILT by calling
the climb builder and mutating the returned object, so observations, rewards,
PD table and the s2r DR package — and hence the 150/170-dim contract — are
inherited verbatim and climb checkpoints warm-start strict. Everything
transition-specific lives in THIS package; climb_wall's sources are never
modified.

The task: the robot stands at the wall base (delivered by the walk policy),
replays the acyclic mount prologue — latch hands low, mount the feet, walk
the hands up — and holds the cycle's frame-0 stance for the climb policy to
take over. The robot is BLIND to its delivery error: standing RSI spawns
scatter the base pose while the reference and its wall anchors stay
canonical, so the policy must absorb the offset through contact and square
the yaw back up during the mount.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_STANDING_JOINT_POS,
  configure_phybot_c2_floor_geom,
)
from scr_phybot_mjab.tasks.climb_transition import mdp
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)

_MOTIONS_DIR = Path(__file__).resolve().parents[4] / "assets" / "motions"
# Acyclic walk->climb mount prologue (scripts/gen_mount_clip.py): stand at
# standoff 0.23 -> lateral arm raise -> hands latch LOW (1.44 m, standing
# reach ceiling) -> feet mount -> toe braces -> two hand steps to 1.53 m ->
# handoff_hold at EXACTLY the cycle's frame-0 stance shifted -0.2234 m.
# Gates 2026-08-09: floor-aware static audit all-frames PASS, handoff joint
# delta vs state-1 4e-08 rad, contract tests in
# tests/microspine/test_mount_clip.py.
_MOUNT_MOTION_FILE = _MOTIONS_DIR / "climb_mount_prologue.npy"

# Band-teacher matrix (docs/plans/ 2026-08-11): per-band mount tasks run a
# discrete grid of NATIVE-angle clips (gen_mount_clip --wall-angle-deg,
# stored de-rotated to the canonical frame) and cover the band by nearest-
# native selection + small runtime rotation. 90 stays the single canonical
# clip above; 88+-2 rides one native-88 clip (+-2 rotation). The 80 band's
# grid is {80, 84}: a76 is PHYSICALLY infeasible (the standing two-step
# mount's balance boundary sits at ~80 deg — CoM crosses the toe line with
# no solution; 2026-08-11 generation report), so the band80 MOUNT training
# window is clamped to [78, 85] (center 81.5 +-3.5) and the deploy story
# for walls below 78 deg is a future slab-specific choreography. The CYCLE
# band teacher still trains the full 75-85 pool.
_BAND_CLIP_GRIDS: dict[float, tuple[float, ...]] = {
  80.0: (80.0, 84.0),
  88.0: (88.0,),
}
# Mount-side wall_angle stage per band: (center, jitter).
_BAND_ANGLE_STAGES: dict[float, tuple[float, float]] = {
  80.0: (81.5, 3.5),  # [78, 85] — the mount-feasible slice of 80+-5
  88.0: (88.0, 2.0),
}
# Cosmetic standing base x per band CENTER clip (RSI overwrites on every
# reset). The generator's standoff is the HORIZONTAL distance to the wall
# base line (a perpendicular standoff would push the toes through the
# surface at z=0 on a tilted wall): a90/a88 stand at 1.5-0.23, the 80-band
# clips at 1.5-0.19 (their retuned standoff).
_BAND_STANDING_X: dict[float, float] = {80.0: 1.31, 88.0: 1.27}


def _mount_clip_path(angle_deg: float) -> Path:
  return _MOTIONS_DIR / f"climb_mount_prologue_a{int(round(angle_deg))}.npy"

# Blind-delivery spawn scatter: walk hands the robot over with tape-mark
# accuracy, and the mount policy must absorb at least this much base-pose
# error with no exteroception. Initial values; the mount-raster basin
# calibration will refine them against the real delivery spec.
_TRANSITION_SPAWN_XY_NOISE = (0.05, 0.03)  # +-wall-distance, +-lateral (m)
_TRANSITION_SPAWN_YAW_NOISE = 0.0873  # +-5 deg (rad)
# RSI spawn bias (v2, iter ~1200): half the resets land in a biased window
# so the gate skill gets real experience share; plain uniform gave the
# standing stretch ~7% and the run plateaued into "stand five seconds,
# refuse the raise, topple".
_TRANSITION_SPAWN_EARLY_FRACTION = 0.5
# v4 (iter ~5100): the window re-aims from [0, first-engage) to the
# place -> set LATCH stretch. v2/v3 taught standing, horizontal arms and
# leaning in, but neither the reach-up (narrow exp tracking kernels pay ~0
# beyond ~1 rad — no local gradient across the horizontal -> overhead gap)
# nor the press-to-latch ever ignited. Spawning AT the latch moment (arms
# on the anchors, one 3 N press away) trains the press directly; the
# reach-up then only has to bridge into a state the policy already values.
# v8-from0 (2026-08-11 night): the window schedule gains a SURVIVAL stage.
# v7 started the reverse curriculum at the latch ending and the window
# probe (window_650.mp4) showed why that fails from zero: spawned with
# arms overhead, the climb-seeded policy topples inside the 25-step
# termination grace (22 resets / 12 s, 6 latch edges in 22 window births)
# — the latch stretch cannot be practiced by a policy that cannot yet
# STAND there. v4's latch window only worked because v2/v3 had spent
# ~2000 iters teaching exactly that. The schedule now retraces the
# validated v2 -> v4 ladder and then continues widening backward:
# survive-and-reach, latch, bridge the raise, full chain. Iters are
# ceilings; the static tuple below is stage 0; drive by lane mastery.
_TRANSITION_SPAWN_BIAS_SEGMENTS = ("ground_stand", "hand_raise", "hand_place")
# 2026-08-15 staggered latch: the clip now lands the LEFT hand first
# (hand_set_left, 1.5 s single-weld classroom), closes the RIGHT under that
# brace (hand_place_right), then holds both (hand_set_hold). The latch
# window covers the whole staggered stretch; RSI births inside
# hand_set_left/hand_place_right seed ONE hand — the cycle's braced-regrip
# condition, imported.
_TRANSITION_LATCH_SEGMENTS = (
  "hand_place",
  "hand_set_left",
  "hand_place_right",
  "hand_set_hold",
)
_TRANSITION_FOOT_SEGMENTS = (
  "left_foot_mount",
  "foot_switch_hold",
  "right_foot_mount",
)
_TRANSITION_SPAWN_WINDOW_STAGES = [
  {"iter": 0, "segments": ("ground_stand", "hand_raise", "hand_place")},
  # v27: the latch stage advanced 1000 -> 550 (mastery call, the ladder's
  # iters are ceilings): v26 grew first latch sparks (0.036 Hz) by ~520
  # with sigma already at 0.17 — the window moves to the sparks while
  # exploration is still alive.
  # v61 fresh-era correction (08-18 night, probe on model_425): the 550
  # ceiling assumed sparks existed by then; the fresh springfield run has
  # first_weld = 0 at 432 and its flail rips latch-window seeds in 6 steps
  # (584/592 LH, 320/324 RH, tension+lateral at 380-456 N). Window follows
  # the sparks (v27 doctrine) — pushed to 1200, advance earlier by mastery
  # call once first_weld ignites.
  # v61c (iter-1334 autopsy): the full latch window opened spark-less and
  # scheduled_contact_loss became the dominant killer (79% of terminations
  # — every seeded birth's gifts die under flail, 2 s grace, execution).
  # Stage 1 shrinks to the SPARK NURSERY: hand_place births are unseeded
  # near-wall stances (press income 4x'd there in 130 iters) with no
  # gifted anchors for the executioner to bill. Seeded births return with
  # the 2200 stage; the executioner itself matures on the grace ladder.
  {"iter": 1200, "segments": ("hand_place",)},
  # v62b (08-19 08:0x, decision point A): still spark-less at 1800 — the
  # seeded-latch return waits for ignition (the window follows the sparks;
  # the v61b lesson about ceilings slipping past spark-less is not repeated).
  # v63c: the seed/foot/full-chain trio shifts +600 (3000/3400/3800 ->
  # 3600/4000/4400) — the window follows the sparks (v27/v62b doctrine);
  # v63 ignition runs later than v62's 2332 and seeded classrooms must
  # not open on a spark-less policy (v61b lesson). Spacing preserved;
  # the 5200 concentration stage stays (4400 + 800, same gap as before).
  {"iter": 3600, "segments": ("hand_raise",) + _TRANSITION_LATCH_SEGMENTS},
  # v54 foot classroom: births inside the foot mounts arrive with both
  # hands (and later the left foot) SEEDED — the crane condition the
  # hands trained under, now for the legs. The v26 ban on seeded-hanging
  # spawns was written for a NEWBORN policy; a two-hand holder (v53 LH
  # occ 0.7) is exactly who these classrooms were waiting for.
  {
    "iter": 4000,
    "segments": _TRANSITION_LATCH_SEGMENTS + _TRANSITION_FOOT_SEGMENTS,
  },
  {
    "iter": 4400,
    "segments": ("ground_stand", "hand_raise")
    + _TRANSITION_LATCH_SEGMENTS
    + _TRANSITION_FOOT_SEGMENTS,
  },
  # v62e6 B (user call, 08-19 evening): approach+RH CONCENTRATION at
  # 0.6 g. Engage graduates before gravity climbs, and the missing
  # courses are DENSITY courses — the full-chain pool spends a sliver of
  # births on the approach stroke and on the LH-welded/RH-free scene
  # (hand_place_right). v59's foot playbook verbatim: concentrate, flip,
  # return to the full chain (the return re-runs the ladder as the exam).
  {
    "iter": 5200,
    "segments": ("hand_place", "hand_set_left", "hand_place_right"),
  },
  {
    "iter": 6700,
    "segments": ("ground_stand", "hand_raise")
    + _TRANSITION_LATCH_SEGMENTS
    + _TRANSITION_FOOT_SEGMENTS,
  },
  # v59 foot classroom CONCENTRATION (user call, 08-17 evening): the hands
  # graduated (RH match 1.0-band, seeds 64/64) but the feet sat at occ
  # 0.003-0.005 with their lesson diluted to a sliver of the full-chain
  # window. Reverse-curriculum doctrine: concentrate births on the
  # unmastered stretch. hand_set_hold = the two-hand entry stance;
  # brace_hold = the FEET's set_hold (born with foot seeds, learn to keep
  # them — the crane classroom that flipped RH). Full chain returns at
  # 12600 (+2000, a ceiling driven by LF mastery as always).
  {
    "iter": 10500,
    "segments": ("hand_set_hold",) + _TRANSITION_FOOT_SEGMENTS + ("brace_hold",),
  },
  {
    "iter": 12600,
    # brace_hold rides along (v59b): the four-weld hold class had NOT
    # flipped at graduation time (62/64 still lost) — and the hands'
    # precedent is that set_hold never left the window once it entered.
    # An unmastered classroom does not close.
    "segments": ("ground_stand", "hand_raise")
    + _TRANSITION_LATCH_SEGMENTS
    + _TRANSITION_FOOT_SEGMENTS
    + ("brace_hold",),
  },
  # v60 RH-specialty window (08-18). NOTE the clock: stage iters are
  # CUMULATIVE learning iterations — the resume path restores
  # common_step_counter (v59b launched at 12450 and its spawn lane read
  # 10500 -> 12600; the early ladder never replays), so a new stage
  # activates by out-numbering the resume iteration. The two unmastered
  # courses concentrate: RH's OWN weld+seat (hand_set_left /
  # hand_place_right births arrive with LH welded and RH's edge 1-2.4 s
  # out; hand_set_hold adds the two-hand hold + post-rip re-welds), and
  # brace_hold (the four-weld class, 61/64 lost at v59b close). RF's
  # own-weld course stays queued behind RH — one specialty at a time
  # (the v59 LF->RF sequencing doctrine).
  {
    "iter": 16350,
    "segments": (
      "hand_set_left",
      "hand_place_right",
      "hand_set_hold",
      "brace_hold",
    ),
  },
  # Full-chain return: a +3000 ceiling, advanced by mastery call as
  # always (lanes to watch: RH self-weld lifetime crossing the 12.5-step
  # ramp, first_charge income, fullrun past 19.3 s).
  {
    "iter": 19500,
    "segments": ("ground_stand", "hand_raise")
    + _TRANSITION_LATCH_SEGMENTS
    + _TRANSITION_FOOT_SEGMENTS
    + ("brace_hold",),
  },
]
# Press-bootstrap (v4, paired with the window): the scheduled-press shaping
# at the climb-parity weight (0.4, income ~0.002) cannot teach pressing
# from scratch — the cycle never needed it, its teacher already pressed.
# Boosted 7.5x at v4 launch and annealed back toward parity by curriculum
# (iterations counted from the v4 process start; the env step counter
# resets on every launch).
# v10: the press bootstrap is RETIRED (flat climb parity). Its mission —
# "never presses at all" (v4 ignition) — is accomplished; the v9 dual-band
# probe showed its 7.5x price now funds SLAMMING instead: latches engage
# hot/tilted and die in 3-4 steps through ultimate (train band) or
# tilt-peel (cal band), and the -8 forced-release bill teaches latch
# AVOIDANCE, not gentleness. press_norm_n stays 12 N (user call): the
# saturation was never the weapon, the boost price was.
_TRANSITION_PRESS_BOOTSTRAP_STAGES = [
  # v56 mastery call: ignition completed at v54@240 — the boosted wage's
  # remaining effect was funding the wall-grind (press-without-weld pay).
  # Down to 0.6 immediately, climb parity with the full-chain stage.
  {"iter": 0, "weight": 0.6},
  {"iter": 4400, "weight": 0.4},
]
# v10 engage-quality gate (user-approved): the stock acceptance threshold
# engage_parallel_cos_threshold=0.25 admits welds up to ~75 deg of pad
# tilt — the v9 dual-band probe showed cold latches engaging hot/tilted
# and dying in 3-4 steps (ultimate at the wide peel band, tilt-peel at
# the calibrated one), which teaches latch AVOIDANCE ("bad engagements
# happen then get billed" -> "don't engage"). The gate flips that to
# "bad engagements don't happen": only near-flat pads weld, so every
# latch that exists is survivable and its value is learnable. Two s2r
# arguments: the real pad's passive self-centering gimbal makes near-flat
# engagement a HARDWARE property the rigid sim pad lacks; and the gate is
# a training aid annealed OFF before graduation (final phase trains under
# the honest stock physics) — same doctrine as the tilt-peel ladder.
# v11 (user call): the ladder runs EASY -> HARD — the strict-first 4-deg
# gate passed ZERO real welds in its first ~200 iters (signal starvation);
# starting loose floods the policy with welds of varying flatness and the
# contrast teaches "flatter lives longer", with the requirement tightening
# as skill grows. Two constraints shape the numbers: each stage sits just
# INSIDE the tilt-peel band's lower edge at that time (a weld looser than
# the band peels on its next step — admitted-then-executed teaches
# nothing), and the final stage still returns to STOCK acceptance so
# graduation physics stay honest.
_TRANSITION_ENGAGE_GATE_STAGES = [
  # v30: the whole ladder shifts +2500 — the counter drifted through two
  # sigma re-mints and the gate tightened to 7 deg BEFORE ignition ever
  # happened (v29 engage rate hit exactly 0.0000 at ~1950 with the gate
  # closed to casual contact). The ladder's premise is "tighten as skill
  # grows"; hold 10 deg until the skill exists.
  {"iter": 0, "max_tilt_deg": 10.0},  # band 12-15
  {"iter": 3700, "max_tilt_deg": 7.0},  # band 8-10
  {"iter": 4900, "max_tilt_deg": 4.5},  # band 5-7
  {"iter": 5500, "max_tilt_deg": 3.5},  # band 3.5-4.5 (calibrated)
  {"iter": 7000, "max_tilt_deg": None},  # gate off (stock threshold)
]
# v30 rip-bill holiday: with hover paid and every latch dying to sigma
# noise in ~0.1-0.4 s, the rip bills (-2.5/event + tilt tax) made
# "never touch" the rational optimum — the risk premium exceeded any
# income the latch could earn before its value is even learnable. The
# bills drop to token levels for the ignition stretch and anneal back to
# full price as holds consolidate (same doctrine as the tilt/gate
# ladders: training aids retire before graduation).
_TRANSITION_RIP_BILL_STAGES = [
  # v55 mastery call (user): the ignition holiday's mission completed at
  # weld@240/hold@400 — from there it only subsidized the churn texture
  # (weld-rip cycling every ~1.4 s across five video generations). Rips
  # bill -1.0 from step 0; full price arrives with the full-chain stage.
  {"iter": 0, "weight": -1.0},
  {"iter": 4400, "weight": -2.5},
]
_TRANSITION_TILT_BILL_STAGES = [
  {"iter": 0, "weight": -0.25},
  {"iter": 4500, "weight": -0.6},
  {"iter": 6000, "weight": -1.0},
]
# v61 fresh-era alignment (08-18 night): the -25 seed-rip bill is a v52+
# invention priced for a MATURE holder; the original lineage crossed its
# early latch stages with no seed bill at all. A fresh policy that cannot
# hold yet would bleed unavoidable -25s the moment seeded births open —
# the 08-13 inversion shape. The bill re-anneals with skill instead; full
# price returns by the seed-classroom era (ceilings, mastery calls).
_TRANSITION_SEED_RIP_BILL_STAGES = [
  {"iter": 0, "weight": -2.5},
  {"iter": 2500, "weight": -10.0},
  {"iter": 4800, "weight": -25.0},
]
# Wide-basin hand tow (v5): the cycle's approach funnel (std 0.05) pays ~0
# beyond ~15 cm, and the mount's raise must cross a ~0.5 m horizontal ->
# overhead gap — the v4 plateau's reward desert. A second, WIDE kernel on
# the hands only (feet would collect stand-still income at 0.35 m and get
# paid for premature lifts) hands exploration a gradient from ~1 m out:
# 0.006 standing hang -> 0.21 horizontal arms -> 0.94 at the place pose.
# The narrow funnel stays untouched (two-scale kernel); LadderMan's contact
# reward runs tau=0.3 (sigma~0.55 m) for the same reason.
_TRANSITION_APPROACH_FAR_STD = 0.4
_TRANSITION_APPROACH_FAR_WEIGHT = 0.5
# v21 full-body fence rosters. The v16-v20 mole family morphed through
# whatever the fence's geom list didn't name (head -> shoulder -> knee +
# dangling arm, each pose folded around the billed parts) — so the fence
# now names every link that never legitimately approaches the wall before
# first engage. Clearances come from the measured reference envelope
# (ref_wall_envelope, 2026-08-13): body links stay >= +13.3 cm through
# stand/raise/place, so 10 cm never taxes the reference even under the
# +-5 cm delivery scatter; the upper arm dips to +8.4 cm at place, so its
# own term runs a 4 cm ring. Hands, forearms and elbows are UNFENCED (the
# place pose lays them on the wall, elbow +2.9 cm), as are feet/toes (the
# wall-base stance is legal and toes carry real contact billing).
_TRANSITION_FENCE_BODY_GEOMS = (
  "neck_pitch_collision",
  "torso_collision",
  "base_link_collision",
  "left_hip_roll_collision",
  "right_hip_roll_collision",
  "left_hip_yaw_collision",
  "right_hip_yaw_collision",
  "left_knee_collision",
  "right_knee_collision",
  "left_shoulder_roll_collision",
  "right_shoulder_roll_collision",
)
_TRANSITION_FENCE_ARM_GEOMS = (
  "left_shoulder_yaw_collision",
  "right_shoulder_yaw_collision",
)
# Asymmetric arm tracking (backup knob, default OFF): under blind-delivery
# scatter the clip's arm path is wrong by the spawn offset while the wall
# anchors stay environment-true — strict arm tracking fights the reach the
# contact terms ask for. ON: arms track at weight/2 and std*2 while
# legs+waist+root stay strict (LadderMan Eq (1)).
_TRANSITION_ARM_JOINT_KEYWORDS = ("shoulder", "elbow")
_TRANSITION_HAND_EE_INDICES = (0, 1)  # microspine order: LH, RH, LF, RF
# v58 foot ladder: the ignition bonuses and the edge-gated approach tows
# cover ALL FOUR limbs — the foot classroom opened at v54 but the ladder
# that ignited the hands (touch -> weld -> hold, plus the tow) was never
# extended, leaving the executioner as the feet's only teacher (probes:
# deadline-welding at exactly the 2 s grace). The edge gate (2.0 s lead)
# keeps stand-still/premature-lift income impossible — the v14 reason for
# hands-only no longer applies to the progress-paid, edge-gated shape.
# The WIDE TRACKING kernels stay hands-only (they are reach teachers).
_TRANSITION_ALL_EE_INDICES = (0, 1, 2, 3)
_TRANSITION_FOOT_EE_INDICES = (2, 3)


def _transition_spec_fn(spec: mujoco.MjSpec, base_fn) -> None:
  """The climb scene spec plus a contact-enabled floor.

  The climb builder's own spec_fn disables the terrain plane (gravity-tilt
  curriculum replaces the floor there); running it first and re-enabling
  contact afterwards means the later write wins without touching climb's
  closure. The mount stands on the floor until the hands take the wall —
  and latching is unaffected: the microspine sensors are pad-vs-WALL pairs,
  floor contact never reaches them.
  """
  base_fn(spec)
  configure_phybot_c2_floor_geom(spec, enable_contact=True)


def phybot_c2_climb_transition_env_cfg(
  play: bool = False,
  domain_rand: bool = True,
  student_obs: bool = False,
  asym_arm_tracking: bool = False,
  wall_band: float | None = None,
  tilt_anneal: bool = False,
  engage_gate: bool = False,
  stand_first: bool = False,
  ground_spawns: bool = False,
) -> ManagerBasedRlEnvCfg:
  """walk->climb transition (mount): stand at the wall, latch all four.

  Deltas against the climb cfg it is built from:

  * motion command swapped for :class:`mdp.MountMotionCommandCfg` on the
    mount prologue — clock clamps on the handoff frame, engage tables are
    acyclic, standing spawns carry the blind-delivery scatter;
  * floor contact enabled (spec_fn wrap);
  * no gravity ladder (default gravity IS the vertical-wall operating
    point) and no tilt-peel annealing (the static config already carries
    the calibrated band);
  * no random-release drill in v1 — an R-series rung for later; this also
    removes the only reseat_hold producer, so the clock cannot be frozen
    short of the handoff frame;
  * climb-rate command pinned to zero (a mount has no climb rate);
  * wide-basin hand tow stacked over the narrow approach funnel (v5 sigma
    therapy for the reach-up reward desert);
  * ``asym_arm_tracking`` swaps the joint/EE tracking terms for strict-leg
    / loose-arm splits (backup knob, default OFF keeps climb parity);
  * ``wall_band`` (80.0 or 88.0) switches to the band-teacher variant:
    native-angle clip grid + band-pinned wall_angle curriculum; None keeps
    the vertical-wall task on the canonical clip, bit-for-bit.
  """
  cfg = phybot_c2_climb_wall_env_cfg(
    contact_model="wang2017",  # preserve the separate legacy spring-field recipe
    play=play,
    domain_rand=domain_rand,
    student_obs=student_obs,
    robustness_release=False,
  )

  base_spec_fn = cfg.scene.spec_fn
  cfg.scene.spec_fn = functools.partial(
    _transition_spec_fn, base_fn=base_spec_fn
  )

  if wall_band is not None:
    grid = _BAND_CLIP_GRIDS[wall_band]
    band_motion_file = str(_mount_clip_path(wall_band))
    # Play evaluates the band CENTER deterministically on the single
    # center-native clip (single-clip machinery + hard-pinned wall).
    band_clips = (
      ()
      if play
      else tuple((a, str(_mount_clip_path(a))) for a in grid)
    )
  else:
    band_motion_file = str(_MOUNT_MOTION_FILE)
    band_clips = ()

  # Swap the command IN PLACE (assignment to an existing key preserves the
  # test-enforced command order). The climb fields are carried over; play
  # mode mirrors the base play block's determinism and adds spawn-noise
  # zeroing — evaluation delivers the canonical frame-0 stance.
  cycle_motion = cfg.commands["climb_motion"]
  cfg.commands["climb_motion"] = mdp.MountMotionCommandCfg(
    entity_name=cycle_motion.entity_name,
    motion_file=band_motion_file,
    affordance_command_name=cycle_motion.affordance_command_name,
    ee_body_names=cycle_motion.ee_body_names,
    sampling_mode="start" if play else "uniform",
    joint_pos_noise=0.0 if play else cycle_motion.joint_pos_noise,
    spawn_xy_noise=(0.0, 0.0) if play else _TRANSITION_SPAWN_XY_NOISE,
    spawn_yaw_noise=0.0 if play else _TRANSITION_SPAWN_YAW_NOISE,
    spawn_early_fraction=0.0 if play else _TRANSITION_SPAWN_EARLY_FRACTION,
    spawn_bias_segments=() if play else _TRANSITION_SPAWN_BIAS_SEGMENTS,
    # Stand-first: the uniform RSI half draws standing frames only — with
    # the engage economy offline a wall-hanging spawn has no income channel
    # to hold itself and can only rip and die (v17: half the batch).
    # ground_spawns (v26): the same restriction under the FULL economy and
    # a marching clock — the v25 one-pot verdict: seeded-hanging spawns fed
    # a fresh policy stall at 1.4-1.6 Hz rips (the v8 pattern) while their
    # gradients diluted the standing skill to zero timeouts. Hanging
    # experience should arrive through ACTUAL mounts first (on-policy, from
    # states the policy reaches); the seeded-RSI half returns by widening
    # once the latch ignites.
    spawn_disengaged_only=(stand_first or ground_spawns) and not play,
    # v37 micro-window, RETIRED at v60: the quota pinned a quarter of ALL
    # births to the LEFT hand's pre-weld 0.3 s (frames 495-510) — outside
    # the v60 specialty window, and a lesson LH graduated long ago.
    spawn_edge_fraction=0.0,
    angle_clip_files=band_clips,
    # Band clips carry true native-world geometry; the cycle's vertical-
    # height spawn compensation solves a different problem and would lift
    # standing frames off the floor. Exactly zero at 90 either way.
    spawn_match_vertical_height=(wall_band is None),
  )

  # v13 gentle-band pay (user call, knob (a)): the press income saturates
  # at 4 N — the mount's reference press is 3 N. v28 replaces the func with
  # the gentle-ONLY variant: pay is also ZERO above the 8 N slam-tax
  # threshold, so [3, 8] N is the only paid contact (the saturating shape
  # paid a 250 N slam the full wage — the v27 probe's 70-252 N hammering).
  press_cfg = cfg.rewards["motion_scheduled_press"]
  cfg.rewards["motion_scheduled_press"] = RewardTermCfg(
    func=mdp.motion_scheduled_press_gentle,
    weight=press_cfg.weight,
    # v29: the pay window widens 0.3 -> 0.8 s of lead — the v28 night
    # shift measured zero gentle presses with the 0.3 s window (the seat
    # must be findable while the hand is still settling, not only in the
    # last 15 frames).
    params={
      "command_name": "climb_motion",
      # v55 (user call): the seating target firms up 8 -> 20 N — the
      # braced-stagger presses can afford real preload, and the 13.5 N
      # median the policy already runs sat at the old band's top edge.
      "press_peak_n": 20.0,
      "press_band_hi_n": 28.0,
      "overdrive_cutoff_n": 45.0,
      "regrip_lead_time_s": 0.8,
      "anchor_radius_m": 0.15,
      # v56: the wage is time-boxed to [lead, +2 s] around each rising
      # edge (matches the contact law's grace) — unlimited it funded the
      # visible wall-grind career. New limbs keep their ignition bridge.
      "wage_window_s": 2.0,
    },
  )
  # v29 wide basin, re-armed: v14 retired the position-paid wide tow as
  # unfixable rent (slam -> jam -> hover), but the v28 gates change that
  # verdict's premises — lead-gated (no standing-prefix rent) and
  # press-cutoff (no jam rent), a position basin is bounded to ~2-3 s of
  # hover pay per episode. The night-shift blocker it answers: episodes
  # born WITH hands at the anchors fold the arms down (balance prior) and
  # the progress tow then reads retreat (-0.0009 lane) — a sustained
  # pull-back basin must exist inside the place window.
  cfg.rewards["motion_swing_approach_wide"] = RewardTermCfg(
    func=mdp.motion_swing_approach_masked,
    weight=0.5,
    params={
      "command_name": "climb_motion",
      "std": 0.3,
      "ee_indices": _TRANSITION_ALL_EE_INDICES,  # v58 foot ladder
      "press_cutoff_n": 6.0,
      "edge_lead_time_s": 2.0,
    },
  )

  # v28 tow honesty: the narrow funnel inherited `active = ~ref & ~robot`
  # from the cycle, where ref goes false ONLY inside swing windows — in
  # the mount's 5.6 s all-false standing prefix it became an unconditional
  # four-EE magnet toward the wall anchors: position rent with no press
  # cutoff (the hand-jam-and-freeze equilibrium) and a standing wage for
  # hovering the lifted foot at its wall anchor (the toe-prop tripod).
  # Both tows now pay only within edge_lead_time_s of that EE's scheduled
  # rising edge, and the narrow funnel gains the same 6 N free-space
  # cutoff the far tow got in v12. Arm PATH guidance during stand/raise
  # belongs to the joint/EE tracking terms (which teach the lateral raise).
  narrow = cfg.rewards["motion_swing_approach"]
  cfg.rewards["motion_swing_approach"] = RewardTermCfg(
    func=mdp.motion_swing_approach_masked,
    weight=narrow.weight,
    params={
      "command_name": "climb_motion",
      "std": narrow.params["std"],
      "ee_indices": (0, 1, 2, 3),
      "press_cutoff_n": 6.0,
      "edge_lead_time_s": 2.0,
    },
  )

  # v28 toe-prop eviction: off-schedule toe press was billed at 0.12/s
  # max (0.3 x weight 0.4) while buying a 79-1124 N support — the tripod
  # crutch's whole rent. The scheduled brace wage is untouched.
  # v45 (user doctrine: force rewards pay by closeness to target, never
  # "more is better"): the toe wage swaps to the target-tracking variant —
  # on-schedule pay peaks at the audit's ~16 N brace and decays either
  # side (the saturating original paid a 345 N jam like the choreographed
  # brace); off-schedule billing unchanged.
  cfg.rewards["motion_toe_brace"] = RewardTermCfg(
    func=mdp.motion_toe_brace_target,
    weight=0.4,
    params={"command_name": "climb_motion", "off_schedule_cost": 2.5},
  )

  # v32 two-scale TRACKING kernels (user diagnosis: walk/squat/cycle learn
  # because their states never leave the imitation kernels' support; the
  # mount's resting state does). The narrow kernels (std 0.2 / 0.1) are
  # numerically DEAD at arms-down — the raise sits ~1.2 rad away and
  # exp(-1.44/0.04) ~ 3e-16, so an RSI birth that drops its arms once has
  # left the reference's pull forever (the v3 "reward desert", patched in
  # v5 for hand-to-anchor DISTANCE but never for the joint-space raise).
  # Wide kernels restore pull from anywhere: arms-down 0.05 -> horizontal
  # 0.48 -> raised 0.92 (std 0.7). Ungated — ignition teachers, not
  # precision judges; the narrow kernels keep owning precision.
  cfg.rewards["motion_joint_pos_wide"] = RewardTermCfg(
    func=mdp.motion_joint_pos_error_exp_masked,
    weight=2.5,
    params={
      "command_name": "climb_motion",
      "std": 0.7,
      "joint_keywords": _TRANSITION_ARM_JOINT_KEYWORDS,
      # v40: the ignition teachers retire INTO the gate (user catch: the
      # ungated wide kernels diluted the engage-mismatch discount to half
      # the pose income, halving the hold-vs-rip margin the gate was
      # designed to create). Pre-engage the gate never bites: the
      # schedule is empty, so an un-engaged robot has no mismatch (and an
      # EARLY engager is correctly discounted); it starts working exactly
      # at the engage phase.
      "engage_gate_floor": 0.25,
      "engage_gate_grace_fraction": 0.15,
    },
  )
  cfg.rewards["motion_ee_pos_wide"] = RewardTermCfg(
    func=mdp.motion_ee_pos_error_exp_masked,
    weight=2.5,
    params={
      "command_name": "climb_motion",
      "std": 0.6,
      "ee_indices": _TRANSITION_HAND_EE_INDICES,
      "engage_gate_floor": 0.25,
      "engage_gate_grace_fraction": 0.15,
    },
  )

  # v37 first-press bonus: the chain's first successful gentle anchor
  # press earns an EVENT (+3, once per episode, gentle band + anchor
  # radius + scheduled window — none farmable) instead of ~0.03 of
  # process wage; pairs with the micro-window births above.
  # v38: the ladder repriced — v37's touch bonus (+3 for a 1.5 N instant)
  # outpaid the weld itself (anchor_landing's 40 dt-scales to ~0.8): 57%
  # of episodes collected the touch, zero converted. Touch now requires a
  # WELD-CAPABLE press (>= 4 N total: the per-sphere gate needs 3 N on one
  # sphere sustained through debounce; a flat 3 N split four ways can
  # never weld) and pays less than the weld event above it.
  cfg.rewards["first_anchor_press"] = RewardTermCfg(
    func=mdp.first_anchor_press_bonus(),
    weight=1.5,
    params={
      "command_name": "climb_motion",
      "min_press_n": 4.0,
      "far_floor": 0.6,
      # (v58 note: this bonus is ALL-EE by construction — its tensors run
      # over the full (E,4); only weld/hold needed the index extension.)
    },
  )
  cfg.rewards["first_weld"] = RewardTermCfg(
    func=mdp.first_weld_bonus(),
    weight=8.0,
    # v49 floor ladder (v48 verdict: the flat 0.25 far-side discount
    # stalled cold-start ignition at first_weld ~0.015 for 250+ iters —
    # hot welds are the only welds a beginner can make, and discounting
    # them 4x starved the value stream). Start forgiving, tighten as the
    # band technique forms; the curricula below own the anneal.
    params={
      "command_name": "climb_motion",
      "far_floor": 0.6,
      "ee_indices": _TRANSITION_ALL_EE_INDICES,  # v58 foot ladder
    },
  )
  # v60 seat-charge rung (08-18, WoCoCo/LadderMan review): the capacity
  # ramp charges only under rake shear >= 10 N. On the wall gravity pays
  # that for free (the cycle's hidden subsidy — every welded limb hangs
  # loaded from birth); the mount's hands weld from a STANDING body, feet
  # carrying the weight, so a fresh weld can idle below the threshold and
  # still sit at 30% capacity when the schedule transfers load — the
  # fullrun's 3-8 step RH rips all die inside the 12.5-step ramp. The
  # ladder stays monotone: weld 8 < charged 9 < held 10, each rung ~0.15 s
  # after the last (charge line 12.5 steps, hold rung 20).
  cfg.rewards["first_charge"] = RewardTermCfg(
    func=mdp.first_charge_bonus(),
    weight=9.0,
    params={
      "command_name": "climb_motion",
      "ee_indices": _TRANSITION_ALL_EE_INDICES,
      "anchor_radius_m": 0.15,
      # Probe verdict (08-18, model_16350, 941 RH self-welds): life med
      # 3 steps (= the first judged step), charge@death 0.16, tension
      # rips 92%, placement 2.1 cm — charging current is abundant
      # (shear-frac 0.67; LH self-welds reach full 129/129), the missing
      # behavior is TENSION management on the newborn weld (press-in).
      # Rule-8: the rung starts at the survive-the-onset bar.
      "full_fraction": 0.4,
    },
  )
  cfg.curriculum["charge_frac_ratchet"] = CurriculumTermCfg(
    func=mdp.reward_param_schedule,
    params={
      "reward_name": "first_charge",
      "param_name": "full_fraction",
      "stages": [
        {"iter": 16350, "value": 0.4},
        {"iter": 17500, "value": 0.7},
        {"iter": 18500, "value": 1.0},
      ],
      "steps_per_iter": 96,
    },
  )
  # v54 precision-desert bridge (probe: welds sit at 7.5 cm median from
  # the anchor — full press wage at 15 cm, landing bonus dead at std 3 cm;
  # nobody paid for the last 7 cm). The wage radius ratchets IN and the
  # landing kernel starts WIDE enough to reach the current distribution,
  # then ratchets back to precision.
  cfg.rewards["motion_anchor_landing"].params["std"] = 0.06
  cfg.curriculum["press_radius_ratchet"] = CurriculumTermCfg(
    func=mdp.reward_param_schedule,
    params={
      "reward_name": "motion_scheduled_press",
      "param_name": "anchor_radius_m",
      "stages": [
        {"iter": 0, "value": 0.15},
        {"iter": 2500, "value": 0.10},
        {"iter": 4500, "value": 0.06},
      ],
      "steps_per_iter": 96,
    },
  )
  cfg.curriculum["landing_std_ratchet"] = CurriculumTermCfg(
    func=mdp.reward_param_schedule,
    params={
      "reward_name": "motion_anchor_landing",
      "param_name": "std",
      "stages": [
        {"iter": 0, "value": 0.06},
        {"iter": 3500, "value": 0.045},
        {"iter": 5000, "value": 0.03},
      ],
      "steps_per_iter": 96,
    },
  )
  for _name, _term in (
    ("weld_floor_anneal", "first_weld"),
    ("press_floor_anneal", "first_anchor_press"),
  ):
    cfg.curriculum[_name] = CurriculumTermCfg(
      func=mdp.reward_param_schedule,
      params={
        "reward_name": _term,
        "param_name": "far_floor",
        "stages": [
          {"iter": 0, "value": 0.6},
          {"iter": 1500, "value": 0.4},
          {"iter": 3000, "value": 0.25},
        ],
        "steps_per_iter": 96,
      },
    )
  # v40 ladder top: holding out-prices welding (0.5 s of continuous
  # engagement), bridging into the engage-gate discount's sustained
  # income — which the same v40 restores to FULL leverage by gating the
  # wide kernels (see their params).
  cfg.rewards["first_hold"] = RewardTermCfg(
    func=mdp.first_hold_bonus(),
    weight=10.0,
    # v43: the bar lowers to a rung the current lifetime distribution can
    # actually reach (welds die in 4-8 steps; a 25-step bonus that never
    # fires teaches nothing — one-shot bonuses only shape what gets
    # SAMPLED). Raise back toward 25 as lifetimes grow.
    # v52 (rule-8 again, one rung lower): 4 ignited hold instantly when
    # lifetimes sat at 3 steps (v43 precedent). v53: lifetimes reached a
    # 62-step mean — the rung climbs to 8 so 0.08 s flash-holds stop
    # collecting; keep stepping toward 25 as the distribution stretches.
    params={
      "ee_indices": _TRANSITION_ALL_EE_INDICES,  # v58 foot ladder
      "command_name": "climb_motion",
      "hold_steps": 20,  # v55 mastery call: lifetimes mean 33 steps
      # v54: only holds AT the scheduled anchor count (position-blind
      # rungs paid 7.5 cm-off holds and taught offset placement).
      "anchor_radius_m": 0.15,
    },
  )

  # v19 rent abolition: the climb battery's support_loss bills a flat
  # 2-anchor shortfall per step — true premise on the wall ("un-latched
  # means falling"), false in the mount's standing prefix (feet on the
  # floor, zero anchors scheduled). The inherited flat bill charged the
  # whole standing stretch -8/s against a ~4.5/s tracking-income ceiling
  # while a termination costs -10 once, so the v15-v18 optimum was dying
  # fast (head-butt / hand-shove = the two quickest exits) and the one
  # net-positive state was a two-pad crutch (rent stopped + pose income).
  # The scheduled variant's requirement follows the clip: zero while it
  # stands, climb parity from any wall-borne stance. Same weight/params.
  support = cfg.rewards["microspine_support_loss"]
  cfg.rewards["microspine_support_loss"] = RewardTermCfg(
    func=mdp.motion_scheduled_support_loss,
    weight=support.weight,
    params={**support.params, "command_name": "climb_motion"},
  )

  if not play:
    if not stand_first:
      cfg.rewards["motion_scheduled_press"].weight = (
        _TRANSITION_PRESS_BOOTSTRAP_STAGES[0]["weight"]
      )
      cfg.curriculum["press_bootstrap_anneal"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
          "reward_name": "motion_scheduled_press",
          "weight_stages": _TRANSITION_PRESS_BOOTSTRAP_STAGES,
          "steps_per_iter": 96,
        },
      )
    # Reverse-curriculum RSI: the biased window opens at the latch stretch
    # and widens backward on the stage ceilings (see the constant block).
    # stand_first pins stage 0 — advancement is a MASTERY call (manual
    # kill/edit/resume), not a clock.
    window_stages = (
      [dict(_TRANSITION_SPAWN_WINDOW_STAGES[0])]
      if stand_first
      else [dict(s) for s in _TRANSITION_SPAWN_WINDOW_STAGES]
    )
    cfg.curriculum["spawn_bias_window"] = CurriculumTermCfg(
      func=mdp.spawn_bias_window,
      params={"stages": window_stages, "steps_per_iter": 96},
    )
    # v34 kernel ratchet: the wide kernels capture from anywhere, but
    # their gentle slope pays half-compliance generously — v33 parked at
    # the half raise (the shape's comfort point). Tighten the stds once
    # the error is down: the halfway wage shrinks, the finish margin
    # grows, and the dead zone never reopens because each stage fires
    # only after the previous one has already pulled the error inside it.
    cfg.curriculum["wide_joint_std_ratchet"] = CurriculumTermCfg(
      func=mdp.reward_param_schedule,
      params={
        "reward_name": "motion_joint_pos_wide",
        "param_name": "std",
        "stages": [
          {"iter": 0, "value": 0.7},
          {"iter": 2200, "value": 0.5},
          {"iter": 3200, "value": 0.35},
        ],
        "steps_per_iter": 96,
      },
    )
    cfg.curriculum["wide_ee_std_ratchet"] = CurriculumTermCfg(
      func=mdp.reward_param_schedule,
      params={
        "reward_name": "motion_ee_pos_wide",
        "param_name": "std",
        "stages": [
          {"iter": 0, "value": 0.6},
          {"iter": 2200, "value": 0.45},
          {"iter": 3200, "value": 0.3},
        ],
        "steps_per_iter": 96,
      },
    )

  # v16 anti-crutch pair (user's video verdict on v15: the stand phase
  # converged to LEANING — body slumped on the wall, pads propped below
  # the slam-tax threshold, balance outsourced to the crutch). Wall
  # contact is a SCHEDULED resource: outside seating windows any pad
  # press bills at the pay's own scale; and the standing prefix tracks
  # root-xy against the clip (the one stretch where global position IS
  # the task — the cycle's position-free doctrine left exactly this
  # coordinate unwatched). Both live in every transition variant.
  cfg.rewards["motion_unscheduled_press"] = RewardTermCfg(
    func=mdp.motion_unscheduled_press_penalty,
    weight=-0.5,
    params={"command_name": "climb_motion"},
  )
  # v54 skating bill (user's video verdict: the stance shuffles and the
  # feet slide — the floor plant checked out, the ledger just never
  # priced a sliding foot). Sole horizontal speed bills while that foot's
  # wall-mount edge is not imminent; the swing itself stays free through
  # the tows' edge-lead gate.
  cfg.rewards["standing_foot_skate"] = RewardTermCfg(
    func=mdp.standing_foot_skate_penalty(),
    weight=-1.0,  # v56: -0.3 realized ~-0.009/ep — invisible; the shuffle stayed
    params={"command_name": "climb_motion"},
  )
  cfg.rewards["standing_root_xy"] = RewardTermCfg(
    func=mdp.standing_root_xy_tracking,
    weight=1.0,
    params={"command_name": "climb_motion"},
  )
  # v17: the wall fence — v16 answered root-xy + the pad tax by bending
  # at the hips and resting its HEAD on the wall (no sensor watched the
  # skull). v21 extends the roster to the whole body minus the parts the
  # choreography lays on the wall (see the roster constants): the v20
  # deterministic policy folded around the two-geom list — chest held at
  # the 10 cm line while the unfenced knee and shoulder sat in the wall
  # plane. Standing prefix only, zero when clear.
  cfg.rewards["standing_wall_fence"] = RewardTermCfg(
    func=mdp.standing_wall_clearance_penalty(),
    weight=-2.0,
    params={
      "command_name": "climb_motion",
      "geom_names": _TRANSITION_FENCE_BODY_GEOMS,
    },
  )
  cfg.rewards["standing_wall_fence_arms"] = RewardTermCfg(
    func=mdp.standing_wall_clearance_penalty(),
    weight=-2.0,
    params={
      "command_name": "climb_motion",
      "geom_names": _TRANSITION_FENCE_ARM_GEOMS,
      "clearance_m": 0.04,
    },
  )
  # v22 hard boundary (user doctrine): only pads and toes may touch the
  # wall — for every other link wall contact IS a fall, terminated like
  # base_height. The fence remains the approach gradient; this is the
  # floor under it. Safe only in the post-v19 economy: while living was
  # net-negative this executioner would have been a free exit door.
  cfg.terminations["wall_contact"] = TerminationTermCfg(
    func=mdp.standing_wall_contact_termination(),
    params={
      "command_name": "climb_motion",
      "geom_names": (
        _TRANSITION_FENCE_BODY_GEOMS + _TRANSITION_FENCE_ARM_GEOMS
      ),
    },
  )
  # v53 contact law (the LadderMan enforcement): a scheduled contact
  # missing for > grace is a FALL, terminated like base_height. The
  # mismatch discount priced desync and lost (v52: LH mastered while RH
  # sat at 0.002 occupancy — a bounded rent never beats a risky door);
  # a termination has no null space. See the term's docstring.
  cfg.terminations["scheduled_contact_loss"] = TerminationTermCfg(
    func=mdp.scheduled_contact_loss_termination(),
    # v54: position-aware (see the term docstring) — only a weld within
    # radius of the EE's own scheduled anchor satisfies the law. Loose at
    # birth (15 cm admits 96% of the v53 distribution: no cliff), annealed
    # to precision by the curriculum below.
    params={
      "command_name": "climb_motion",
      "grace_s": 2.0,
      "anchor_radius_m": 0.15,
    },
  )
  cfg.curriculum["contact_law_radius"] = CurriculumTermCfg(
    func=mdp.termination_param_schedule,
    params={
      "termination_name": "scheduled_contact_loss",
      "param_name": "anchor_radius_m",
      "stages": [
        {"iter": 0, "value": 0.15},
        {"iter": 3000, "value": 0.10},
        {"iter": 5000, "value": 0.08},
      ],
      "steps_per_iter": 96,
    },
  )
  # v61c fresh-era alignment: the 2 s execution grace is v54 machinery
  # priced for a policy that CAN weld — on the fresh run it guillotined
  # 79% of latch-window episodes before any spark existed (iter-1334
  # autopsy). The executioner matures with skill: long runway while the
  # policy learns to weld at all, honest 2 s by the precision era.
  cfg.curriculum["contact_law_grace"] = CurriculumTermCfg(
    func=mdp.termination_param_schedule,
    params={
      "termination_name": "scheduled_contact_loss",
      "param_name": "grace_s",
      "stages": [
        {"iter": 0, "value": 8.0},
        {"iter": 2500, "value": 4.0},
        {"iter": 4800, "value": 2.0},
      ],
      "steps_per_iter": 96,
    },
  )

  # Seed economy (2026-08-15, the latch-window autopsy's verdict): seeded
  # births are the mount's crane classroom — the one place "hold quietly"
  # can be learned — and the ledger let the policy demolish it: rips at
  # holiday price (-0.25), first-weld bonuses collectable by rip-and-redo,
  # and nothing anywhere paying calm. Three entries close it:
  # 1. Demolition bills full price, un-annealed (failure reasons only; the
  #    reverse-shear unhook stays legal — the choreography itself releases
  #    the seeded hands at the walk-up).
  cfg.rewards["seeded_anchor_rip"] = RewardTermCfg(
    func=mdp.seeded_anchor_rip_penalty,
    # v57: -2.5 realized only -0.05/event through dt scaling (the 08-13
    # lesson's arithmetic, rediscovered on the bills side) — demolishing a
    # seed cost a third of one weld bonus. -25 realizes -0.5/event, ~3x
    # the weld bonus: a rip is now a loss, not a rounding error.
    # v61: the fresh era starts at the ladder's first stage and re-anneals
    # to the full v57 price (see _TRANSITION_SEED_RIP_BILL_STAGES).
    weight=_TRANSITION_SEED_RIP_BILL_STAGES[0]["weight"],
    params={},
  )
  cfg.curriculum["seed_rip_bill_anneal"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "seeded_anchor_rip",
      "weight_stages": _TRANSITION_SEED_RIP_BILL_STAGES,
      "steps_per_iter": 96,
    },
  )
  # v57 seed settle guard: the RSI birth transient (~90 N shear + 40 N
  # tension for exactly steps 2-3) instant-ripped ~40-100% of set_hold RH
  # seeds since the classroom opened — the policy has never once seen a
  # surviving RH weld. 6 steps (0.12 s) outwaits the settle; in-play
  # engagements keep the stock 2-step readout guard.
  cfg.commands[
    "microspine_affordance"
  ].microspine_cfg.limit_surface.seed_hold_judge_guard_steps = 2
  # v3 plant (08-18, user spec): the virtual spring field replaces the
  # rigid first touch — a soft 2-5 mm per-unit suspension layer (Wang 2017
  # geometry: 15 deg off-normal, kn-derived stiffness), touch+drag
  # welding (>=1 sphere contact + >=10 N down-shear; the 3 N normal gate
  # and spine-fraction gate retire), remaining-length seating quality x
  # per-unit catch lottery with empty tickets, recruitment ramp RETAINED
  # (Q1=B: first_charge stays live). All parameters are cfg API
  # (MicrospineCfg.spring_* / LimitSurfaceCfg.unit_catch_*) for bench
  # calibration. Flip False to rebuild the legacy plant for old-ckpt
  # replay exams.
  cfg.commands[
    "microspine_affordance"
  ].microspine_cfg.spring_field_enabled = True
  # v3 ignition ladder (probe-driven): 10 N contact shear = genuine weight
  # transfer through the pad (a PD-held arm routes a 30 N pull as only
  # 2-8 N of CONTACT shear) — a cliff for a newborn policy. Easy -> hard,
  # graduation returns to the honest 10 N spec (v11 engage-gate doctrine).
  # Fresh-run clock: iters count from 0; ceilings are mastery calls.
  cfg.curriculum["engage_shear_ladder"] = CurriculumTermCfg(
    func=mdp.microspine_param_schedule,
    params={
      "param_name": "engage_shear_threshold_n",
      "stages": [
        # v63c (08-20 05:3x, last-lever mastery call): the gate must sit
        # INSIDE the reachable tail. After the wage boost AND the dwell
        # rebalance the touching down-shear tail reached max 4.35 N but
        # p99 = 2.47 N — a 4 N gate stays ~p99.99 territory (both v63
        # sweeps flat, zero welds at 2644 vs v62's 2332). 3 N puts sparks
        # at the tail's edge; the ladder still graduates to the honest
        # 10 N with the gravity rungs.
        {"iter": 0, "value": 3.0},
        # v61d restage (night shift): the original 1500/3000 ceilings
        # slipped past with ZERO ignition (a protocol violation — the
        # gate tightened to 7 N while no weld had ever fired). Held at
        # 4 N until sparks exist; advance by mastery call.
        # v62d push (08-19 day shift, mastery call): at 3375 the fresh-weld
        # rate is ~0.005/ep (charge_probe: 0 self-welds in 192x900) — the
        # 7 N step would starve the only weld supply the hold/charge
        # classrooms have. Spacing kept; graduation still lands on 10 N.
        # v62e3 re-key (08-19 evening, mastery call): 7 N landed at 4200
        # UNDER 0.6 g and strangled the pipeline for real (weld 0.021 ->
        # 0.0025 -> 0.0013 over 110 windows, no recovery slope) — lighter
        # gravity donates less free down-shear, so the same gate is a
        # taller hill. The ladder rides the GRAVITY ladder: 4 N through
        # 0.6-0.7 g, 7 N when the 0.8 g rung lands, the honest 10 N spec
        # when full gravity returns. v62e5: shifted in lockstep with the
        # gravity push (engage graduates at 0.6 g first).
        {"iter": 8800, "value": 7.0},
        {"iter": 11400, "value": 10.0},
      ],
    },
  )
  # v62e gravity ladder (user call, 08-19 afternoon): the seeded-birth
  # settle transient scales with gravity — at 1 g the set_hold RH seed died
  # at guard expiry (med = p90 = 6 steps, 227-343 N) with 0% easing over
  # 5 h while LH graduated 3% -> 82%: a gradient desert, not a slow course.
  # 0.6 g cuts the transient under the ultimate trip so the classroom
  # EXISTS; the ramp returns to honest physics (v11 doctrine). The seed
  # guard returns to the climbing-parity 2 steps in the same package.
  # Exam scripts resolve this ladder at the checkpoint's iteration
  # (curriculum-gravity exams, user call 08-19; `gravity=` token pins a
  # scale — the graduation exam pins 1.0).
  cfg.curriculum["gravity_ladder"] = CurriculumTermCfg(
    func=mdp.gravity_magnitude_schedule,
    params={
      # v62e5 push (user call, 08-19 evening: "engage 学会了吗？就要进
      # 0.7 了？"): gravity does NOT rise until engage GRADUATES at
      # 0.6 g — the ramp's job is robustification of a formed skill, not
      # a deadline. At the original 5000 ceiling the skill was 1-in-11
      # episodes, classroom-only, 105 N slam-style, RH zero. Mastery
      # criteria for pulling these ceilings earlier: weld lane >= 0.045
      # (~1/3 episodes) + approach-context welds in fullrun + RH
      # self-weld ignition. Iters below are CEILINGS (v11 doctrine).
      "stages": [
        {"iter": 0, "value": 0.6},
        {"iter": 8000, "value": 0.7},
        {"iter": 8800, "value": 0.8},
        {"iter": 9600, "value": 0.9},
        {"iter": 10400, "value": 1.0},
      ],
      "steps_per_iter": 96,
    },
  )
  # v61d ignition aid (night shift, iter-1467 diagnosis): the drag half of
  # touch+drag welding had no income anywhere (touching down-shear p90 =
  # 0.62 N, max 3.83 N, never one step over the 4 N gate — a structural
  # absence exploration cannot cross). The wage is dense from the first
  # newton and shares the press wage's anti-farm gates; it anneals to
  # ZERO before graduation (ignition scaffolding, v4 press-boot doctrine).
  # v61e boost (night shift, 176-iter verdict): at 0.4 the wage lane read
  # EXACTLY 0.0000 while the same-gated press wage read 0.0001 — gates
  # work, the signal is an order below visibility. The v4 press-bootstrap
  # playbook verbatim: boost the ignition wage, widen its payable window
  # (lead 0.8 -> 2.0 s so the whole at-anchor hover pays), anneal back
  # once the mission completes.
  cfg.rewards["motion_predrag_wage"] = RewardTermCfg(
    func=mdp.motion_predrag_wage,
    weight=2.5,
    params={
      "command_name": "climb_motion",
      "shear_peak_n": 8.0,
      "shear_band_hi_n": 16.0,
      "overdrive_cutoff_n": 30.0,
      "regrip_lead_time_s": 2.0,
      "anchor_radius_m": 0.15,
      "wage_window_s": 2.0,
    },
  )
  # v62e6 A (user call, 08-19 evening): the DWELL wage — the rung UNDER
  # the drag wage. The weld gate needs contact sustained through the
  # debounce; nothing ever paid persistence, so hop-and-slap cost nothing
  # and the policy never learned to stay. Continuity-ramped (full pay at
  # 10 consecutive steps), same gates as the drag wage, anneals on the
  # graduation timeline.
  cfg.rewards["motion_anchor_dwell_wage"] = RewardTermCfg(
    func=mdp.motion_anchor_dwell_wage(),
    weight=0.8,
    params={
      "command_name": "climb_motion",
      "regrip_lead_time_s": 2.0,
      "wage_window_s": 2.0,
    },
  )
  cfg.curriculum["dwell_wage_anneal"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "motion_anchor_dwell_wage",
      "weight_stages": [
        {"iter": 0, "weight": 0.8},
        # v63b rebalance (08-20 04:2x, ignition diagnosis): the dwell
        # course GRADUATED (touch occupancy 0.35) and its full pay was
        # BLOCKING the next rung — the 134k-sample probe showed touching
        # down-shear p50=0.00, max=3.50 N, >=4N never: the policy farms
        # flat dwell income instead of dragging (dwell lane 0.11 = 6x
        # the drag lane). The v37 ordering law: a lower rung must never
        # outpay the rung above it. Dropped to 0.25 so touch income
        # re-points at the drag wage (2.5, untouched).
        {"iter": 2300, "weight": 0.25},
        {"iter": 8000, "weight": 0.1},
        {"iter": 9600, "weight": 0.0},
      ],
      # v62e6b: omitting this once quadrupled the clock and zeroed the
      # wage at birth (the scheduler default was still the early-era 24;
      # it is 96 since 08-19). Kept explicit — the test guard enforces it.
      "steps_per_iter": 96,
    },
  )
  cfg.curriculum["predrag_wage_anneal"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "motion_predrag_wage",
      # v62c (08-19 11:0x): ignition landed at 2332/2592 and is still
      # multiplying (30% of windows at 2732) — the anneal shifts +1500 so
      # the booster outlives its mission (v4 doctrine: the press boost
      # retired only after ignition completed). Ceilings, mastery calls.
      # v62e6: the 6000/7500 tail shifts onto the graduation timeline
      # (engage graduates at 0.6 g before gravity climbs; retiring the
      # drag income mid-concentration-course would starve the course).
      "weight_stages": [
        {"iter": 0, "weight": 2.5},
        {"iter": 4500, "weight": 1.0},
        {"iter": 8000, "weight": 0.4},
        {"iter": 9600, "weight": 0.0},
      ],
      "steps_per_iter": 96,
    },
  )
  # v3 contract extension (user call): per-EE pad->anchor vectors in the
  # base frame — the blind-map fix. Actor AND critic; pre-v3 checkpoint
  # replay deletes this term before env construction.
  for _grp in ("actor", "critic"):
    cfg.observations[_grp].terms["motion_anchor_relative"] = (
      ObservationTermCfg(func=mdp.motion_anchor_relative)
    )
    # v62 contract extension (user call, 08-19 night): absolute stance —
    # robot root pose + wall base, both in the env-local world frame. The
    # blind-map fix's second half: anchors were relative, now the frame
    # itself is observable. Deploy = state estimation + calibrated wall.
    cfg.observations[_grp].terms["root_pose_world"] = ObservationTermCfg(
      func=mdp.root_pose_world
    )
    cfg.observations[_grp].terms["wall_base_position"] = (
      ObservationTermCfg(func=mdp.wall_base_position)
    )
  # 2. Born-welded EEs collect no first-press/first-weld bonus this episode
  #    (wired inside the bonus factories — nothing to configure here).
  # 3. A welded hand's arm is paid to be STILL: |qvel| of the same side's
  #    shoulder/elbow bills while that hand is engaged. At the policy's
  #    flail speeds (~2-3 rad/s) this is ~0.2-0.3/step against the ~2.5+
  #    pose income of actually holding the stance.
  cfg.rewards["engaged_arm_stillness"] = RewardTermCfg(
    func=mdp.engaged_arm_stillness_penalty(),
    weight=-0.3,  # v55 mastery call: churn texture is the complaint
    params={},
  )

  if stand_first:
    # Phase "stand steady FIRST" (user call): the engage economy is fully
    # OFFLINE so standing/raising/holding is the WHOLE objective — no
    # mismatch bills for not latching, no press pay, and the pose income
    # is UNGATED (the schedule-adherence gate otherwise pays a standing
    # robot 25 cents on the dollar). The rip bills, the slam tax and the
    # anti-crutch pair stay: standing means standing FREE. Graduation to
    # the engage phase is a lane-mastery call, then relaunch on the plain
    # -From0 task.
    cfg.rewards["motion_engage_mismatch"].weight = 0.0
    cfg.rewards["motion_scheduled_press"].weight = 0.0
    # v24: no free press window while standing. The clamped clock (below)
    # parks one frame short of the first rising edge, which would hold
    # the regrip-imminent press exemption open FOREVER — a sub-8N hand
    # prop anywhere on the wall would ride it for free (the v15 crutch,
    # relicensed). Press pay is zero here, so the window has no business:
    # any un-latched press bills.
    cfg.rewards["motion_unscheduled_press"].params["bill_all_press"] = True
    # v23: the stand clock holds the last pre-engage frame forever. The
    # marching clock crossed first_engage at 5.6 s and switched OFF every
    # standing-prefix gate (fence, wall_contact executioner, root-xy) for
    # the remaining 24 s of a timeout episode — the v20/v21 motionless
    # wall-lean lived in exactly that unprotected stretch (caught by the
    # user asking "do these bills even apply here?"). Clamped, the whole
    # episode is a standing lesson and the raised place pose is the only
    # full-income stance — the raise cannot be waited out.
    cfg.commands["climb_motion"].clamp_at_first_engage = True
    # The safety rent is engage economy too: past first_engage the marching
    # schedule requires hands on the wall, and a standing robot would bleed
    # -8/s again for the rest of the episode — in a phase where latching
    # pays nothing, that just re-teaches "die at the boundary". The v19
    # probe's residual -0.08 lane was exactly episodes crossing the latch
    # frame; standing the FULL clock out must stay net-positive here.
    cfg.rewards["microspine_support_loss"].weight = 0.0
    for name in ("motion_joint_pos", "motion_ee_pos", "motion_root_ori"):
      cfg.rewards[name].params["engage_gate_floor"] = None

  if engage_gate and not play:
    # Engage-quality gate (see the constant block): loose at birth,
    # tightening with the peel band, stock by graduation. The static
    # write covers the pre-first-reset state; the curriculum owns it
    # afterwards.
    cfg.commands[
      "microspine_affordance"
    ].microspine_cfg.engage_parallel_cos_threshold = math.cos(
      math.radians(_TRANSITION_ENGAGE_GATE_STAGES[0]["max_tilt_deg"])
    )
    cfg.curriculum["engage_gate_anneal"] = CurriculumTermCfg(
      func=mdp.engage_gate_schedule,
      params={
        "stages": [dict(s) for s in _TRANSITION_ENGAGE_GATE_STAGES],
        "steps_per_iter": 96,
      },
    )

  # v14: the hand tow pays PROGRESS, not position (see
  # motion_approach_progress — the rent-seeking endgame of the position
  # tow: slam -> jam -> hover, each patch just moved the tenant). Approach
  # earns once per meter, hovering earns zero, retreat refunds. Key name
  # kept for lane continuity.
  cfg.rewards["motion_swing_approach_far"] = RewardTermCfg(
    func=mdp.motion_approach_progress(),
    weight=_TRANSITION_APPROACH_FAR_WEIGHT,
    params={
      "command_name": "climb_motion",
      "ee_indices": _TRANSITION_ALL_EE_INDICES,  # v58 foot ladder
      "edge_lead_time_s": 2.0,
    },
  )
  # v12 slam tax (probe: median 94 N / p95 379 N vs the 3 N reference —
  # the ram that topples the robot AND ultimate-rips every seeded anchor).
  # Zero through 8 N, then 0.1/N, capped.
  # v56 grind tax: unwelded press that slides along the wall — the
  # visible rubbing, priced directly (slide bill covers engaged pads
  # only; the skate tax covers the floor; this covers the wall).
  cfg.rewards["pad_grind"] = RewardTermCfg(
    func=mdp.pad_grind_penalty,
    weight=-0.3,
    params={"command_name": "climb_motion"},
  )
  cfg.rewards["press_overdrive"] = RewardTermCfg(
    func=mdp.press_overdrive_penalty,
    weight=-0.5,
    # v47 (user catch: the flat-seated WELD needs ~8-14 N total — one
    # sphere at 3 N under a 4-way even split — and the old 8 N tax start
    # criminalized the entire weldable band while the 5 N wage peak paid
    # an un-weldable force): tax starts past the weldable band, just
    # inside the ~17-35 N standing tip-over budget.
    params={"command_name": "climb_motion", "limit_n": 32.0, "cap": 8.0},
  )

  if asym_arm_tracking:
    # Strict legs+waist keep the original weight/std; arms move to a second
    # term at weight/2 and std*2 (gate params carried over verbatim). Same
    # split for root-frame EE tracking: feet strict, hands loose.
    jp = cfg.rewards["motion_joint_pos"]
    cfg.rewards["motion_joint_pos"] = RewardTermCfg(
      func=mdp.motion_joint_pos_error_exp_masked,
      weight=jp.weight,
      params={
        **jp.params,
        "joint_keywords": _TRANSITION_ARM_JOINT_KEYWORDS,
        "exclude_matches": True,
      },
    )
    cfg.rewards["motion_joint_pos_arm"] = RewardTermCfg(
      func=mdp.motion_joint_pos_error_exp_masked,
      weight=jp.weight * 0.5,
      params={
        **jp.params,
        "std": jp.params["std"] * 2.0,
        "joint_keywords": _TRANSITION_ARM_JOINT_KEYWORDS,
        "exclude_matches": False,
      },
    )
    ee = cfg.rewards["motion_ee_pos"]
    cfg.rewards["motion_ee_pos"] = RewardTermCfg(
      func=mdp.motion_ee_pos_error_exp_masked,
      weight=ee.weight,
      params={**ee.params, "ee_indices": _TRANSITION_FOOT_EE_INDICES},
    )
    cfg.rewards["motion_ee_pos_hands"] = RewardTermCfg(
      func=mdp.motion_ee_pos_error_exp_masked,
      weight=ee.weight * 0.5,
      params={
        **ee.params,
        "std": ee.params["std"] * 2.0,
        "ee_indices": _TRANSITION_HAND_EE_INDICES,
      },
    )

  # A mount has no climb rate (the clip's step_height is 0); the command
  # obs lane reads an honest constant zero. Hardcoded rather than derived:
  # clip_average_climb_rate falls back to the CYCLE's nominal rate when the
  # asset is absent at config-build time.
  cfg.commands["climb_velocity"].ranges.lin_vel_z = (0.0, 0.0)

  if wall_band is None:
    # Vertical-wall task: no wall_angle curriculum at all — the mount runs
    # on the identity wall.
    cfg.curriculum.pop("wall_angle", None)
  elif play:
    # Deterministic band eval: hard-pin the band CENTER (matches the
    # single center-native clip installed above).
    cfg.curriculum["wall_angle"].params["angle_stages"] = [
      {"iter": 0, "angle_deg": wall_band}
    ]
  else:
    # Band training: the inherited wall_angle term samples uniform inside
    # the MOUNT-FEASIBLE window per episode and writes the wall pose
    # BEFORE the command's reset — the sampled angle then picks the
    # nearest native clip.
    center, jitter = _BAND_ANGLE_STAGES[wall_band]
    cfg.curriculum["wall_angle"].params["angle_stages"] = [
      {"iter": 0, "angle_deg": center, "jitter_deg": jitter}
    ]
  if not tilt_anneal:
    # Warm-started lineages run the calibrated band from step 0 (the
    # static config carries it); a from-scratch MOUNT run keeps the
    # inherited climb anneal instead — the v8 window probe confirmed the
    # motionref18 cold-start failure in this posture too: at the
    # calibrated peel band every fresh standing-height hand latch rips
    # through the TILT-PEEL channel (7/8 forced releases) before its
    # value is ever experienced. Wide birth band -> calibrated by 3000,
    # ahead of the window schedule's full-chain stage (3800).
    cfg.curriculum.pop("tilt_peel", None)
  elif not play and "tilt_peel" in cfg.curriculum:
    # v30: the peel-band ladder shifts with the engage gate (+2500, see
    # the gate constant) — both presumed ignition by ~1000 and the
    # counter drifted past that through the sigma re-mints.
    tilt_stages = cfg.curriculum["tilt_peel"].params["stages"]
    cfg.curriculum["tilt_peel"].params["stages"] = [
      {**s, "iter": s["iter"] + 2500} if s["iter"] > 0 else dict(s)
      for s in tilt_stages
    ]
  if not play:
    # v30 rip-bill holiday (see the constant block): token bills during
    # ignition, full price restored by curriculum as holds consolidate.
    cfg.rewards["unsafe_microspine_release"].weight = (
      _TRANSITION_RIP_BILL_STAGES[0]["weight"]
    )
    cfg.rewards["engaged_tilt"].weight = (
      _TRANSITION_TILT_BILL_STAGES[0]["weight"]
    )
    cfg.curriculum["rip_bill_anneal"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "unsafe_microspine_release",
        "weight_stages": [dict(s) for s in _TRANSITION_RIP_BILL_STAGES],
        "steps_per_iter": 96,
      },
    )
    cfg.curriculum["tilt_bill_anneal"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "engaged_tilt",
        "weight_stages": [dict(s) for s in _TRANSITION_TILT_BILL_STAGES],
        "steps_per_iter": 96,
      },
    )

  # Cosmetic: the pre-reset init pose is the mount's standing start (RSI
  # overwrites it on every reset). Base x = 1.5 - the center clip's
  # horizontal standoff (see the _BAND_STANDING_X note).
  base_x = 1.27 if wall_band is None else _BAND_STANDING_X[wall_band]
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(base_x, 0.0, 0.726),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos=dict(PHYBOT_C2_STANDING_JOINT_POS),
    joint_vel={".*": 0.0},
  )

  cfg.metrics["mount_handoff_ready"] = MetricsTermCfg(
    func=mdp.mount_handoff_ready,
    params={
      "command_name": "climb_motion",
      "affordance_command_name": "microspine_affordance",
    },
  )

  return cfg


def phybot_c2_climb_unified_env_cfg(
  play: bool = False,
  domain_rand: bool = True,
  student_obs: bool = False,
) -> ManagerBasedRlEnvCfg:
  """L-D unified full-chain env: standing spawn -> mount -> seam -> cycles.

  The KL-distillation student's world (docs/plans/ 2026-08-10 plan §6 and
  the matrix plan). Deltas against the climb cfg it is built from:

  * command = :class:`mdp.UnifiedMountClimbCommandCfg` on the STITCHED
    table (mount prologue rolls into the cycle at the handoff frame; no
    terminal clamp) with the three-way episode-start mix;
  * floor contact enabled (spec_fn wrap, same as the transition task);
  * climb-rate command swapped for the schedule-aware variant: honest zero
    in the mount segment, the sampled cycle rate past the seam;
  * rewards at CLIMB PARITY — no press bootstrap (ignition is the KL
    teachers' job now), plus the wide hand tow for the mount segment;
  * no release drill in v1 (``robustness_release=False``); R-series comes
    back per episode type later;
  * unified v1 runs the vertical wall only (band-unified is a follow-up).
  """
  cfg = phybot_c2_climb_wall_env_cfg(
    contact_model="wang2017",  # preserve the separate legacy spring-field recipe
    play=play,
    domain_rand=domain_rand,
    student_obs=student_obs,
    robustness_release=False,
  )

  base_spec_fn = cfg.scene.spec_fn
  cfg.scene.spec_fn = functools.partial(
    _transition_spec_fn, base_fn=base_spec_fn
  )

  cycle_motion = cfg.commands["climb_motion"]
  cfg.commands["climb_motion"] = mdp.UnifiedMountClimbCommandCfg(
    entity_name=cycle_motion.entity_name,
    motion_file=str(_MOUNT_MOTION_FILE),
    cycle_motion_file=cycle_motion.motion_file,
    affordance_command_name=cycle_motion.affordance_command_name,
    ee_body_names=cycle_motion.ee_body_names,
    sampling_mode="start" if play else "uniform",
    joint_pos_noise=0.0 if play else cycle_motion.joint_pos_noise,
    spawn_xy_noise=(0.0, 0.0) if play else _TRANSITION_SPAWN_XY_NOISE,
    spawn_yaw_noise=0.0 if play else _TRANSITION_SPAWN_YAW_NOISE,
    episode_mix=(1.0, 0.0, 0.0) if play else (0.3, 0.5, 0.2),
    seam_window_s=0.5,
  )

  # Schedule-aware climb rate: carry every sampled field of the cycle's
  # command cfg, only the class (and its segment gate) changes.
  vel = cfg.commands["climb_velocity"]
  vel_fields = {
    f.name: getattr(vel, f.name) for f in dataclasses.fields(vel)
  }
  cfg.commands["climb_velocity"] = mdp.ScheduleAwareClimbVelocityCommandCfg(
    **vel_fields
  )

  # Progress-paid hand tow for the mount segment (same term as the
  # transition task; the cycle segment's swings sit inside the narrow
  # funnel anyway).
  cfg.rewards["motion_swing_approach_far"] = RewardTermCfg(
    func=mdp.motion_approach_progress(),
    weight=_TRANSITION_APPROACH_FAR_WEIGHT,
    params={
      "command_name": "climb_motion",
      "ee_indices": _TRANSITION_ALL_EE_INDICES,  # v58 foot ladder
      "edge_lead_time_s": 2.0,
    },
  )
  cfg.rewards["press_overdrive"] = RewardTermCfg(
    func=mdp.press_overdrive_penalty,
    weight=-0.5,
    # v47 (user catch: the flat-seated WELD needs ~8-14 N total — one
    # sphere at 3 N under a 4-way even split — and the old 8 N tax start
    # criminalized the entire weldable band while the 5 N wage peak paid
    # an un-weldable force): tax starts past the weldable band, just
    # inside the ~17-35 N standing tip-over budget.
    params={"command_name": "climb_motion", "limit_n": 20.0, "cap": 8.0},
  )
  # Gentle-ONLY press pay + honest tows + toe eviction for the mount
  # segment (see the transition builder's v28 notes); the cycle teacher's
  # ~3 N presses and sub-2s swing windows sit inside all three gates, so
  # cycle-region behavior is unchanged and the KL anneal stays drift-safe.
  press_cfg = cfg.rewards["motion_scheduled_press"]
  cfg.rewards["motion_scheduled_press"] = RewardTermCfg(
    func=mdp.motion_scheduled_press_gentle,
    weight=press_cfg.weight,
    params={
    "command_name": "climb_motion",
    "press_peak_n": 8.0,
    "anchor_radius_m": 0.15,
  },
  )
  narrow = cfg.rewards["motion_swing_approach"]
  cfg.rewards["motion_swing_approach"] = RewardTermCfg(
    func=mdp.motion_swing_approach_masked,
    weight=narrow.weight,
    params={
      "command_name": "climb_motion",
      "std": narrow.params["std"],
      "ee_indices": (0, 1, 2, 3),
      "press_cutoff_n": 6.0,
      "edge_lead_time_s": 2.0,
    },
  )
  cfg.rewards["motion_toe_brace"] = RewardTermCfg(
    func=mdp.motion_toe_brace_target,
    weight=0.4,
    params={"command_name": "climb_motion", "off_schedule_cost": 2.5},
  )
  # v32 two-scale tracking kernels for the mount segment (see the
  # transition builder's note); the cycle segment lives inside the narrow
  # kernels anyway, so these add pull only where the narrow ones are dead.
  cfg.rewards["motion_joint_pos_wide"] = RewardTermCfg(
    func=mdp.motion_joint_pos_error_exp_masked,
    weight=2.5,
    params={
      "command_name": "climb_motion",
      "std": 0.7,
      "joint_keywords": _TRANSITION_ARM_JOINT_KEYWORDS,
      # v40: the ignition teachers retire INTO the gate (user catch: the
      # ungated wide kernels diluted the engage-mismatch discount to half
      # the pose income, halving the hold-vs-rip margin the gate was
      # designed to create). Pre-engage the gate never bites: the
      # schedule is empty, so an un-engaged robot has no mismatch (and an
      # EARLY engager is correctly discounted); it starts working exactly
      # at the engage phase.
      "engage_gate_floor": 0.25,
      "engage_gate_grace_fraction": 0.3,
    },
  )
  cfg.rewards["motion_ee_pos_wide"] = RewardTermCfg(
    func=mdp.motion_ee_pos_error_exp_masked,
    weight=2.5,
    params={
      "command_name": "climb_motion",
      "std": 0.6,
      "ee_indices": _TRANSITION_HAND_EE_INDICES,
      "engage_gate_floor": 0.25,
      "engage_gate_grace_fraction": 0.3,
    },
  )
  # Schedule-aware support shortfall (see the transition builder's v19
  # note): the unified episode's standing prefix must not pay the wall's
  # rent either; cycle-region behavior is climb parity bit-for-bit.
  support = cfg.rewards["microspine_support_loss"]
  cfg.rewards["microspine_support_loss"] = RewardTermCfg(
    func=mdp.motion_scheduled_support_loss,
    weight=support.weight,
    params={**support.params, "command_name": "climb_motion"},
  )
  # Illegal-wall-contact executioner for the unified standing prefix (see
  # the transition builder's v22 note); inert past first engage.
  cfg.terminations["wall_contact"] = TerminationTermCfg(
    func=mdp.standing_wall_contact_termination(),
    params={
      "command_name": "climb_motion",
      "geom_names": (
        _TRANSITION_FENCE_BODY_GEOMS + _TRANSITION_FENCE_ARM_GEOMS
      ),
    },
  )

  # Unified v1 = vertical wall; same rationale as the transition task.
  cfg.curriculum.pop("wall_angle", None)
  cfg.curriculum.pop("tilt_peel", None)

  # Full chains need the runway: mount (~14.4 s) + a few cycles.
  cfg.episode_length_s = max(float(cfg.episode_length_s), 30.0)

  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(1.27, 0.0, 0.726),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos=dict(PHYBOT_C2_STANDING_JOINT_POS),
    joint_vel={".*": 0.0},
  )

  cfg.metrics["mount_handoff_ready"] = MetricsTermCfg(
    func=mdp.mount_handoff_ready,
    params={
      "command_name": "climb_motion",
      "affordance_command_name": "microspine_affordance",
    },
  )

  return cfg
