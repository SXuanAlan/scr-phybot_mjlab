"""Config contract of the walk->climb transition (mount) task.

The transition lives in its own PARALLEL package
(``tasks/climb_transition``) and builds its cfg by mutating the climb
builder's output. Its load-bearing promises, asserted here: CONTRACT PARITY
with climb (identical observation/action terms, so climb checkpoints
warm-start strict), one-shot playback via the MountMotionCommand subclass,
floor enabled through the spec_fn wrap, blind-delivery scatter on standing
starts only — and ZERO edits to the climb_wall package itself.
"""

import mujoco
import pytest

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_PAD_GIMBAL,
  configure_phybot_c2_floor_geom,
)
from scr_phybot_mjab.tasks.climb_transition.config.phybot_c2.env_cfgs import (
  _TRANSITION_SPAWN_XY_NOISE,
  _TRANSITION_SPAWN_YAW_NOISE,
  _transition_spec_fn,
  phybot_c2_climb_transition_env_cfg,
)
from scr_phybot_mjab.tasks.climb_transition.mdp.motion import (
  MountMotionCommandCfg,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)
from scr_phybot_mjab.tasks.climb_wall.mdp.motion import ClimbMotionCommandCfg


def test_transition_keeps_the_climb_contract() -> None:
  """Climb obs terms in the same order PLUS the v3 anchor lane at the end,
  same commands, same action. v3 (08-18, user call) deliberately broke
  strict contract parity: motion_anchor_relative joins actor+critic (the
  blind-map fix), so the mount trains FRESH — warm-starting from climb
  checkpoints retired with the old contract."""
  climb = phybot_c2_climb_wall_env_cfg()
  trans = phybot_c2_climb_transition_env_cfg()

  _v62_tail = (
    "motion_anchor_relative",
    "root_pose_world",
    "wall_base_position",
  )
  assert tuple(trans.observations["actor"].terms) == tuple(
    climb.observations["actor"].terms
  ) + _v62_tail
  assert tuple(trans.observations["critic"].terms) == tuple(
    climb.observations["critic"].terms
  ) + _v62_tail
  # v3 plant: spring field ON for the mount (legacy default stays OFF so
  # the cycle plant is byte-identical).
  assert trans.commands[
    "microspine_affordance"
  ].microspine_cfg.spring_field_enabled
  assert not climb.commands[
    "microspine_affordance"
  ].microspine_cfg.spring_field_enabled
  # v62e4: lightly damped field (Wang publishes no damping; user call =
  # small) + release-side retraction rides the re-engage cooldown.
  assert trans.commands[
    "microspine_affordance"
  ].microspine_cfg.spring_damping_range == (0.2, 0.8)
  assert tuple(trans.commands) == (
    "climb_velocity",
    "microspine_affordance",
    "climb_motion",
  )
  assert tuple(trans.actions) == tuple(climb.actions)
  assert (
    trans.actions["joint_pos"].actuator_names
    == climb.actions["joint_pos"].actuator_names
  )


def test_transition_motion_command_is_the_mount_subclass() -> None:
  cfg = phybot_c2_climb_transition_env_cfg()
  motion = cfg.commands["climb_motion"]
  assert isinstance(motion, MountMotionCommandCfg)
  assert motion.motion_file.endswith("climb_mount_prologue.npy")
  assert motion.sampling_mode == "uniform"
  assert motion.joint_pos_noise == pytest.approx(0.02)
  assert motion.spawn_xy_noise == _TRANSITION_SPAWN_XY_NOISE
  assert motion.spawn_yaw_noise == pytest.approx(_TRANSITION_SPAWN_YAW_NOISE)
  # v8: the window schedule leads with the SURVIVAL stage (v2's window —
  # stand and reach without toppling), then retraces the validated ladder:
  # latch, bridge the raise, full chain. Static tuple = stage 0.
  assert motion.spawn_bias_segments == (
    "ground_stand",
    "hand_raise",
    "hand_place",
  )
  window_stages = cfg.curriculum["spawn_bias_window"].params["stages"]
  assert window_stages[0]["segments"] == motion.spawn_bias_segments
  # v61c fresh era: stage 1 is the SPARK NURSERY (unseeded near-wall
  # births only); the full staggered-latch window returns at 2200.
  assert window_stages[1]["segments"] == ("hand_place",)
  assert "hand_set_left" in window_stages[2]["segments"]
  # v61c: the executioner matures on a grace ladder (8 -> 4 -> 2 s).
  grace_stages = cfg.curriculum["contact_law_grace"].params["stages"]
  assert grace_stages[-1]["value"] == pytest.approx(2.0)
  # v61d: the pre-weld drag wage (ignition scaffolding) exists, pays the
  # touch+drag rule's missing half, and anneals to ZERO before graduation.
  drag = cfg.rewards["motion_predrag_wage"]
  assert drag.weight == pytest.approx(2.5)  # v61e ignition boost
  assert drag.params["shear_peak_n"] == pytest.approx(8.0)
  assert drag.params["regrip_lead_time_s"] == pytest.approx(2.0)
  drag_stages = cfg.curriculum["predrag_wage_anneal"].params["weight_stages"]
  assert drag_stages[-1]["weight"] == pytest.approx(0.0)
  # v61d: the shear ignition ladder holds 4 N until sparks (restaged).
  # v62d: pushed 3500/5000 -> 4200/5600 (weld supply 0.005/ep at 4 N).
  # v62e3: re-keyed to ride the gravity ladder (7 N at 0.6 g strangled).
  # v62e5: gravity holds 0.6 g until engage graduates; shear in lockstep.
  shear_stages = cfg.curriculum["engage_shear_ladder"].params["stages"]
  assert [s["iter"] for s in shear_stages] == [0, 8800, 11400]
  grav_iters = [
    s["iter"] for s in cfg.curriculum["gravity_ladder"].params["stages"]
  ]
  assert grav_iters == [0, 8000, 8800, 9600, 10400]
  # v62e gravity ladder (user call): the seeded-birth settle transient
  # scales with gravity; graduation returns to honest physics.
  grav_stages = cfg.curriculum["gravity_ladder"].params["stages"]
  assert [s["value"] for s in grav_stages] == [0.6, 0.7, 0.8, 0.9, 1.0]
  assert grav_stages[-1]["value"] == pytest.approx(1.0)
  assert window_stages[-1]["segments"][0] == "ground_stand"
  # v54: the foot classrooms join at 3400 (seeded two-hand births inside
  # the mounts — the crane condition, now for the legs) and stay in the
  # full-chain stage.
  # v59: foot-classroom concentration at 10500 (brace_hold = the feet's
  # seeded-hold crane classroom), full chain returns at 12600.
  # v60 (08-18): RH-specialty window at 16350 (cumulative-iter clock — the
  # resume path restores common_step_counter), full-chain ceiling 19500.
  # v61 fresh era: the latch stage waits for sparks (550 -> 1200);
  # v62b: the seeded-latch return too (2200 -> 3000).
  # v62e6: approach+RH concentration (5200) + full-chain return (6700).
  # v63c: seed/foot/full-chain trio +600 (window follows the sparks).
  assert [s["iter"] for s in window_stages] == [
    0, 1200, 3600, 4000, 4400, 5200, 6700, 10500, 12600, 16350, 19500,
  ]
  assert "left_foot_mount" in window_stages[3]["segments"]
  assert window_stages[5]["segments"] == (
    "hand_place", "hand_set_left", "hand_place_right",
  )
  assert window_stages[7]["segments"] == (
    "hand_set_hold", "left_foot_mount", "foot_switch_hold",
    "right_foot_mount", "brace_hold",
  )
  assert window_stages[9]["segments"] == (
    "hand_set_left", "hand_place_right", "hand_set_hold", "brace_hold",
  )
  # v62e6 A: the dwell wage (continuity-ramped touch persistence).
  assert cfg.rewards["motion_anchor_dwell_wage"].weight == pytest.approx(
    0.8
  )
  # v62e6b: every transition-side reward_weight schedule MUST pin the
  # 96-step iteration clock explicitly (the dwell wage was born zeroed
  # when this was omitted under the old 24 default; the default is 96
  # since 08-19, but explicit-always keeps the clock audit-proof).
  for _name, _term in cfg.curriculum.items():
    if _term.params and "weight_stages" in _term.params:
      assert _term.params.get("steps_per_iter") == 96, _name
  assert "right_foot_mount" in window_stages[-1]["segments"]
  # v59b: the unmastered brace class stays through full chain.
  assert "brace_hold" in window_stages[-1]["segments"]
  # v54 precision machinery: position-aware contact law + desert bridges.
  assert cfg.terminations["scheduled_contact_loss"].params[
    "anchor_radius_m"
  ] == pytest.approx(0.15)
  assert "contact_law_radius" in cfg.curriculum
  assert "press_radius_ratchet" in cfg.curriculum
  assert "landing_std_ratchet" in cfg.curriculum
  assert cfg.rewards["motion_anchor_landing"].params["std"] == pytest.approx(
    0.06
  )
  assert cfg.rewards["first_hold"].params["anchor_radius_m"] == pytest.approx(
    0.15
  )
  assert cfg.rewards["standing_foot_skate"].weight == pytest.approx(-1.0)  # v56
  # v57 seed settle guard (RSI birth transient was instant-ripping the RH
  # seed classroom); v62e returns it to the climbing-parity 2 steps — the
  # gravity ladder now shrinks the transient at the source.
  ls = cfg.commands["microspine_affordance"].microspine_cfg.limit_surface
  assert ls.seed_hold_judge_guard_steps == 2
  # v61 fresh era: the seed-rip bill re-anneals with skill (-2.5 -> -25).
  assert cfg.rewards["seeded_anchor_rip"].weight == pytest.approx(-2.5)
  rip_stages = cfg.curriculum["seed_rip_bill_anneal"].params["weight_stages"]
  assert rip_stages[-1]["weight"] == pytest.approx(-25.0)
  # v28: the bootstrap returns at a moderate price on the GENTLE-ONLY wage
  # shape (slams unpaid + taxed make the raise safe; v10's complaint was
  # the saturating shape paying slams the full wage). Anneals to parity.
  from scr_phybot_mjab.tasks.climb_transition.mdp.rewards import (
    motion_scheduled_press_gentle,
  )

  press = cfg.rewards["motion_scheduled_press"]
  assert press.func is motion_scheduled_press_gentle
  assert press.weight == pytest.approx(0.6)  # v56
  # v34 triangle wage; v38 moves the peak to the WELD-capable 5 N (a flat
  # 3 N split over four spheres can never cross the 3 N per-sphere gate).
  assert press.params["press_peak_n"] == pytest.approx(20.0)  # v55 firm grip
  # v35 spatial anchor: the wage pays only presses AT the EE's own anchor
  # (the hip-height wall-slap farm is out of the window).
  assert press.params["anchor_radius_m"] == pytest.approx(0.15)
  assert press.params["wage_window_s"] == pytest.approx(2.0)  # v56 time-box
  assert cfg.rewards["pad_grind"].weight == pytest.approx(-0.3)  # v56
  # v37 ignition pair + v38 ladder repricing: touch requires weld-capable
  # force and pays LESS than the weld event above it.
  bonus = cfg.rewards["first_anchor_press"]
  assert bonus.weight == pytest.approx(1.5)
  assert bonus.params["min_press_n"] == pytest.approx(4.0)
  assert bonus.func.__qualname__.startswith("first_anchor_press_bonus")
  weld = cfg.rewards["first_weld"]
  assert weld.weight == pytest.approx(8.0)
  assert weld.func.__qualname__.startswith("first_weld_bonus")
  assert weld.weight > bonus.weight
  # v60 seat-charge rung: weld < charged < held stays monotone; the bar
  # starts at the survive-the-onset fraction (rule-8, 08-18 probe: RH
  # self-welds die at 3 steps / 0.16 charge) and ratchets to 1.0.
  charge = cfg.rewards["first_charge"]
  assert charge.func.__qualname__.startswith("first_charge_bonus")
  assert weld.weight < charge.weight < cfg.rewards["first_hold"].weight
  assert charge.params["ee_indices"] == (0, 1, 2, 3)
  assert charge.params["anchor_radius_m"] == pytest.approx(0.15)
  assert charge.params["full_fraction"] == pytest.approx(0.4)
  charge_stages = cfg.curriculum["charge_frac_ratchet"].params["stages"]
  assert charge_stages[-1]["value"] == pytest.approx(1.0)
  # v60: the v37 LH-edge micro-window quota is RETIRED (it pinned 25% of
  # births to the LH pre-weld frames 495-510, outside the specialty
  # window, for a lesson LH graduated long ago).
  assert cfg.commands["climb_motion"].spawn_edge_fraction == pytest.approx(
    0.0
  )
  # v34 kernel ratchet: wide stds tighten by stages, never reopening the
  # dead zone (each stage fires after the error is already inside it).
  for name, final in (
    ("wide_joint_std_ratchet", 0.35),
    ("wide_ee_std_ratchet", 0.3),
  ):
    stages = cfg.curriculum[name].params["stages"]
    assert stages[-1]["value"] == pytest.approx(final)
    assert [s["iter"] for s in stages] == sorted(s["iter"] for s in stages)
  anneal = cfg.curriculum["press_bootstrap_anneal"]
  assert [s["weight"] for s in anneal.params["weight_stages"]] == [
    0.6,  # v56: ignition scaffolding mostly retired
    0.4,
  ]
  assert anneal.params["steps_per_iter"] == 96
  # A mount has no climb rate; the command obs lane reads constant zero.
  assert cfg.commands["climb_velocity"].ranges.lin_vel_z == (0.0, 0.0)
  # Floor comes in through the spec_fn wrap (introspectable partial).
  assert cfg.scene.spec_fn.func is _transition_spec_fn


def test_transition_wide_hand_tow_over_narrow_funnel() -> None:
  """v14 progress tow on the hands only, stacked over the cycle's
  untouched narrow funnel."""
  for play in (False, True):  # reward keys stay train/play identical
    cfg = phybot_c2_climb_transition_env_cfg(play=play)
    far = cfg.rewards["motion_swing_approach_far"]
    # v14: PROGRESS-paid tow (position pay was a rent: slam -> jam ->
    # hover; the derivative pays approach once and hovering zero).
    assert far.func.__qualname__.startswith("motion_approach_progress")
    assert far.weight == pytest.approx(0.5)
    # v58 foot ladder: the edge-gated progress tow covers all four limbs
    # (stand-still income impossible under the 2 s edge gate + delta pay).
    assert far.params["ee_indices"] == (0, 1, 2, 3)
    # v28 honest tows: both gated to imminent scheduled edges; the narrow
    # funnel keeps its precision std but gains the 6 N free-space cutoff
    # (no more jam rent) and covers all four EEs like the term it replaces.
    assert far.params["edge_lead_time_s"] == pytest.approx(2.0)
    narrow = cfg.rewards["motion_swing_approach"]
    assert narrow.func.__name__ == "motion_swing_approach_masked"
    assert narrow.weight == pytest.approx(0.5)
    assert narrow.params["std"] == pytest.approx(0.05)
    assert narrow.params["ee_indices"] == (0, 1, 2, 3)
    assert narrow.params["press_cutoff_n"] == pytest.approx(6.0)
    assert narrow.params["edge_lead_time_s"] == pytest.approx(2.0)
    # v28 toe-prop eviction: off-schedule toe press bills at a real rate.
    # v45: the wage side swaps to the target-tracking variant (peaks at
    # the audit's ~16 N brace — force rewards pay by closeness to target).
    toe = cfg.rewards["motion_toe_brace"]
    assert toe.func.__name__ == "motion_toe_brace_target"
    assert toe.params["off_schedule_cost"] == pytest.approx(2.5)
    # v29 wide basin: position-paid hand tow returns UNDER the v28 gates
    # (lead + press cutoff bound the rent the v14 retirement feared).
    wide = cfg.rewards["motion_swing_approach_wide"]
    assert wide.params["std"] == pytest.approx(0.3)
    assert wide.params["ee_indices"] == (0, 1, 2, 3)  # v58 foot ladder
    assert wide.params["press_cutoff_n"] == pytest.approx(6.0)
    assert wide.params["edge_lead_time_s"] == pytest.approx(2.0)
    # v32 two-scale tracking: wide arm-joint and hand-EE kernels restore
    # the reference's pull from arms-down (narrow kernels are numerically
    # dead 1.2 rad out); ungated ignition teachers, narrow keeps precision.
    jw = cfg.rewards["motion_joint_pos_wide"]
    assert jw.weight == pytest.approx(2.5)  # v33: pull > fall-risk premium
    assert jw.params["std"] == pytest.approx(0.7)
    assert jw.params["joint_keywords"] == ("shoulder", "elbow")
    # v40: the ignition teachers retire INTO the engage gate — ungated
    # they diluted the mismatch discount to half the pose income.
    assert jw.params["engage_gate_floor"] == pytest.approx(0.25)
    ew = cfg.rewards["motion_ee_pos_wide"]
    assert ew.weight == pytest.approx(2.5)
    assert ew.params["std"] == pytest.approx(0.6)
    assert ew.params["ee_indices"] == (0, 1)
    assert ew.params["engage_gate_floor"] == pytest.approx(0.25)
    # v40 ladder top: hold > weld > touch, strictly monotone.
    hold = cfg.rewards["first_hold"]
    assert hold.weight == pytest.approx(10.0)
    assert hold.weight > cfg.rewards["first_weld"].weight
    assert cfg.rewards["motion_scheduled_press"].params[
      "regrip_lead_time_s"
    ] == pytest.approx(0.8)
    over = cfg.rewards["press_overdrive"]
    assert over.weight == pytest.approx(-0.5)
    # v47 rule, v55 numbers: the tax starts past the paid band ([20, 28] N
    # since the firm-grip retarget), never inside it.
    assert over.params["limit_n"] == pytest.approx(32.0)


def test_asym_arm_tracking_flag() -> None:
  """Backup knob: OFF keeps climb-parity tracking terms; ON splits into
  strict legs+waist / loose arms (weight/2, std*2, gate params verbatim)."""
  from scr_phybot_mjab.tasks.climb_transition.mdp.rewards import (
    motion_ee_pos_error_exp_masked,
    motion_joint_pos_error_exp_masked,
  )
  from scr_phybot_mjab.tasks.climb_wall.mdp.rewards import (
    motion_ee_pos_error_exp,
    motion_joint_pos_error_exp,
  )

  off = phybot_c2_climb_transition_env_cfg()
  assert off.rewards["motion_joint_pos"].func is motion_joint_pos_error_exp
  assert off.rewards["motion_ee_pos"].func is motion_ee_pos_error_exp
  assert "motion_joint_pos_arm" not in off.rewards
  assert "motion_ee_pos_hands" not in off.rewards

  on = phybot_c2_climb_transition_env_cfg(asym_arm_tracking=True)
  legs = on.rewards["motion_joint_pos"]
  arms = on.rewards["motion_joint_pos_arm"]
  assert legs.func is motion_joint_pos_error_exp_masked
  assert legs.params["exclude_matches"] is True
  assert legs.weight == pytest.approx(2.0)
  assert legs.params["std"] == pytest.approx(0.2)
  assert arms.weight == pytest.approx(1.0)
  assert arms.params["std"] == pytest.approx(0.4)
  assert arms.params["joint_keywords"] == ("shoulder", "elbow")
  # Gate economics carried over verbatim on every split term.
  for term in (legs, arms):
    assert term.params["engage_gate_floor"] == pytest.approx(0.25)
    assert term.params["engage_gate_grace_fraction"] == pytest.approx(0.3)
  feet = on.rewards["motion_ee_pos"]
  hands = on.rewards["motion_ee_pos_hands"]
  assert feet.func is motion_ee_pos_error_exp_masked
  assert feet.params["ee_indices"] == (2, 3)
  assert feet.weight == pytest.approx(2.5)
  assert hands.params["ee_indices"] == (0, 1)
  assert hands.weight == pytest.approx(1.25)
  assert hands.params["std"] == pytest.approx(0.2)


def test_transition_strips_cycle_training_aids() -> None:
  cfg = phybot_c2_climb_transition_env_cfg()
  # Default gravity IS the vertical-wall operating point; the tilt-peel
  # band is statically configured at the calibrated values.
  assert "wall_angle" not in cfg.curriculum
  assert "tilt_peel" not in cfg.curriculum
  # The From0 variant keeps the climb anneal (cold-start latches rip
  # through the tilt-peel channel at the calibrated band — v8 probe).
  from0 = phybot_c2_climb_transition_env_cfg(
    tilt_anneal=True, engage_gate=True
  )
  if PHYBOT_C2_PAD_GIMBAL:
    # Angle release retired with the pad gimbal — no tilt ladder to anneal.
    assert "tilt_peel" not in from0.curriculum
  else:
    tilt_stages = from0.curriculum["tilt_peel"].params["stages"]
    assert tilt_stages[0]["deg_range"] == (12.0, 15.0)
    assert tilt_stages[-1]["deg_range"] == (3.5, 4.5)
  # Engage-quality gate: strict cos(4 deg) at birth, annealed to the
  # STOCK threshold (None stage) before graduation; the plain task and
  # climb keep the stock 0.25 untouched.
  import math as _math

  gate_cos = from0.commands[
    "microspine_affordance"
  ].microspine_cfg.engage_parallel_cos_threshold
  assert gate_cos == pytest.approx(_math.cos(_math.radians(10.0)))
  gate_stages = from0.curriculum["engage_gate_anneal"].params["stages"]
  # v11 easy -> hard: loose birth gate tracking just inside the tilt-peel
  # band, tightening with it, stock (None) at graduation. v30 shifts the
  # ladder +2500 (the counter drifted past the old thresholds through
  # sigma re-mints before ignition).
  assert [s["max_tilt_deg"] for s in gate_stages] == [
    10.0,
    7.0,
    4.5,
    3.5,
    None,
  ]
  assert [s["iter"] for s in gate_stages] == [0, 3700, 4900, 5500, 7000]
  if not PHYBOT_C2_PAD_GIMBAL:
    tilt_iters = [
      s["iter"] for s in from0.curriculum["tilt_peel"].params["stages"]
    ]
    assert tilt_iters[0] == 0 and all(
      i >= 2500 for i in tilt_iters[1:]
    ), tilt_iters
  # v30 rip-bill holiday: token bills during ignition, annealed back.
  assert from0.rewards["unsafe_microspine_release"].weight == pytest.approx(
    -1.0  # v55: ignition holiday over
  )
  assert from0.rewards["engaged_tilt"].weight == pytest.approx(-0.25)
  rip_stages = from0.curriculum["rip_bill_anneal"].params["weight_stages"]
  assert rip_stages[-1]["weight"] == pytest.approx(-2.5)
  # Seed economy (08-15): demolition of a gifted anchor bills full price
  # from iter 0 — deliberately OUTSIDE the rip-bill holiday anneal — and a
  # welded hand's arm is paid to hold still.
  from scr_phybot_mjab.tasks.climb_transition.mdp.rewards import (
    seeded_anchor_rip_penalty,
  )

  assert from0.rewards["seeded_anchor_rip"].func is seeded_anchor_rip_penalty
  # v61: starts at the fresh-era ladder's first rung, anneals to the v57
  # un-diluted -25 (see seed_rip_bill_anneal).
  assert from0.rewards["seeded_anchor_rip"].weight == pytest.approx(-2.5)
  assert all(
    s["reward_name"] != "seeded_anchor_rip"
    for s in [from0.curriculum["rip_bill_anneal"].params]
  )
  assert from0.rewards["engaged_arm_stillness"].weight == pytest.approx(-0.3)  # v55
  tiltbill = from0.curriculum["tilt_bill_anneal"].params["weight_stages"]
  assert tiltbill[-1]["weight"] == pytest.approx(-1.0)
  plain = phybot_c2_climb_transition_env_cfg()
  assert plain.commands[
    "microspine_affordance"
  ].microspine_cfg.engage_parallel_cos_threshold == pytest.approx(0.25)
  assert "engage_gate_anneal" not in plain.curriculum

  # Stand-first phase: engage economy offline, ungated pose income, the
  # window pinned to the survival stage; gentleness bills stay.
  stand = phybot_c2_climb_transition_env_cfg(
    tilt_anneal=True, engage_gate=True, stand_first=True
  )
  assert stand.rewards["motion_engage_mismatch"].weight == 0.0
  assert stand.rewards["motion_scheduled_press"].weight == 0.0
  assert "press_bootstrap_anneal" not in stand.curriculum
  for name in ("motion_joint_pos", "motion_ee_pos", "motion_root_ori"):
    assert stand.rewards[name].params["engage_gate_floor"] is None
  stand_stages = stand.curriculum["spawn_bias_window"].params["stages"]
  assert len(stand_stages) == 1
  assert stand_stages[0]["segments"] == (
    "ground_stand",
    "hand_raise",
    "hand_place",
  )
  assert stand.rewards["press_overdrive"].weight == pytest.approx(-0.5)
  # v16 anti-crutch pair (the leaning verdict): unscheduled pad press
  # bills, and the standing prefix tracks root-xy against the clip.
  assert stand.rewards["motion_unscheduled_press"].weight == pytest.approx(
    -0.5
  )
  assert stand.rewards["standing_root_xy"].weight == pytest.approx(1.0)
  # v17 wall fence, v21 full-body roster: the mole family morphed through
  # unlisted geoms (head -> shoulder -> knee), so the fence names every
  # link the choreography never lays on the wall; the upper arm gets its
  # own 4 cm ring (place legally brings it to +8.4 cm, minus scatter).
  fence = stand.rewards["standing_wall_fence"]
  assert fence.weight == pytest.approx(-2.0)
  for name in (
    "left_knee_collision",
    "right_shoulder_roll_collision",
    "base_link_collision",
    "neck_pitch_collision",
  ):
    assert name in fence.params["geom_names"]
  for name in fence.params["geom_names"]:  # place-legal parts stay out
    assert "elbow" not in name and "hand" not in name and "ankle" not in name
  arms = stand.rewards["standing_wall_fence_arms"]
  assert arms.weight == pytest.approx(-2.0)
  assert arms.params["clearance_m"] == pytest.approx(0.04)
  assert set(arms.params["geom_names"]) == {
    "left_shoulder_yaw_collision",
    "right_shoulder_yaw_collision",
  }
  # v22 hard boundary: illegal wall contact terminates like a fall — the
  # fence prices the approach, the executioner owns the plane itself.
  for variant in (cfg, from0, stand):
    wall_term = variant.terminations["wall_contact"]
    assert wall_term.func.__qualname__.startswith(
      "standing_wall_contact_termination"
    )
    roster = wall_term.params["geom_names"]
    assert "left_knee_collision" in roster
    assert "right_shoulder_yaw_collision" in roster
    for name in roster:  # pads/toes/elbows/hands never in the roster
      assert "spine" not in name and "toe" not in name
      assert "elbow" not in name and "ankle" not in name
  # v19 rent abolition: the support-shortfall requirement follows the
  # schedule (zero while the clip stands) in EVERY transition variant —
  # the inherited flat-2 bill charged the standing prefix -8/s of
  # unavoidable rent and made dying the optimum. Weight/params carry over.
  from scr_phybot_mjab.tasks.climb_transition.mdp.rewards import (
    motion_scheduled_support_loss,
  )

  for variant in (cfg, from0, stand):
    support = variant.rewards["microspine_support_loss"]
    assert support.func is motion_scheduled_support_loss
    assert support.params["command_name"] == "climb_motion"
    assert support.params["minimum_supports"] == 2
  assert cfg.rewards["microspine_support_loss"].weight == pytest.approx(-4.0)
  assert from0.rewards["microspine_support_loss"].weight == pytest.approx(-4.0)
  # Stand-first zeroes even the scheduled rent: past first_engage the
  # marching schedule requires hands on and would re-open the -8/s bill
  # in a phase where latching pays nothing (v19 probe residual).
  assert stand.rewards["microspine_support_loss"].weight == 0.0
  # Stand-first also restricts the uniform RSI half to standing frames
  # (a wall-hanging spawn with the engage economy offline can only rip
  # and die); the plain task keeps full-clip uniform for hold experience.
  assert stand.commands["climb_motion"].spawn_disengaged_only is True
  assert cfg.commands["climb_motion"].spawn_disengaged_only is False
  # v23: the stand clock never leaves the standing prefix — the marching
  # clock had been switching off every prefix-gated term at 5.6 s.
  assert stand.commands["climb_motion"].clamp_at_first_engage is True
  assert cfg.commands["climb_motion"].clamp_at_first_engage is False
  # v24: the clamped clock would hold the regrip-imminent press exemption
  # open forever; stand bills every un-latched press instead.
  assert stand.rewards["motion_unscheduled_press"].params[
    "bill_all_press"
  ] is True
  assert not cfg.rewards["motion_unscheduled_press"].params.get(
    "bill_all_press", False
  )
  # v26 ground-spawn From0: full economy + marching clock, only the RSI
  # birthplace is pinned to disengaged frames; none of the stand-first
  # surgeries ride along.
  ground = phybot_c2_climb_transition_env_cfg(
    tilt_anneal=True, engage_gate=True, ground_spawns=True
  )
  gm = ground.commands["climb_motion"]
  assert gm.spawn_disengaged_only is True
  assert gm.clamp_at_first_engage is False
  assert ground.rewards["motion_engage_mismatch"].weight != 0.0
  assert ground.rewards["motion_scheduled_press"].weight != 0.0
  assert ground.rewards["microspine_support_loss"].weight != 0.0
  assert not ground.rewards["motion_unscheduled_press"].params.get(
    "bill_all_press", False
  )
  # v1 trains the nominal mount; the release drill is a later R-series
  # rung. This also removes the only reseat_hold producer, so the terminal
  # clock cannot be frozen short of the handoff frame.
  assert "random_release" not in cfg.events
  assert "mount_handoff_ready" in cfg.metrics
  # The base DR package stays (contract parity includes s2r hardening).
  for dr_event in ("encoder_bias", "body_friction", "base_mass"):
    assert dr_event in cfg.events


def test_band_variants_pin_the_band_and_carry_the_clip_grid() -> None:
  """Band-teacher tasks: native-angle clip grid + band-pinned wall_angle."""
  b80 = phybot_c2_climb_transition_env_cfg(wall_band=80.0)
  motion = b80.commands["climb_motion"]
  assert motion.motion_file.endswith("climb_mount_prologue_a80.npy")
  # a76 is physically infeasible (standing-mount balance boundary ~80 deg):
  # the grid is {80, 84} and the mount training window clamps to [78, 85].
  assert tuple(a for a, _ in motion.angle_clip_files) == (80.0, 84.0)
  for angle, path in motion.angle_clip_files:
    assert path.endswith(f"climb_mount_prologue_a{int(angle)}.npy")
  # Native clips carry true world geometry — the cycle's vertical-height
  # compensation must stay off.
  assert motion.spawn_match_vertical_height is False
  assert b80.curriculum["wall_angle"].params["angle_stages"] == [
    {"iter": 0, "angle_deg": 81.5, "jitter_deg": 3.5}
  ]
  assert "tilt_peel" not in b80.curriculum
  # Blind-delivery scatter still trains (band training keeps the v4 knobs).
  assert motion.spawn_xy_noise == _TRANSITION_SPAWN_XY_NOISE
  # Standing init pose: 1.5 minus the 80-band clips' horizontal standoff.
  base_x = b80.scene.entities["robot"].init_state.pos[0]
  assert base_x == pytest.approx(1.31)

  b88 = phybot_c2_climb_transition_env_cfg(wall_band=88.0)
  assert tuple(a for a, _ in b88.commands["climb_motion"].angle_clip_files) == (
    88.0,
  )
  assert b88.curriculum["wall_angle"].params["angle_stages"] == [
    {"iter": 0, "angle_deg": 88.0, "jitter_deg": 2.0}
  ]

  # Band play: hard-pinned center, single center-native clip, start clock.
  b80_play = phybot_c2_climb_transition_env_cfg(play=True, wall_band=80.0)
  play_motion = b80_play.commands["climb_motion"]
  assert play_motion.angle_clip_files == ()
  assert play_motion.motion_file.endswith("climb_mount_prologue_a80.npy")
  assert play_motion.sampling_mode == "start"
  assert b80_play.curriculum["wall_angle"].params["angle_stages"] == [
    {"iter": 0, "angle_deg": 80.0}
  ]

  # The vertical-wall task stays single-clip with no wall_angle curriculum.
  base = phybot_c2_climb_transition_env_cfg()
  assert base.commands["climb_motion"].angle_clip_files == ()
  assert base.commands["climb_motion"].spawn_match_vertical_height is True
  assert base.scene.entities["robot"].init_state.pos[0] == pytest.approx(1.27)


def test_transition_play_is_deterministic_and_noise_free() -> None:
  cfg = phybot_c2_climb_transition_env_cfg(play=True)
  motion = cfg.commands["climb_motion"]
  # Evaluation delivers the canonical frame-0 stance: no spawn scatter, no
  # joint noise, clock from the standing start.
  assert motion.sampling_mode == "start"
  assert motion.joint_pos_noise == 0.0
  assert motion.spawn_xy_noise == (0.0, 0.0)
  assert motion.spawn_yaw_noise == 0.0
  # Play keeps climb-parity press weight and no bootstrap curriculum.
  assert cfg.rewards["motion_scheduled_press"].weight == pytest.approx(0.4)
  assert "press_bootstrap_anneal" not in cfg.curriculum
  assert "spawn_bias_window" not in cfg.curriculum
  for dr_event in ("encoder_bias", "body_friction", "base_mass"):
    assert dr_event not in cfg.events


def test_terminal_worlds_align_for_the_handoff(monkeypatch) -> None:
  """Climb's 90° world == transition's world: same wall, same live floor.

  The point of the wall-angle representation: the climb ladder's terminal
  rung is physically the transition task's environment, so a mount policy
  hands off to a climb policy in one consistent world.
  """
  climb = phybot_c2_climb_wall_env_cfg(play=True)
  trans = phybot_c2_climb_transition_env_cfg(play=True)

  # Identical wall entity binding (same factory, identity compiled pose).
  from scr_phybot_mjab.assets.wall import WALL_OFFSET

  for cfg in (climb, trans):
    wall = cfg.scene.entities["wall"]
    assert tuple(wall.init_state.pos) == tuple(WALL_OFFSET)
    assert tuple(wall.init_state.rot) == (1.0, 0.0, 0.0, 0.0)
  # Climb's play curriculum pins exactly 90 — the identity wall pose.
  stages = climb.curriculum["wall_angle"].params["angle_stages"]
  assert stages == [{"iter": 0, "angle_deg": 90.0}]

  # Both scene spec_fns leave the terrain floor CONTACT-ENABLED (climb's
  # own enable + transition's re-enable are idempotent, not a tug-of-war).
  # The microspine assembly helpers need the composed robot spec, so they
  # are no-op'ed here; the floor call runs for real on a bare terrain spec.
  import scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs as climb_mod

  for helper in (
    "add_contact_pairs",
    "add_pad_wall_contact_excludes",
    "add_toe_contact_pairs",
    "add_attachment_equalities",
  ):
    monkeypatch.setattr(climb_mod, helper, lambda *a, **k: None)

  def floor_contype(scene_spec_fn) -> int:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="terrain")
    body.add_geom(
      name="terrain",
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=[10.0, 10.0, 0.1],
    )
    scene_spec_fn(spec)
    return int(spec.geom("terrain").contype)

  assert floor_contype(climb.scene.spec_fn) == 1
  assert floor_contype(trans.scene.spec_fn) == 1


def test_unified_env_cfg_contract() -> None:
  """L-D unified full-chain env: stitched command, schedule-aware rate,
  climb-parity rewards (KL owns ignition — no press bootstrap)."""
  from scr_phybot_mjab.tasks.climb_transition.config.phybot_c2.env_cfgs import (
    phybot_c2_climb_unified_env_cfg,
  )
  from scr_phybot_mjab.tasks.climb_transition.mdp.commands import (
    ScheduleAwareClimbVelocityCommandCfg,
  )
  from scr_phybot_mjab.tasks.climb_transition.mdp.motion import (
    UnifiedMountClimbCommandCfg,
  )

  cfg = phybot_c2_climb_unified_env_cfg()
  motion = cfg.commands["climb_motion"]
  assert isinstance(motion, UnifiedMountClimbCommandCfg)
  assert motion.motion_file.endswith("climb_mount_prologue.npy")
  assert motion.cycle_motion_file.endswith("climb_cycle_moment_safe_paced.npy")
  assert motion.episode_mix == (0.3, 0.5, 0.2)
  # Schedule-aware climb rate keeps the cycle's sampled ranges.
  vel = cfg.commands["climb_velocity"]
  assert isinstance(vel, ScheduleAwareClimbVelocityCommandCfg)
  assert vel.ranges.lin_vel_z != (0.0, 0.0)
  # Climb parity: press at 0.4, no bootstrap curriculum; wide tow present.
  assert cfg.rewards["motion_scheduled_press"].weight == pytest.approx(0.4)
  assert "press_bootstrap_anneal" not in cfg.curriculum
  assert cfg.rewards["motion_swing_approach_far"].params["ee_indices"] == (
    0, 1, 2, 3,
  )  # v58 foot ladder (unified mirrors transition)
  # The unified standing prefix pays no wall rent either (v19).
  from scr_phybot_mjab.tasks.climb_transition.mdp.rewards import (
    motion_scheduled_support_loss,
  )

  assert (
    cfg.rewards["microspine_support_loss"].func is motion_scheduled_support_loss
  )
  assert "wall_contact" in cfg.terminations
  # Unified v1 = vertical wall; full chains get the runway.
  assert "wall_angle" not in cfg.curriculum
  assert float(cfg.episode_length_s) >= 30.0
  # No release drill in v1.
  assert "random_release" not in cfg.events

  play = phybot_c2_climb_unified_env_cfg(play=True)
  assert play.commands["climb_motion"].episode_mix == (1.0, 0.0, 0.0)
  assert play.commands["climb_motion"].sampling_mode == "start"


def test_climb_wall_package_is_untouched() -> None:
  """The parallel-architecture promise: climb's cfg carries NONE of the
  transition machinery."""
  cfg = phybot_c2_climb_wall_env_cfg()
  motion = cfg.commands["climb_motion"]
  assert type(motion) is ClimbMotionCommandCfg  # not the mount subclass
  assert not hasattr(motion, "spawn_xy_noise")
  assert motion.motion_file.endswith("climb_cad_empirical.npy")
  assert "wall_angle" in cfg.curriculum
  if PHYBOT_C2_PAD_GIMBAL:
    assert "tilt_peel" not in cfg.curriculum  # angle release retired
  else:
    assert "tilt_peel" in cfg.curriculum
  assert "mount_handoff_ready" not in cfg.metrics
  # The transition-only reward variants never leak into the climb battery.
  assert "motion_swing_approach_far" not in cfg.rewards
  assert "motion_joint_pos_arm" not in cfg.rewards
  assert "seeded_anchor_rip" not in cfg.rewards
  assert "engaged_arm_stillness" not in cfg.rewards
  # The cycle keeps its flat support bill (on the wall, un-latched really
  # does mean falling); only the transition swaps the scheduled variant.
  from scr_phybot_mjab.tasks.climb_wall.mdp.rewards import (
    microspine_support_loss,
  )

  assert cfg.rewards["microspine_support_loss"].func is microspine_support_loss
  assert "wall_contact" not in cfg.terminations
  # The mount cfg's own noise knobs default OFF at the class level.
  fields = MountMotionCommandCfg.__dataclass_fields__
  assert fields["spawn_yaw_noise"].default == 0.0


def test_floor_geom_toggle_controls_contact_bits() -> None:
  """The spec_fn wrap's ground truth, on a minimal synthetic spec."""

  def build(enable: bool) -> mujoco.MjsGeom:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="terrain")
    geom = body.add_geom(
      name="terrain",
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=[1.0, 1.0, 0.1],
    )
    configure_phybot_c2_floor_geom(spec, enable_contact=enable)
    return geom

  enabled = build(True)
  assert enabled.contype == 1 and enabled.conaffinity == 1
  disabled = build(False)
  assert disabled.contype == 0 and disabled.conaffinity == 0


def test_scheduled_contact_loss_termination_contract():
  """v53 contact law: config wiring + the counter semantics on stubs."""
  import torch
  from types import SimpleNamespace

  from scr_phybot_mjab.tasks.climb_transition.mdp.terminations import (
    scheduled_contact_loss_termination,
  )

  cfg = phybot_c2_climb_transition_env_cfg(
    tilt_anneal=True, engage_gate=True, ground_spawns=True
  )
  term_cfg = cfg.terminations["scheduled_contact_loss"]
  assert term_cfg.params["grace_s"] == pytest.approx(2.0)

  term = scheduled_contact_loss_termination()
  motion = SimpleNamespace(ref_engaged=torch.zeros(2, 4, dtype=torch.bool))
  aff = SimpleNamespace(
    engaged=torch.zeros(2, 4, dtype=torch.bool),
    dr_release_pending=torch.zeros(2, 4, dtype=torch.bool),
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(
      get_term=lambda n: {
        "climb_motion": motion,
        "microspine_affordance": aff,
      }[n]
    ),
    episode_length_buf=torch.full((2,), 10, dtype=torch.long),
    step_dt=0.02,
  )
  # No scheduled contact -> never fires.
  for _ in range(20):
    assert not term(env, grace_s=0.1).any()
  # Schedule demands LH on env0; grace 0.1 s = 5 steps -> fires on step 6.
  motion.ref_engaged[0, 0] = True
  fired = [bool(term(env, grace_s=0.1)[0]) for _ in range(6)]
  assert fired == [False] * 5 + [True]
  assert not fired[-1] or not term(env, grace_s=0.1)[1]  # env1 untouched
  # Welding clears the count; a later rip restarts it from zero.
  aff.engaged[0, 0] = True
  assert not term(env, grace_s=0.1).any()
  aff.engaged[0, 0] = False
  fired = [bool(term(env, grace_s=0.1)[0]) for _ in range(6)]
  assert fired == [False] * 5 + [True]
  # R4 reseat holds are exempt while pending.
  aff.dr_release_pending[0, 0] = True
  for _ in range(20):
    assert not term(env, grace_s=0.1).any()


def test_from0_ground_runs_asym_arm_tracking():
  """v53: the training task splits tracking strict-legs / loose-arms."""
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Mjlab-Climb-Transition-Phybot-C2-From0-Ground")
  assert "motion_joint_pos_arm" in cfg.rewards
  assert "motion_ee_pos_hands" in cfg.rewards
  assert cfg.rewards["motion_joint_pos_arm"].weight == pytest.approx(1.0)
  assert cfg.rewards["motion_ee_pos_hands"].weight == pytest.approx(1.25)
  assert cfg.rewards["first_hold"].params["hold_steps"] == 20
  assert "scheduled_contact_loss" in cfg.terminations
