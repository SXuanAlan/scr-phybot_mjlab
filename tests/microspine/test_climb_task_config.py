import mujoco
import pytest
import mjlab  # noqa: F401
import scr_phybot_mjab.tasks  # noqa: F401  # Initialize the registry before task modules.

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_PAD_GIMBAL,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  _CLIMB_MOTION_FILE,
  phybot_c2_climb_wall_corners60_env_cfg,
  phybot_c2_climb_wall_env_cfg,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.rl_cfg import (
  phybot_c2_ppo_runner_cfg,
)
from scr_phybot_mjab.tasks.climb_wall.mdp.motion import clip_average_climb_rate
from scr_phybot_mjab.tasks.walk.config.phybot_c2.env_cfgs import (
  PHYBOT_C2_WALK_COLLISION,
)


def test_climb_microspine_contract_and_command_ordering():
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  microspine = cfg.commands["microspine_affordance"].microspine_cfg

  assert microspine.resolve_initial_engaged_mask() == (
    True,
    True,
    True,
    True,
  )
  assert microspine.contact_friction[:2] == (0.2, 0.2)
  assert microspine.reengage_cooldown_steps == 10
  assert microspine.engage_consecutive_steps == 3
  assert microspine.engage_normal_force_threshold == 3.0
  # Proxy count/force settings remain available for legacy recipes; empirical
  # latching uses estimated preload on every physical unit instead.
  assert microspine.engage_min_spheres == 1
  assert microspine.virtual_spine_grid == (5, 5)
  assert microspine.virtual_spine_travel == 0.005  # paper le_max
  assert microspine.engage_min_spine_fraction == 1.0
  assert microspine.engage_min_total_normal_force == 0.0
  assert microspine.engage_unit_preload_n == 10.0
  from scr_phybot_mjab.microspine.contact_sensor import WrenchContactSensorCfg
  sensor = next(s for s in cfg.scene.sensors if isinstance(s, WrenchContactSensorCfg))
  assert {"force", "torque", "pos"}.issubset(sensor.fields)
  assert sensor.history_length == cfg.decimation
  assert len(sensor.reference_sites) == 16
  # Each box can create four contacts even before floor/toe support is counted.
  assert cfg.sim.nconmax > 4 * len(sensor.reference_sites)
  # The empirical distribution is conditioned on a seated unit. Require
  # full geometric seating and do not multiply it by a second capacity DR.
  assert microspine.seat_quality_scales_capacity is False
  # 1 (was 2): latched-contact retraction killed the one-step equality
  # transients the debounce guarded against; the first genuine violation
  # now releases, closing the flicker-through loophole for good.
  assert microspine.limit_surface.violation_consecutive_steps == 1
  # Latched pads hand their load path to the anchor alone: sphere contacts
  # switch off while latched (kills the anchor-vs-contact oscillation that
  # let motionref6's 90-deg feet-hang ride through the release debounce).
  assert microspine.disable_contacts_while_latched is True
  # "down" (was free-follow): during the grace window the anchor follows
  # only the pad's down-rake seating motion, 1 mm/step, orientation frozen
  # at engage — hooked spines seat by dragging down (Wang 2017), they never
  # relocate outward (free-follow let the press->hang flip drag hand
  # anchors 12-44 mm off the wall). Bench 9-grid (07-27): survival 8/8 in
  # every window x mode combo (retraction killed the old contact standoff
  # that free-follow was for), outward drift 0.0 mm in "down" vs 5 mm
  # free, down-drag follow 3.7 mm <= cap under 60 N rake shear.
  assert microspine.anchor_reseat_mode == "down"
  assert microspine.anchor_reseat_step_cap == 0.001
  assert microspine.reanchor_after_grace is True
  # False (motionref14): hooked spines hold the pad where it gripped —
  # projecting the anchor onto the wall plane while the pad sat compressed
  # into it caused the 4-11 mm post-engage pop once contacts retracted.
  assert microspine.anchor_project_to_wall is False
  # The unjudged grace window is gone (judge guard = readout settling only);
  # young anchors are WEAK instead of immune: capacity ramps 0.3 -> 1 over
  # 0.25 s of rake-shear-loaded time (Wang loading-phase recruitment), and
  # every latch redraws F0 (per-grip asperity lottery, paper Fig 6).
  assert microspine.limit_surface.hold_judge_guard_steps == 2
  # v57: the seed settle guard is a TRANSITION fix; climb keeps stock
  # judging (None = no seeded special case) — its RSI births never showed
  # the settle rip and its exams are green.
  assert microspine.limit_surface.seed_hold_judge_guard_steps is None
  assert microspine.limit_surface.capacity_ramp_floor == 1.0
  assert microspine.limit_surface.capacity_ramp_time_s == 0.0
  assert microspine.limit_surface.capacity_ramp_shear_n == 10.0
  assert microspine.limit_surface.redraw_f_t_max_per_latch is False
  assert microspine.limit_surface.model == "empirical_unit"
  assert "random_release" not in cfg.events
  assert cfg.commands["climb_motion"].joint_pos_noise == 0
  assert microspine.limit_surface.hold_grace_steps == 14  # re-seat window only
  assert microspine.limit_surface.moment_capacity_enable is True
  # The main task uses the complete 3x4 CAD array for preload and holding.
  # torquescale = reference-hold-calibrated compliance.
  assert abs(microspine.limit_surface.pad_span_rake_m - 0.063) < 1.0e-6
  assert abs(microspine.limit_surface.pad_span_lateral_m - 0.084) < 1.0e-6
  assert microspine.limit_surface.moment_billing_mode == "per_unit"
  assert microspine.limit_surface.unit_layout == "all"
  assert microspine.limit_surface.unit_rows_rake == 3
  assert microspine.limit_surface.unit_cols_lateral == 4
  assert microspine.limit_surface.f_t_max_range == (360.,360.)
  if PHYBOT_C2_PAD_GIMBAL:
    # Gimbal-era weld: rotationally RIGID — the passive wrist gimbal owns
    # compliance and angle-based release is retired (moment billing only).
    assert microspine.anchor_torquescale == 1.0
  else:
    assert microspine.anchor_torquescale == 0.01
  assert tuple(cfg.actions) == ("joint_pos",)
  # The motion command must follow the affordance: its reset performs RSI and
  # seeds the per-env latch state AFTER the affordance's own reset ran.
  assert tuple(cfg.commands) == (
    "climb_velocity",
    "microspine_affordance",
    "climb_motion",
  )
  assert cfg.metrics["microspine_engage_event_rate_hz"].params == {
    "command_name": "microspine_affordance"
  }


@pytest.mark.parametrize("play", [False, True])
def test_main_climbing_task_uses_twelve_units_and_the_legacy_90_degree_ladder(play):
  from mjlab.tasks.registry import load_env_cfg
  from scr_phybot_mjab.tasks.climb_wall.mdp.curriculums import WALL_ANGLE_STAGES

  cfg = load_env_cfg("Mjlab-Climb-Wall-Phybot-C2", play=play)
  microspine = cfg.commands["microspine_affordance"].microspine_cfg
  ls = microspine.limit_surface
  assert (ls.unit_layout, ls.unit_rows_rake, ls.unit_cols_lateral) == ("all", 3, 4)
  assert microspine.engage_unit_preload_n == 10.0
  event_microspine = cfg.events["microspine_grip"].params["microspine_cfg"]
  assert event_microspine.limit_surface.unit_layout == "all"
  assert event_microspine.engage_unit_preload_n == 10.0
  wall = cfg.curriculum["wall_angle"].params
  if play:
    assert wall["angle_stages"] == [{"iter": 0, "angle_deg": 90.0}]
    assert wall["angle_jitter_deg"] == 0.0
  else:
    assert wall["angle_stages"] == list(WALL_ANGLE_STAGES)


@pytest.mark.parametrize("play", [False, True])
def test_corner_climbing_60_task_has_full_training_and_play_configs(play):
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from scr_phybot_mjab.assets.motions import CAD_CLIMB_CORNERS60_MOTION

  cfg = load_env_cfg("Mjlab-Climb-Wall-Phybot-C2-Corners-A60", play=play)
  main = load_env_cfg("Mjlab-Climb-Wall-Phybot-C2", play=play)
  microspine = cfg.commands["microspine_affordance"].microspine_cfg
  ls = microspine.limit_surface
  assert (ls.unit_layout, ls.unit_rows_rake, ls.unit_cols_lateral) == ("corners", 2, 2)
  assert microspine.engage_unit_preload_n == 10.0
  assert cfg.events["microspine_grip"].params["microspine_cfg"].limit_surface.unit_layout == "corners"
  motion = cfg.commands["climb_motion"]
  assert motion.motion_file == str(CAD_CLIMB_CORNERS60_MOTION)
  assert cfg.commands["climb_velocity"].ranges.lin_vel_z == pytest.approx(
    (clip_average_climb_rate(CAD_CLIMB_CORNERS60_MOTION),) * 2)
  assert cfg.scene.num_envs == main.scene.num_envs
  assert cfg.episode_length_s == main.episode_length_s
  assert cfg.actions == main.actions
  assert tuple(cfg.commands) == tuple(main.commands)
  assert tuple(cfg.observations["actor"].terms) == tuple(main.observations["actor"].terms)
  assert tuple(cfg.observations["critic"].terms) == tuple(main.observations["critic"].terms)
  assert {key: term.weight for key, term in cfg.rewards.items()} == {
    key: term.weight for key, term in main.rewards.items()}
  assert tuple(cfg.terminations) == tuple(main.terminations)
  wall = cfg.curriculum["wall_angle"].params
  if play:
    assert wall["angle_stages"] == [{"iter": 0, "angle_deg": 60.0}]
    assert wall["angle_jitter_deg"] == 0.0
    assert motion.sampling_mode == "start"
    assert motion.height_offset_range == (0.0, 0.0)
    assert not cfg.observations["actor"].enable_corruption
  else:
    assert len(wall["angle_stages"]) > 1
    assert wall["angle_stages"][0]["angle_deg"] == 45.0
    assert wall["angle_stages"][-1]["angle_deg"] == 60.0
    assert motion.sampling_mode == "adaptive"
    assert cfg.observations["actor"].enable_corruption
  agent = load_rl_cfg("Mjlab-Climb-Wall-Phybot-C2-Corners-A60")
  main_agent = load_rl_cfg("Mjlab-Climb-Wall-Phybot-C2")
  assert agent.experiment_name != main_agent.experiment_name
  assert agent.algorithm == main_agent.algorithm
  assert agent.actor == main_agent.actor and agent.critic == main_agent.critic
  assert agent.num_steps_per_env == wall["steps_per_iter"]


def test_corner_climbing_curriculum_cannot_mutate_the_main_task():
  from scr_phybot_mjab.tasks.climb_wall.mdp.curriculums import CORNERS60_WALL_ANGLE_STAGES

  corner = phybot_c2_climb_wall_corners60_env_cfg()
  corner.curriculum["wall_angle"].params["angle_stages"][0]["angle_deg"] = 10.0
  assert CORNERS60_WALL_ANGLE_STAGES[0]["angle_deg"] == 45.0
  assert phybot_c2_climb_wall_corners60_env_cfg().curriculum["wall_angle"].params[
    "angle_stages"][0]["angle_deg"] == 45.0
  assert phybot_c2_climb_wall_env_cfg().curriculum["wall_angle"].params[
    "angle_stages"][-1]["angle_deg"] == 90.0


@pytest.mark.parametrize("play", [False, True])
def test_corner_climbing_80_task_keeps_its_angle_in_training_and_play(play):
  from mjlab.tasks.registry import load_env_cfg
  from scr_phybot_mjab.assets.motions import CAD_CLIMB_CORNERS80_MOTION
  cfg = load_env_cfg("Mjlab-Climb-Wall-Phybot-C2-Corners-A80", play=play)
  microspine = cfg.commands["microspine_affordance"].microspine_cfg
  assert microspine.limit_surface.unit_layout == "corners"
  assert microspine.engage_unit_preload_n == 10.0
  wall = cfg.curriculum["wall_angle"].params
  assert wall["angle_stages"] == [{"iter":0, "angle_deg":80.0}]
  assert wall["angle_jitter_deg"] == 0.0
  assert cfg.events["microspine_grip"].params["microspine_cfg"].limit_surface.unit_layout == "corners"
  assert cfg.commands["climb_motion"].motion_file == str(CAD_CLIMB_CORNERS80_MOTION)


@pytest.mark.parametrize("wall_angle,layout", [(60., "corners"), (80., "corners"), (90., "all")])
def test_empirical_climbing_rewards_unloaded_release_at_each_angle_and_layout(wall_angle, layout):
  cfg = phybot_c2_climb_wall_env_cfg(wall_angle_deg=wall_angle, unit_layout=layout)
  release = cfg.rewards["microspine_release_load"]
  assert callable(release.func) and release.weight == 1.
  assert release.params["shear_scale_n"] == release.params["lateral_scale_n"] == 2.
  assert release.params["shear_cost"] > release.params["lateral_cost"] > 0
  assert release.params["minimum_supports"] == 3
  assert cfg.rewards["unsafe_microspine_release"].weight < 0
  assert cfg.commands["microspine_affordance"].microspine_cfg.engage_unit_preload_n == 10.


def test_release_load_reward_requires_empirical_unit_telemetry():
  cfg = phybot_c2_climb_wall_env_cfg(contact_model="wang2017")
  assert "microspine_release_load" not in cfg.rewards


def test_full_array_comparison_and_legacy_wang_remain_available():
  for kwargs in ({"unit_layout":"all"}, {"contact_model":"wang2017"}):
    cfg = phybot_c2_climb_wall_env_cfg(**kwargs)
    ls = cfg.commands["microspine_affordance"].microspine_cfg.limit_surface
    assert (ls.unit_layout,ls.unit_rows_rake,ls.unit_cols_lateral) == ("all",3,4)


def test_climb_motion_command_is_rsi_over_the_ik_cycle():
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  motion = cfg.commands["climb_motion"]

  assert motion.motion_file == str(_CLIMB_MOTION_FILE)
  assert motion.ee_body_names == (
    "left_hand_spine",
    "right_hand_spine",
    "left_foot_spine",
    "right_foot_spine",
  )
  assert motion.sampling_mode == "adaptive"
  assert motion.affordance_command_name == "microspine_affordance"
  # RSI owns the reset pose: the generic perturbation events must be gone or
  # they would fight (be overwritten by) the reference-state write.
  assert "reset_robot_joints" not in cfg.events
  assert "reset_robot_base" not in cfg.events
  assert "reset_scene_to_default" in cfg.events
  # The commanded climb rate equals the clip's own average cycle rate, so the
  # progress normalization and velocity observation agree with the motion.
  vz = cfg.commands["climb_velocity"].ranges.lin_vel_z
  assert vz[0] == vz[1] == pytest.approx(clip_average_climb_rate(_CLIMB_MOTION_FILE))


def test_climb_play_defaults_to_one_env_without_sync_debug():
  cfg = phybot_c2_climb_wall_env_cfg(play=True)
  command_cfg = cfg.commands["microspine_affordance"].microspine_cfg
  event_cfg = cfg.events["microspine_grip"].params["microspine_cfg"]

  assert cfg.scene.num_envs == 1
  assert not command_cfg.debug
  assert not event_cfg.debug
  # Evaluation shows one continuous climb from the canonical stance.
  assert cfg.commands["climb_motion"].sampling_mode == "start"
  assert cfg.commands["climb_motion"].joint_pos_noise == 0.0


def test_climb_reuses_walk_robot_collision_config():
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  collisions = cfg.scene.entities["robot"].collisions

  assert len(collisions) == 1
  assert collisions[0] is PHYBOT_C2_WALK_COLLISION


def test_climb_wall_keeps_visual_box_but_collides_as_surface_plane():
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  spec = cfg.scene.entities["wall"].spec_fn()
  visual = spec.geom("wall_visual_geom")
  collision = spec.geom("wall_geom")

  assert visual.type == mujoco.mjtGeom.mjGEOM_BOX
  assert visual.contype == 0
  assert visual.conaffinity == 0
  assert collision.type == mujoco.mjtGeom.mjGEOM_PLANE
  assert tuple(collision.pos) == (-0.5, 0.0, 0.0)
  # size[0] is the half-thickness contract (microspine surface projection);
  # size[1] mirrors the real 1.2 m test wall's half-width (visual only).
  assert tuple(collision.size) == (0.5, 0.6, 0.1)


def test_legacy_wang_reward_contract_remains_deepmimic_split():
  cfg = phybot_c2_climb_wall_env_cfg(play=False, contact_model="wang2017")

  # Imitation teaches the motion; every tracking term is position-free
  # (joint space / root frame / orientation) — no global-pose kernel exists.
  assert cfg.rewards["motion_joint_pos"].weight == 2.0
  assert cfg.rewards["motion_joint_vel"].weight == 0.5
  # 2.5: reaching along the reference arc is the first link of the regrip
  # chain and must out-earn holding a parked limb still (motionref5).
  assert cfg.rewards["motion_ee_pos"].weight == 2.5
  assert cfg.rewards["motion_root_ori"].weight == 0.5
  # 0.25 gate floor (motionref10): pose income is scaled by schedule
  # adherence — the engagement-blind pose kernels were the welded statue's
  # last income pillar (ref8/ref9 parked a diagonal at match~0.5 and lived
  # off "looking right"). A quarter stays ungated so birth pose learning
  # never starves; the gate reuses the mismatch bill's edge-grace ramp.
  for gated in ("motion_joint_pos", "motion_ee_pos", "motion_root_ori"):
    assert cfg.rewards[gated].params["engage_gate_floor"] == 0.25
    # Window-proportional grace (motionref17): the gate's edge forgiveness
    # scales with the violated window itself, so clip retiming cannot
    # silently discount welding (the fixed 0.25 s grace had halved the fast
    # clip's foot-window weld bill).
    assert cfg.rewards[gated].params["engage_gate_grace_fraction"] == 0.3
  # Seat-phase step credit: pays pad press only inside the scheduled regrip
  # window (countdown-gated on the observed channel), stops at latch, and
  # its ceiling must stay strictly below the mismatch bill so pressing
  # without latching never profits.
  assert cfg.rewards["motion_scheduled_press"].weight == 0.4
  assert (
    cfg.rewards["motion_scheduled_press"].params["regrip_lead_time_s"] == 0.3
  )
  assert cfg.rewards["motion_scheduled_press"].weight < abs(
    cfg.rewards["motion_engage_mismatch"].weight
  )
  # -1.0: schedule adherence must be priced against the budgeted task income
  # (motionref1 ignored the schedule at -0.25). The bill ramps in over the
  # outer 30% of the violated window around a scheduled flip — the countdown
  # signal the policy sees — because debounce + cooldown make instant
  # compliance impossible; as a window fraction the weld-through price
  # survives clip retiming (motionref17).
  assert cfg.rewards["motion_engage_mismatch"].weight == -1.0
  assert (
    cfg.rewards["motion_engage_mismatch"].params["grace_fraction"] == 0.3
  )
  assert "grace_time_s" not in cfg.rewards["motion_engage_mismatch"].params

  # The climbing drive IS an imitation term (motionref8): income follows the
  # reference's own climb profile, decaying symmetrically when behind or
  # ahead. The free-standing height rewards (and their cap/budget patches)
  # are gone — they funded every reference-ignoring gait.
  assert "climb_height_progress" not in cfg.rewards
  assert "signed_height_progress" not in cfg.rewards
  assert cfg.rewards["motion_ref_height"].weight == 2.5
  # 0.83 step heights of the 30 mm clip — the ratio the term was designed at
  # (0.05 vs the old 60 mm step). See env_cfgs for the iter-1400 basin data.
  assert cfg.rewards["motion_ref_height"].params["std"] == 0.025
  # Landing accuracy on the anchor ladder: the anti-dance term. Ceiling
  # (4 landings / 4.24 s cycle x weight x dt) must stay below the tracking
  # income scale so farmed landings can never dominate real climbing.
  assert cfg.rewards["motion_anchor_landing"].weight == 40.0
  assert cfg.rewards["motion_anchor_landing"].params["std"] == 0.03
  # Dense swing-approach shaping (motionref17): pays progress toward the
  # upcoming landing rung during scheduled swings only, breaking the landing
  # bonus's chicken-and-egg (measured motionref16: landing income stuck at
  # 3-5% of ceiling). Ceiling 0.5/s per swinging EE — strictly below the
  # mismatch bill so hovering released never profits.
  assert cfg.rewards["motion_swing_approach"].weight == 0.5
  assert cfg.rewards["motion_swing_approach"].params["std"] == 0.05
  assert cfg.rewards["motion_swing_approach"].weight < abs(
    cfg.rewards["motion_engage_mismatch"].weight
  )

  # The commanded-gait battery is gone with the state machine.
  for legacy in (
    "commanded_anchor_advance",
    "productive_gait_cycle",
    "confirmed_anchor_advance",
    "climb_gait_support_shortfall",
    "unscheduled_microspine_release",
    "climb_gait_phase_timeout",
    "climb_swing_progress_shaping",
    "climb_seat_press_shaping",
    "commanded_ee_target_error",
    "climb_velocity_z_error",
  ):
    assert legacy not in cfg.rewards

  # Engage/physics contract: support loss outbids the ~4.5/s tracking
  # ceiling so a one-anchor dangle is strictly net-negative.
  assert cfg.rewards["microspine_support_loss"].weight == -4.0
  # -2.5: tension/lateral/ultimate rips must bill visibly (40% of motionref1's
  # 3.5 Hz rip storm was genuine tension pull-off at a negligible price).
  assert cfg.rewards["unsafe_microspine_release"].weight == -2.5
  # Scheduled-window reverse-shear billing (motionref17): the free unhook
  # direction stays free AT scheduled release edges but costs 0.3x while the
  # clip says ENGAGED — motionref16b farmed it as a release valve (43-45
  # latch-jiggle-unhook cycles per 30 s on the abandoned right foot).
  unsafe_params = cfg.rewards["unsafe_microspine_release"].params
  assert unsafe_params["motion_command_name"] == "climb_motion"
  assert unsafe_params["scheduled_reverse_cost"] == 0.3
  assert unsafe_params["scheduled_reverse_grace_s"] == 0.25
  assert cfg.rewards["ee_slide_penalty"].weight == -0.1
  # -0.1 (was -0.01): policy-mean chatter (3-4 Hz, +-4-5 cm hand buzz on
  # motionref8) was economically free and kept blowing the 60 ms engage
  # window; walk prices the same term at -0.25/-0.3.
  assert cfg.rewards["action_rate_l2"].weight == -0.1
  # Engaged-tilt tax ("4-deg" recalibration): capture-relative rotation
  # priced from 3 deg — the gradient companion of the tilt-peel release
  # (DR 3.5-4.5 deg, reason 5). Reference uses <= 1.1 deg.
  assert cfg.rewards["engaged_tilt"].weight == -1.0
  assert cfg.rewards["engaged_tilt"].params["threshold_deg"] == 3.0
  assert cfg.rewards["engaged_tilt"].params["scale_deg"] == 5.0
  assert unsafe_params["tilt_release_cost"] == 0.5
  # Spike tax above 70% of the deploy torque caps (motionref17b): the
  # unpriced near-saturation unhook yank (0.45 -> 1.00 of the shoulder cap
  # between model_2175 and model_4300) must bill; support duty and the
  # gentle weight-shift unhook stay below the threshold and free.
  assert cfg.rewards["torque_headroom"].weight == -8.0
  assert cfg.rewards["torque_headroom"].params["threshold"] == 0.7
  # Tilt-peel physics ("4-deg" pass): suspension travel le_max 5 mm over
  # the 74.7 mm rake span = ~3.8 deg of chassis rotation about the loaded
  # edge; per-grip lottery like F0.
  ls_cfg = cfg.commands["microspine_affordance"].microspine_cfg.limit_surface
  if PHYBOT_C2_PAD_GIMBAL:
    assert ls_cfg.tilt_peel_max_deg_range == (0.0, 0.0)  # angle release off
  else:
    assert ls_cfg.tilt_peel_max_deg_range == (3.5, 4.5)
  assert cfg.rewards["bounded_dof_acc_l2"].weight == -0.02
  assert cfg.rewards["bounded_dof_acc_l2"].params["max_cost"] == 10.0
  assert cfg.rewards["dof_pos_limits"].weight == -5.0
  assert cfg.rewards["termination_penalty"].weight == -500.0

  # The no_progress executioner is retired (motionref17): under the
  # motionref16 economics it truncated every learning episode at ~8-13 s and
  # priced failed regrip experiments at -10, dismantling the dance ->
  # choreography -> climb ladder motionref15 had climbed. Climb pressure now
  # rides motion_ref_height + motion_anchor_landing + motion_swing_approach.
  assert "no_progress" not in cfg.terminations
  assert cfg.terminations["bad_tracking"].params["threshold"] == 0.7
  assert "gait_phase_timeout" not in cfg.terminations
  assert "gait_curriculum_success" not in cfg.terminations


def test_climb_curriculum_retrains_from_scratch():
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  runner_cfg = phybot_c2_ppo_runner_cfg()

  # From-scratch ladder (2026-08-02 evening, user call): the config must
  # fully retrain from iter 0. The iters transcribe the one validated
  # trajectory — v3 ran 0->3300 on 45/75+-12 (with this exact tilt anneal)
  # and the v4 bridge took the same lineage 3300->5050 through
  # 85+-3/88+-2/exact-90 to the ship gate (90-deg cal exams
  # 0.962/0.961/0.963). 2026-08-10: the 80+-5 rung joins for the
  # band-teacher matrix (deploy bands 80+-5/88+-2/90 each mint a teacher at
  # its rung's end), shifting later rungs +350. Every hop overlaps its
  # neighbor's pool; rung boundaries sit >=300 iters from tilt-anneal
  # steps; drive by mastery gates in practice (the iters are ceilings).
  assert "climb_difficulty" not in cfg.curriculum
  term = cfg.curriculum["wall_angle"]
  stages = term.params["angle_stages"]
  assert [(s["iter"], s["angle_deg"], s.get("jitter_deg")) for s in stages] == [
    (0, 45.0, None),
    (1500, 75.0, 12.0),
    (3300, 80.0, 5.0),
    (3900, 85.0, 3.0),
    (4550, 88.0, 2.0),
    (5150, 90.0, None),
  ]
  assert term.params["steps_per_iter"] == runner_cfg.num_steps_per_env == 96

  if PHYBOT_C2_PAD_GIMBAL:
    # Angle release retired with the pad gimbal: no tilt ladder at all —
    # the schedule would just rewrite live bands onto a dead channel.
    assert "tilt_peel" not in cfg.curriculum
  else:
    # v3's birth-validated tilt anneal, ending on the calibrated band well
    # before the wall gets steep (3000 < 3550).
    tilt_term = cfg.curriculum["tilt_peel"]
    tilt_stages = tilt_term.params["stages"]
    assert [
      (s["iter"], s["deg_range"], s["tax_threshold_deg"], s["tax_scale_deg"])
      for s in tilt_stages
    ] == [
      (0, (12.0, 15.0), 10.0, 8.0),
      (1200, (8.0, 10.0), 6.5, 6.0),
      (2400, (5.0, 7.0), 4.0, 5.0),
      (3000, (3.5, 4.5), 3.0, 5.0),
    ]
    assert tilt_stages[-1]["iter"] < stages[2]["iter"]
    assert tilt_term.params["steps_per_iter"] == 96

  # Play/eval pins to the ladder's terminal values: exams always run the
  # honest physics at the target angle regardless of training progress.
  play_cfg = phybot_c2_climb_wall_env_cfg(play=True)
  play_term = play_cfg.curriculum["wall_angle"]
  assert play_term.params["angle_stages"][-1]["angle_deg"] == 90.0
  if PHYBOT_C2_PAD_GIMBAL:
    assert "tilt_peel" not in play_cfg.curriculum
  else:
    play_tilt = play_cfg.curriculum["tilt_peel"].params["stages"]
    assert play_tilt[-1]["deg_range"] == (3.5, 4.5)


def test_climb_action_contract_is_bounded_in_target_space_only():
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  runner_cfg = phybot_c2_ppo_runner_cfg()
  action_cfg = cfg.actions["joint_pos"]

  # IsaacLab-style contract: the Gaussian policy is unclipped in action space;
  # safety bounds live in target space (SoftJointPositionAction + soft limits).
  assert runner_cfg.clip_actions is None
  assert runner_cfg.num_steps_per_env == 96
  # init_std 0.6 is load-bearing for microspine health: 1.0-wide birth
  # exploration overloads engaged anchors and collapses engagement.
  assert runner_cfg.actor.distribution_cfg == {
    "class_name": "GaussianDistribution",
    "init_std": 0.6,
    "std_type": "scalar",
  }
  assert action_cfg.scale == 0.25
  assert action_cfg.__class__.__name__ == "SoftJointPositionActionCfg"
  # 0.98 (was 0.9): the reference works close to some stops (shoulder_roll
  # peaks 0.06 rad from its hard limit at state 4 — kinematically required);
  # tracking must never fight the target clamp. The clip keeps its own
  # clearance via the gen script's limit-avoidance springs.
  assert cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor == 0.98
