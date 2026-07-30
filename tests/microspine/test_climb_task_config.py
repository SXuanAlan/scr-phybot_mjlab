import mujoco
import pytest

from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  _CLIMB_MOTION_FILE,
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
  # 1 (was 2): a rigid pad generically touches a flat wall with ONE sphere —
  # the 2-sphere gate demanded sub-degree parallelism and never passed on
  # motionref8 (78 N pressed through a single sphere, engage counter never
  # started), making every release an irreversible loss and welding limbs.
  # Seating is judged by the virtual spine field instead (Wang 2017: spine
  # normal suspension is force-free over le_max = 5 mm, so seating is
  # kinematic reach, not multi-sphere force).
  assert microspine.engage_min_spheres == 1
  assert microspine.virtual_spine_grid == (5, 5)
  assert microspine.virtual_spine_travel == 0.005  # paper le_max
  assert microspine.engage_min_spine_fraction == 0.5
  # A half-seated latch holds half the load (paper Sec 2.4: capacity is the
  # sum of engaged spines) — seating quality gets a continuous physical
  # payoff, and a corner-poke latch cannot ride a full-strength anchor.
  assert microspine.seat_quality_scales_capacity is True
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
  assert microspine.limit_surface.capacity_ramp_floor == 0.3
  assert microspine.limit_surface.capacity_ramp_time_s == 0.25
  assert microspine.limit_surface.capacity_ramp_shear_n == 10.0
  assert microspine.limit_surface.redraw_f_t_max_per_latch is True
  assert microspine.limit_surface.hold_grace_steps == 14  # re-seat window only
  assert microspine.limit_surface.moment_capacity_enable is True
  # Billing spans derive from the robot XML's sphere grid (pad_grid_spans),
  # not from hand-maintained constants; arm = spacing / 2 (discrete 2-row
  # aggregate fold), torquescale = reference-hold-calibrated compliance.
  assert abs(microspine.limit_surface.pad_span_rake_m - 0.0746) < 1.0e-3
  assert abs(microspine.limit_surface.pad_span_lateral_m - 0.096) < 1.0e-3
  assert microspine.limit_surface.moment_arm_fraction == 0.5
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
  assert motion.sampling_mode == "uniform"  # DeepMimic RSI.
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
  assert tuple(collision.size) == (0.5, 1.5, 0.1)


def test_climb_reward_contract_is_deepmimic_split():
  cfg = phybot_c2_climb_wall_env_cfg(play=False)

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
  # 0.25 s around a scheduled flip — the countdown signal the policy sees —
  # because debounce + cooldown make instant compliance impossible.
  assert cfg.rewards["motion_engage_mismatch"].weight == -1.0
  assert (
    cfg.rewards["motion_engage_mismatch"].params["grace_time_s"] == 0.25
  )

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
  assert cfg.rewards["ee_slide_penalty"].weight == -0.1
  # -0.1 (was -0.01): policy-mean chatter (3-4 Hz, +-4-5 cm hand buzz on
  # motionref8) was economically free and kept blowing the 60 ms engage
  # window; walk prices the same term at -0.25/-0.3.
  assert cfg.rewards["action_rate_l2"].weight == -0.1
  assert cfg.rewards["bounded_dof_acc_l2"].weight == -0.02
  assert cfg.rewards["bounded_dof_acc_l2"].params["max_cost"] == 10.0
  assert cfg.rewards["dof_pos_limits"].weight == -5.0
  assert cfg.rewards["termination_penalty"].weight == -500.0

  # Stall guard plus DeepMimic early termination; the gait disqualifications
  # left with the command. 1.5 cm / 8 s (was 3 cm / 5 s): with the drive
  # reference-locked there is no farmable stationary income left to execute,
  # and the tight bar was killing the stance -> shift -> climb ladder
  # (motionref7 learned "never truly latch" because standing properly was a
  # death sentence).
  assert "no_progress" in cfg.terminations
  # 18% of the commanded rate — see env_cfgs: strict enough to execute the
  # measured cycle-in-place basin, loose enough not to execute early
  # learners a third of the way up the ladder.
  assert cfg.terminations["no_progress"].params["minimum_progress"] == 0.010
  assert cfg.terminations["no_progress"].params["stall_time_s"] == 8.0
  # The activation gate must sit BELOW the commanded climb rate or the
  # executioner is silently disarmed (the 10 mm/s bar vs the moment-safe
  # clip's 4.84 mm/s let cycling-in-place survive full episodes).
  assert cfg.terminations["no_progress"].params["command_threshold"] == 0.003
  clip_vz = float(cfg.commands["climb_velocity"].ranges.lin_vel_z[0])
  assert clip_vz > cfg.terminations["no_progress"].params["command_threshold"]
  assert cfg.terminations["bad_tracking"].params["threshold"] == 0.7
  assert "gait_phase_timeout" not in cfg.terminations
  assert "gait_curriculum_success" not in cfg.terminations


def test_climb_curriculum_is_fixed_iteration_gravity_schedule():
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  runner_cfg = phybot_c2_ppo_runner_cfg()

  # No performance gate: graduation-by-height let a 20x-overspeed scramble
  # promote 45->75 deg in 400 iterations (motionref1). Fixed-iteration rungs
  # stay; the ladder is compressed and regains a 75-deg rung (motionref12):
  # post-sensor-fix the policy saturates 45 in ~550 iters and steps 45->60
  # with zero shock, while the 60-trained policy probed at 90 ripped 65% of
  # its releases — pre-unload must be learned on a slope, not off a cliff.
  # The vertical stage runs exactly 90 (jitter above 90 would mean
  # overhangs) and owns iters 3000-12000.
  assert "climb_difficulty" not in cfg.curriculum
  term = cfg.curriculum["gravity_angle"]
  stages = term.params["angle_stages"]
  assert [(stage["iter"], stage["angle_deg"]) for stage in stages] == [
    (0, 45.0),
    (600, 60.0),
    (1500, 75.0),
    (3000, 90.0),
  ]
  assert term.params["angle_jitter_deg"] == 5.0
  # The iteration clock derives from common_step_counter; it must match the
  # runner's rollout length or every rung shifts.
  assert term.params["steps_per_iter"] == runner_cfg.num_steps_per_env == 96
  assert runner_cfg.max_iterations == 12_000

  play_cfg = phybot_c2_climb_wall_env_cfg(play=True)
  play_term = play_cfg.curriculum["gravity_angle"]
  assert play_term.params["angle_stages"] == [{"iter": 0, "angle_deg": 90.0}]
  # Evaluation/diagnostic rollouts must be deterministic at the pinned angle.
  assert play_term.params["angle_jitter_deg"] == 0.0


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
