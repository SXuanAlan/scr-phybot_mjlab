import mjlab  # noqa: F401
import mujoco
import pytest

import scr_phybot_mjab.tasks  # noqa: F401  # Initialize the registry before task modules.
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
  assert microspine.limit_surface.f_t_max_range == (360.0, 360.0)
  # The passive wrist gimbal owns pad compliance, so the weld is rotationally
  # rigid and failure is judged by moment billing only.
  assert microspine.anchor_torquescale == 1.0
  assert microspine.limit_surface.tilt_peel_max_deg_range == (0.0, 0.0)
  assert tuple(cfg.actions) == ("joint_pos",)
  # The motion command must follow the affordance: its reset performs RSI and
  # seeds the per-env latch state AFTER the affordance's own reset ran.
  assert tuple(cfg.commands) == (
    "climb_velocity",
    "microspine_affordance",
    "climb_motion",
  )
  assert cfg.metrics["microspine_engage_event_rate_hz"].params == {"command_name": "microspine_affordance"}


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
    (clip_average_climb_rate(CAD_CLIMB_CORNERS60_MOTION),) * 2
  )
  assert cfg.scene.num_envs == main.scene.num_envs
  assert cfg.episode_length_s == main.episode_length_s
  assert cfg.actions == main.actions
  assert tuple(cfg.commands) == tuple(main.commands)
  assert tuple(cfg.observations["actor"].terms) == tuple(main.observations["actor"].terms)
  assert tuple(cfg.observations["critic"].terms) == tuple(main.observations["critic"].terms)
  assert {key: term.weight for key, term in cfg.rewards.items()} == {
    key: term.weight for key, term in main.rewards.items()
  }
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
  assert (
    phybot_c2_climb_wall_corners60_env_cfg().curriculum["wall_angle"].params["angle_stages"][0]["angle_deg"] == 45.0
  )
  assert phybot_c2_climb_wall_env_cfg().curriculum["wall_angle"].params["angle_stages"][-1]["angle_deg"] == 90.0


@pytest.mark.parametrize("play", [False, True])
def test_corner_climbing_80_task_keeps_its_angle_in_training_and_play(play):
  from mjlab.tasks.registry import load_env_cfg

  from scr_phybot_mjab.assets.motions import CAD_CLIMB_CORNERS80_MOTION

  cfg = load_env_cfg("Mjlab-Climb-Wall-Phybot-C2-Corners-A80", play=play)
  microspine = cfg.commands["microspine_affordance"].microspine_cfg
  assert microspine.limit_surface.unit_layout == "corners"
  assert microspine.engage_unit_preload_n == 10.0
  wall = cfg.curriculum["wall_angle"].params
  assert wall["angle_stages"] == [{"iter": 0, "angle_deg": 80.0}]
  assert wall["angle_jitter_deg"] == 0.0
  assert cfg.events["microspine_grip"].params["microspine_cfg"].limit_surface.unit_layout == "corners"
  assert cfg.commands["climb_motion"].motion_file == str(CAD_CLIMB_CORNERS80_MOTION)


@pytest.mark.parametrize("wall_angle,layout", [(60.0, "corners"), (80.0, "corners"), (90.0, "all")])
def test_empirical_climbing_rewards_unloaded_release_at_each_angle_and_layout(wall_angle, layout):
  cfg = phybot_c2_climb_wall_env_cfg(wall_angle_deg=wall_angle, unit_layout=layout)
  release = cfg.rewards["microspine_release_load"]
  assert callable(release.func) and release.weight == 1.0
  assert release.params["shear_scale_n"] == release.params["lateral_scale_n"] == 2.0
  assert release.params["shear_cost"] > release.params["lateral_cost"] > 0
  assert release.params["minimum_supports"] == 3
  assert cfg.rewards["unsafe_microspine_release"].weight < 0
  assert cfg.commands["microspine_affordance"].microspine_cfg.engage_unit_preload_n == 10.0


def test_full_array_layout_uses_the_3x4_unit_grid():
  cfg = phybot_c2_climb_wall_env_cfg(unit_layout="all")
  ls = cfg.commands["microspine_affordance"].microspine_cfg.limit_surface
  assert (ls.unit_layout, ls.unit_rows_rake, ls.unit_cols_lateral) == ("all", 3, 4)


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


def test_climb_curriculum_retrains_from_scratch():
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  runner_cfg = phybot_c2_ppo_runner_cfg()

  # The config retrains from iter 0. Every hop overlaps its neighbor's
  # jitter pool; the iterations are ceilings driven by mastery gates.
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
  assert tuple(cfg.curriculum) == ("wall_angle",)

  # Play/eval pins to the ladder's terminal values: exams always run the
  # honest physics at the target angle regardless of training progress.
  play_cfg = phybot_c2_climb_wall_env_cfg(play=True)
  play_term = play_cfg.curriculum["wall_angle"]
  assert play_term.params["angle_stages"] == [{"iter": 0, "angle_deg": 90.0}]


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
