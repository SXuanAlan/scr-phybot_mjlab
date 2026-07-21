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
  assert microspine.engage_min_spheres == 2
  assert microspine.limit_surface.violation_consecutive_steps == 2
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
  assert cfg.rewards["motion_ee_pos"].weight == 1.5
  assert cfg.rewards["motion_root_ori"].weight == 0.5
  assert cfg.rewards["motion_engage_mismatch"].weight == -0.25

  # Height/velocity remain the climbing drive. The per-step cap must clear
  # the reference's 0.8 s step-up burst (peak ~10x the average rate).
  assert cfg.rewards["climb_height_progress"].weight == 2.0
  assert (
    cfg.rewards["climb_height_progress"].params["max_normalized_progress"]
    == 12.0
  )
  assert cfg.rewards["signed_height_progress"].weight == 0.5

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
  assert cfg.rewards["unsafe_microspine_release"].weight == -1.0
  assert cfg.rewards["ee_slide_penalty"].weight == -0.1
  assert cfg.rewards["bounded_dof_acc_l2"].weight == -0.02
  assert cfg.rewards["bounded_dof_acc_l2"].params["max_cost"] == 10.0
  assert cfg.rewards["dof_pos_limits"].weight == -5.0
  assert cfg.rewards["termination_penalty"].weight == -500.0

  # Stall guard plus DeepMimic early termination; the gait disqualifications
  # left with the command. minimum_progress 0.03/5s is the anti-farming
  # executioner: partial tracking income while creeping at the old 1 cm/5 s
  # threshold out-earned honest climbing's discovery phase.
  assert "no_progress" in cfg.terminations
  assert cfg.terminations["no_progress"].params["minimum_progress"] == 0.03
  assert cfg.terminations["bad_tracking"].params["threshold"] == 0.7
  assert "gait_phase_timeout" not in cfg.terminations
  assert "gait_curriculum_success" not in cfg.terminations


def test_climb_curriculum_is_gravity_only_gated_on_height_gain():
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  runner_cfg = phybot_c2_ppo_runner_cfg()

  term = cfg.curriculum["climb_difficulty"]
  stages = term.params["stages"]
  assert [stage["angle_deg"] for stage in stages] == [
    45.0,
    45.0,
    60.0,
    75.0,
    90.0,
  ]
  # 0.06 m = one full reference cycle; 0.18 m = three chained cycles.
  assert [stage["min_height_gain"] for stage in stages] == [
    0.06,
    0.18,
    0.18,
    0.18,
    0.18,
  ]
  assert term.params["advance_success_rate"] == 0.7
  assert term.params["motion_command_name"] == "climb_motion"
  assert runner_cfg.max_iterations == 12_000

  play_cfg = phybot_c2_climb_wall_env_cfg(play=True)
  play_stages = play_cfg.curriculum["climb_difficulty"].params["stages"]
  assert play_stages == [stages[-1]]


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
