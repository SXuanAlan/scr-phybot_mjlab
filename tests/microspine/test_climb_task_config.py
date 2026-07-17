import mujoco

from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.rl_cfg import (
  phybot_c2_ppo_runner_cfg,
)
from scr_phybot_mjab.tasks.walk.config.phybot_c2.env_cfgs import (
  PHYBOT_C2_WALK_COLLISION,
)


def test_climb_first_stage_engages_all_anchors_without_grip_action():
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
  assert tuple(cfg.commands) == (
    "climb_velocity",
    "microspine_affordance",
    "climb_gait",
  )
  assert cfg.metrics["microspine_engage_event_rate_hz"].params == {
    "command_name": "microspine_affordance"
  }


def test_climb_play_defaults_to_one_env_without_sync_debug():
  cfg = phybot_c2_climb_wall_env_cfg(play=True)
  command_cfg = cfg.commands["microspine_affordance"].microspine_cfg
  event_cfg = cfg.events["microspine_grip"].params["microspine_cfg"]

  assert cfg.scene.num_envs == 1
  assert not command_cfg.debug
  assert not event_cfg.debug


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


def test_climb_reward_contract_has_no_static_positive_income():
  cfg = phybot_c2_climb_wall_env_cfg(play=False)

  assert "track_lin_vel_xy" not in cfg.rewards
  assert "track_lin_vel_z" not in cfg.rewards
  assert "ee_grip_time" not in cfg.rewards
  assert "ee_face_parallel_to_wall" not in cfg.rewards
  assert "microspine_forced_release_penalty" not in cfg.rewards
  assert cfg.rewards["climb_height_progress"].weight == 2.0
  assert cfg.rewards["signed_height_progress"].weight == 0.5
  assert cfg.rewards["confirmed_anchor_advance"].weight == 0.5
  assert cfg.rewards["commanded_anchor_advance"].weight == 4.0
  assert cfg.rewards["productive_gait_cycle"].weight == 2.0
  assert cfg.rewards["unscheduled_microspine_release"].weight == -2.0
  assert cfg.rewards["climb_gait_support_shortfall"].weight == -2.0
  assert cfg.rewards["commanded_ee_target_error"].weight == -0.2
  assert "dof_acc_l2" not in cfg.rewards
  assert cfg.rewards["bounded_dof_acc_l2"].weight == -0.02
  assert cfg.rewards["bounded_dof_acc_l2"].params["max_cost"] == 10.0
  assert cfg.rewards["dof_pos_limits"].weight == -5.0
  assert cfg.rewards["termination_penalty"].weight == -500.0
  assert "no_progress" in cfg.terminations


def test_climb_curriculum_is_performance_gated_through_vertical_wall():
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  runner_cfg = phybot_c2_ppo_runner_cfg()

  term = cfg.curriculum["climb_difficulty"]
  stages = term.params["stages"]
  assert [stage["angle_deg"] for stage in stages] == [
    45.0,
    45.0,
    45.0,
    45.0,
    45.0,
    45.0,
    60.0,
    75.0,
    90.0,
  ]
  # Skill ladder v2 (2026-07-17): proven full-cycle stages, then a dedicated
  # hand rung (all anchors engaged, hands-only targets, tiny 2 cm step —
  # the one skill that never emerged), then full task.
  assert [s["required_productive_regrips"] for s in stages] == [
    1, 2, 1, 2, 4, 4, 4, 4, 4,
  ]
  hands = ("left_hand_spine", "right_hand_spine")
  assert "gait_ee_names" not in stages[0]
  assert stages[2]["gait_ee_names"] == hands
  assert stages[2]["step_height"] == 0.02
  assert stages[3]["gait_ee_names"] == hands
  assert "gait_ee_names" not in stages[4]  # full cycle resumes
  assert stages[5]["initial_engaged_ee_names"] == hands
  assert term.params["advance_success_rate"] == 0.7
  assert runner_cfg.max_iterations == 12_000


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
  assert cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor == 0.9


def test_climb_disobedience_is_terminal_and_press_is_shaped():
  cfg = phybot_c2_climb_wall_env_cfg(play=False)
  # Disqualification, not a fine: reward-level penalties lost the economics
  # against stretch-farming twice (2026-07-15).
  assert "gait_phase_timeout" in cfg.terminations
  assert cfg.terminations["gait_phase_timeout"].time_out is False
  press = cfg.rewards["climb_seat_press_shaping"]
  assert press.weight > 0
  assert press.params["force_scale"] == 12.0

  play_cfg = phybot_c2_climb_wall_env_cfg(play=True)
  assert "gait_phase_timeout" not in play_cfg.terminations
