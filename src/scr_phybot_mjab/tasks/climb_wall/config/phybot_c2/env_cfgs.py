"""PHYBot C2 wall-climbing environment configuration.

Reward set is a port of IsaacLab's quadruped velocity-tracking template
(``locomotion/velocity/velocity_env_cfg.py``) with two task-specific swaps:

  * Velocity tracking uses a climb-specific world-frame command fixed at
    ``vz_cmd = 0.08`` m/s.
  * Contact-sensor signals → microspine ``engaged`` flag for all 4 spines.

C2-specific notes:
  * 4 grip end-effectors: left/right hand_spine + left/right foot_spine.
    Each spine has a 2x2 sphere grid (16 spheres total).
  * No ``waist_roll`` joint/body — Spot/Anymal-style torso-deviation terms
    are not migrated here (intentionally minimal).
"""

from __future__ import annotations

import dataclasses

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_ACTION_SCALE,
  PHYBOT_C2_POLICY_JOINT_NAMES,
  configure_phybot_c2_floor_geom,
  configure_phybot_c2_terrain_cfg,
  get_phybot_c2_robot_cfg,
)
from scr_phybot_mjab.assets.wall import get_wall_cfg
from scr_phybot_mjab.microspine.config import (
  MicrospineAffordanceCfg,
  MicrospineCfg,
)
from scr_phybot_mjab.microspine.contact import (
  add_contact_pairs,
  get_sphere_geom_names,
)
from scr_phybot_mjab.microspine.constraint import (
  add_attachment_equalities,
  make_constraint_event_cfg,
)
from scr_phybot_mjab.tasks.climb_wall import mdp
from scr_phybot_mjab.tasks.climb_wall.climb_wall_env_cfg import make_climb_wall_env_cfg
from scr_phybot_mjab.tasks.climb_wall.mdp.curriculums import gravity_angle


def _policy_joint_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    joint_names=PHYBOT_C2_POLICY_JOINT_NAMES,
    preserve_order=True,
  )


def phybot_c2_climb_wall_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the constraint-based PHYBot C2 wall-climbing configuration."""
  velocity_command_name = "climb_velocity"
  microspine_affordance_name = "microspine_affordance"
  microspine_contact_sensor_name = "microspine_wall_contact"
  microspine_event_name = "microspine_grip"
  vz_cmd = 0.08

  microspine_cfg = MicrospineCfg(
    ee_body_names=(
      "left_hand_spine",
      "right_hand_spine",
      "left_foot_spine",
      "right_foot_spine",
    ),
    spheres_per_ee_rows=2,
    spheres_per_ee_cols=2,
  )
  # Constraint grip engages from compression count + face alignment; the
  # limit surface owns release.
  # A real spine pad locks position and orientation across the contact patch.
  # Capacity follows the pad-local rake direction.
  microspine_cfg.anchor_lock_rotation = True
  microspine_cfg.limit_surface.rake_mode = "ee_local"

  cfg = make_climb_wall_env_cfg()
  configure_phybot_c2_terrain_cfg(cfg.scene.terrain)
  terms = cfg.observations["actor"].terms
  terms["base_lin_vel"].func = mdp.builtin_sensor
  terms["base_lin_vel"].params = {"sensor_name": "robot/baselink-velocity"}
  terms["base_ang_vel"].func = mdp.builtin_sensor
  terms["base_ang_vel"].params = {"sensor_name": "robot/baselink-gyro"}
  terms["projected_gravity"].func = mdp.projected_gravity_from_framequat_sensor
  terms["projected_gravity"].params = {"sensor_name": "robot/baselink-quat"}
  # Match MJLab/IsaacLab velocity tracking style: no explicit gait phase
  # observation. Gait should emerge from velocity tracking + grip shaping.
  cfg.observations["actor"].terms.pop("gait_clock", None)

  cfg.commands[microspine_affordance_name] = MicrospineAffordanceCfg(
    microspine_cfg=microspine_cfg,
    contact_sensor_name=microspine_contact_sensor_name,
    physics_event_name=microspine_event_name,
  )

  robot_cfg = get_phybot_c2_robot_cfg()
  wall_cfg = get_wall_cfg()

  # Override init state at task level (do NOT mutate the shared keyframe
  # singleton on robot_cfg.init_state — other c2 tasks reuse it).
  robot_cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(1.245, 0.0, 1.0),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
      "left_hip_pitch": -0.815,
      "left_hip_roll": 0.09,
      "left_hip_yaw": -0.265,
      "left_knee": 0.765,
      "left_ankle_pitch": 1.34,
      "left_ankle_roll": 0.0,
      "right_hip_pitch": -0.815,
      "right_hip_roll": -0.09,
      "right_hip_yaw": 0.265,
      "right_knee": 0.765,
      "right_ankle_pitch": 1.34,
      "right_ankle_roll": 0.0,
      "waist_yaw": 0.0,
      "neck_yaw": 0.0,
      "neck_pitch": 0.0,
      "left_shoulder_pitch": -2.0,
      "left_shoulder_roll": -0.17,
      "left_shoulder_yaw": 0.0,
      "left_elbow_pitch": -1.14,
      "right_shoulder_pitch": -2.0,
      "right_shoulder_roll": 0.17,
      "right_shoulder_yaw": 0.0,
      "right_elbow_pitch": -1.14,
    },
    joint_vel={".*": 0.0},
  )

  # Wall: keep the default WALL_OFFSET = (2.0, 0.0, 7.5) — same as tune_pose,
  # so the spawned base position lines up with what was visualized there.

  cfg.scene.entities = {
    "robot": robot_cfg,
    "wall": wall_cfg,
  }
  cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch, 500)
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name=microspine_contact_sensor_name,
      primary=ContactMatch(
        mode="geom",
        pattern=get_sphere_geom_names(microspine_cfg),
        entity="robot",
      ),
      secondary=ContactMatch(mode="geom", pattern="wall_geom", entity="wall"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      global_frame=False,
      history_length=10,
    ),
  )

  # Ordinary per-sphere contacts handle compression and pre-engage sliding;
  # per-EE equalities provide grip.
  def _spec_fn(spec: mujoco.MjSpec) -> None:
    add_contact_pairs(spec, microspine_cfg)
    add_attachment_equalities(spec, microspine_cfg)
    # Disable the aligned terrain ground plane — gravity-tilt curriculum
    # replaces the floor; this forces the robot to actually grip the wall.
    configure_phybot_c2_floor_geom(spec, enable_contact=False)

  cfg.scene.spec_fn = _spec_fn

  cfg.events[microspine_event_name] = make_constraint_event_cfg(
    microspine_cfg,
    affordance_name=microspine_affordance_name,
  )

  # ------------------------------------------------------------------------
  # Rewards — Anymal D quadruped template (locomotion/velocity), with climb
  # velocity command tracking and engage signal instead of contact. Track and
  # gait weights are down-scaled so dense velocity tracking does not swamp the
  # gait signal.
  # ------------------------------------------------------------------------
  robot_cfg_e = SceneEntityCfg("robot")
  cfg.rewards = {
    # -- task: commanded world-frame climb velocity target. Split x/y from z
    # so lateral drift correction is softer than upward climb-rate tracking.
    "track_lin_vel_xy": RewardTermCfg(
      func=mdp.climb_linear_velocity_xy_gauss,
      weight=4.0,
      params={
        "asset_cfg": robot_cfg_e,
        "command_name": velocity_command_name,
        "sigma": 0.05,
      },
    ),
    "track_lin_vel_z": RewardTermCfg(
      func=mdp.climb_linear_velocity_z_gauss,
      weight=8.0,
      params={
        "asset_cfg": robot_cfg_e,
        "command_name": velocity_command_name,
        "sigma": 0.05,
      },
    ),
    "climb_height_progress": RewardTermCfg(
      func=mdp.climb_height_progress(),
      weight=10.0,
      params={
        "asset_cfg": robot_cfg_e,
        "command_name": velocity_command_name,
        "command_threshold": 0.01,
        "max_normalized_progress": 1.5,
      },
    ),
    # -- penalties (Anymal weights verbatim except as noted).
    "ang_vel_xy_l2": RewardTermCfg(
      func=mdp.ang_vel_xy_l2,
      weight=-0.05,
      params={"asset_cfg": robot_cfg_e},
    ),
    "dof_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-1.0e-5,
      params={"asset_cfg": robot_cfg_e},
    ),
    "dof_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-1.0e-5,
      params={"asset_cfg": robot_cfg_e},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    # -- gait shaping from the 2026-04-28_14-51-31 run: dense min-mode
    # timer gated on exactly two engaged EEs (two-anchor / two-swing).
    "ee_grip_time": RewardTermCfg(
      func=mdp.hand_grip_time_min_climb(),
      weight=10.0,
      params={
        "tau": 0.5,
        "vz_cmd": vz_cmd,
        "cmd_eps": 0.0,
        "event_term_name": microspine_event_name,
        "target_engaged_count": 2,
      },
    ),
    "ee_slide_penalty": RewardTermCfg(
      func=mdp.ee_slide_penalty,
      weight=-0.1,
      params={
        "asset_cfg": robot_cfg_e,
        "hand_body_names": microspine_cfg.ee_body_names,
        "event_term_name": microspine_event_name,
      },
    ),
    # Spine contact face (local -y) staying flush against the wall — MIN
    # across 4 EEs (strictest: any 1 spine rotated off-axis drops reward).
    "ee_face_parallel_to_wall": RewardTermCfg(
      func=mdp.ee_face_parallel_to_wall,
      weight=2.0,
      params={
        "asset_cfg": robot_cfg_e,
        "ee_body_names": microspine_cfg.ee_body_names,
        "face_local_axis": microspine_cfg.face_local_axis,
      },
    ),
    # -- optional Anymal terms, default disabled (weight 0) — left in the
    # cfg so they can be flipped on without restructuring.
    "flat_orientation_l2": RewardTermCfg(
      func=mdp.flat_orientation_l2,
      weight=0.0,
      params={"asset_cfg": robot_cfg_e},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=0.0,
      params={"asset_cfg": robot_cfg_e},
    ),
    # -- termination penalty (common addition not in Anymal generic).
    "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    # -- limit-surface release penalty.
    # Rewards are dt-scaled (step_dt = 0.02), so -5.0 = -0.1 per release
    # event. Bump toward -50 if the signal is too weak to shape loading.
    "microspine_forced_release_penalty": RewardTermCfg(
      func=mdp.microspine_forced_release_penalty,
      weight=-5.0,
      params={"command_name": microspine_affordance_name},
    ),
  }

  cfg.metrics = {
    "microspine_engage_rate": MetricsTermCfg(
      func=mdp.microspine_engage_rate,
      params={"command_name": microspine_affordance_name},
    ),
    "microspine_forced_release_rate": MetricsTermCfg(
      func=mdp.microspine_forced_release_rate,
      params={"command_name": microspine_affordance_name},
    ),
    "microspine_hold_margin_engaged": MetricsTermCfg(
      func=mdp.microspine_hold_margin_engaged,
      params={"command_name": microspine_affordance_name},
    ),
  }

  cfg.curriculum = {
    "gravity_angle": CurriculumTermCfg(
      func=gravity_angle,
      params={
        "angle_stages": [
          {"iter": 0, "angle_deg": 30.0},
          {"iter": 3000, "angle_deg": 60.0},
          {"iter": 5000, "angle_deg": 90.0},
          {"iter": 10000, "angle_deg": 90.0},
        ],
      },
    )
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = PHYBOT_C2_POLICY_JOINT_NAMES
  joint_pos_action.preserve_order = True
  joint_pos_action.scale = PHYBOT_C2_ACTION_SCALE

  actor_terms = cfg.observations["actor"].terms
  actor_terms["joint_pos_rel"].params = {"asset_cfg": _policy_joint_cfg()}
  actor_terms["joint_vel_rel"].params = {"asset_cfg": _policy_joint_cfg()}

  # Asymmetric actor-critic: the critic group is the actor group plus
  # privileged microspine state (hold margin, anchor loads, forced releases)
  # that a real robot cannot sense. Term cfgs are re-instantiated so the two
  # groups never share mutable cfg objects. Built after all actor-term edits
  # above so the copies pick them up.
  critic_terms = {
    name: dataclasses.replace(term) for name, term in actor_terms.items()
  }
  critic_terms["microspine_hold_margin"] = ObservationTermCfg(
    func=mdp.microspine_hold_margin,
    params={"command_name": microspine_affordance_name},
  )
  critic_terms["microspine_anchor_load"] = ObservationTermCfg(
    func=mdp.microspine_anchor_load,
    params={"command_name": microspine_affordance_name},
    scale=0.01,  # Newtons → O(1).
  )
  critic_terms["microspine_forced_release"] = ObservationTermCfg(
    func=mdp.microspine_forced_release,
    params={"command_name": microspine_affordance_name},
  )
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    microspine_cfg.debug = True
    cfg.events[microspine_event_name].params["microspine_cfg"].debug = True

  return cfg
