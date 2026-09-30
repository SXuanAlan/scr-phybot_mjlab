"""Velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from scr_phybot_mjab.tasks.walk import mdp
from scr_phybot_mjab.tasks.walk.mdp import UniformVelocityCommandCfg

_TORSO_BODY = "torso"
_FOOT_SITE_NAMES = ("left_foot", "right_foot")
_FEET_GROUND_CONTACT_SENSOR = "feet_ground_contact"
_SELF_COLLISION_SENSOR = "self_collision"


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration."""

  ##
  # Observations
  ##

  # No base_lin_vel: the real robot has no linear-velocity estimate, so the
  # actor must work without it (the critic still gets it as privileged info).
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.15, n_max=0.15),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
  }

  critic_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    **actor_terms,
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": _FEET_GROUND_CONTACT_SENSOR},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": _FEET_GROUND_CONTACT_SENSOR},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": _FEET_GROUND_CONTACT_SENSOR},
    ),
  }

  # Official PhyBot Lab feeds 10 stacked frames to both actor and critic.
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=10,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=10,
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.25,
      rel_heading_envs=0.3,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "velocity_range": {}},  # Set per-robot.
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_scale,
      mode="reset",
      params={
        "position_range": (1.0, 1.0),  # Set per-robot.
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 1.0),  # Set per-robot.
      params={"velocity_range": {}},  # Set per-robot.
    ),
    "body_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (1.0, 1.0),  # Set per-robot.
        "shared_random": True,
      },
    ),
    "base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": (0.0, 0.0),  # Set per-robot.
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "upright": RewardTermCfg(
      func=mdp.flat_orientation,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=(_TORSO_BODY,)),
      },
    ),
    # Without AMP this is the only gait-style prior: weight 1.0 and tight leg
    # stds (MJLab G1 values) to keep the legs near the default pose.
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {".*": 0.05},
        "std_walking": {
          r".*hip_pitch.*": 0.3,
          r".*hip_roll.*": 0.15,
          r".*hip_yaw.*": 0.15,
          r".*knee.*": 0.35,
          r".*ankle_pitch.*": 0.25,
          r".*ankle_roll.*": 0.1,
          r".*waist_yaw.*": 0.2,
          r".*neck.*": 0.1,
          r".*shoulder_pitch.*": 0.15,
          r".*shoulder_roll.*": 0.15,
          r".*shoulder_yaw.*": 0.1,
          r".*elbow.*": 0.15,
        },
        "std_running": {
          r".*hip_pitch.*": 0.5,
          r".*hip_roll.*": 0.2,
          r".*hip_yaw.*": 0.2,
          r".*knee.*": 0.6,
          r".*ankle_pitch.*": 0.35,
          r".*ankle_roll.*": 0.15,
          r".*waist_yaw.*": 0.3,
          r".*neck.*": 0.2,
          r".*shoulder_pitch.*": 0.5,
          r".*shoulder_roll.*": 0.2,
          r".*shoulder_yaw.*": 0.15,
          r".*elbow.*": 0.35,
        },
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    # Anti-crouch: measured zero-action settled base height in the real-robot
    # zero pose (ankle_pitch -0.295, keyframe z 0.726, PD sag included).
    # Absolute world z — only valid on flat terrain.
    "base_height": RewardTermCfg(
      func=mdp.base_height,
      weight=0.5,
      params={"target_height": 0.713, "std": 0.1},
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=(_TORSO_BODY,))},
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.04),
    # Weights below follow the official PhyBot Lab LiteRewardCfg.
    "energy": RewardTermCfg(
      func=mdp.energy,
      weight=-5e-4,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    # -5e-4 (official AMP value) made heel-balancing profitable without the
    # AMP style prior: zero ankle torque is worth ~30% of the tracking reward.
    "ankle_torque": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-5e-5,
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*_ankle_pitch", ".*_ankle_roll"))},
    ),
    "dof_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-1.5e-8),
    "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-20.0),
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.15,
      params={
        "sensor_name": _FEET_GROUND_CONTACT_SENSOR,
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.5,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITE_NAMES),
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-0.25,
      params={
        "sensor_name": _FEET_GROUND_CONTACT_SENSOR,
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITE_NAMES),
      },
    ),
    # Flat-foot contact: discourages the heel-point balancing that the old
    # high-friction microspine contact made profitable.
    "feet_flat": RewardTermCfg(
      func=mdp.feet_flat_orientation,
      weight=0.5,
      params={
        "std": math.sqrt(0.1),
        "sensor_name": _FEET_GROUND_CONTACT_SENSOR,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    # Stance width: keep the feet at least 0.15 m apart laterally (shoe shells
    # touch at ~0.10 m; default stance is 0.297 m, so normal gait is unaffected).
    "feet_distance": RewardTermCfg(
      func=mdp.feet_lateral_distance,
      weight=-5.0,
      params={
        "min_dist": 0.15,
        "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITE_NAMES),
      },
    ),
    # Gentle calm-stance preference only. Static standing is impossible at the
    # deploy ankle gains (2x50 N*m/rad << m*g*h ~ 271), so the policy must
    # balance actively (weight shifts / stepping in place). At -0.3 the motion
    # cost dwarfed the -20 fall penalty and the policy learned to freeze and
    # tip over instead.
    "stand_still": RewardTermCfg(
      func=mdp.stand_still_joint_vel,
      weight=-0.05,
      params={"command_name": "twist", "command_threshold": 0.05},
    ),
    "stand_feet_contact": RewardTermCfg(
      func=mdp.feet_contact_when_standing,
      weight=0.3,
      params={
        "command_name": "twist",
        "sensor_name": _FEET_GROUND_CONTACT_SENSOR,
        "command_threshold": 0.05,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": _FEET_GROUND_CONTACT_SENSOR,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITE_NAMES),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": _FEET_GROUND_CONTACT_SENSOR,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    # Sole self-collision deterrent (there is no self-collision termination:
    # it starved the policy of standing experience; see phybot env cfg).
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-3.0,
      params={"sensor_name": _SELF_COLLISION_SENSOR, "force_threshold": 10.0},
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
          {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
          {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=mdp.rough_terrain_cfg(max_init_terrain_level=5),
      sensors=(),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
