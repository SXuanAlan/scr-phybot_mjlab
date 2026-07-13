"""Full-body squat motion-tracking task configuration (plain PPO).

One episode is one standalone squat clip.  Robot-specific configs set the exact
motion duration and track every policy joint plus the reference base and planted
feet; non-policy neck joints are held by their actuators and excluded.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from scr_phybot_mjab.tasks.squat import mdp
from scr_phybot_mjab.tasks.squat.mdp import SquatMotionCommandCfg

_COMMAND_NAME = "motion"


def make_squat_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base squat motion-tracking task configuration."""

  ##
  # Observations
  ##

  # No base_lin_vel in the actor (the real robot has no linear-velocity estimate);
  # the phase clock is the only task-conditioning input. Matches the walk actor.
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},  # Set per-robot.
      noise=Unoise(n_min=-0.15, n_max=0.15),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,  # Set per-robot (framequat sensor).
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
      params={"command_name": _COMMAND_NAME},
    ),
  }

  critic_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},  # Set per-robot.
    ),
    **actor_terms,
  }

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
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),  # Set per-robot.
      scale=0.5,  # Set per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    _COMMAND_NAME: SquatMotionCommandCfg(
      entity_name="robot",
      # Never resample mid-episode; every episode starts at motion frame zero.
      resampling_time_range=(1.0e9, 1.0e9),
      motion_files=(),  # Set per-robot (standalone squat clip).
      joint_names=(),  # Set per-robot (the tracked policy joints).
      sampling_mode="start",
      # Reset exactly to motion frame zero. Robustness comes from physical
      # parameter randomization, not from corrupting the motion timeline.
      pose_range={},
      velocity_range={},
      joint_position_range=(0.0, 0.0),
    )
  }

  ##
  # Events
  ##

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(3.0, 6.0),
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

  rewards: dict[str, RewardTermCfg] = {
    # Track every motion joint. The neck is not policy-controlled and is absent
    # from the motion's joints_list.
    "track_legs": RewardTermCfg(
      func=mdp.track_joint_pos,
      weight=3.0,
      params={"command_name": _COMMAND_NAME, "std": 0.3, "joint_names": ()},  # legs, set per-robot.
    ),
    # Arm joints need independent dense scores. A single exponential over all
    # eight joints saturated near zero and let the policy sacrifice its arms to
    # optimize the planted-foot and base rewards.
    "track_arms": RewardTermCfg(
      func=mdp.track_joint_pos_cauchy,
      weight=3.0,
      params={
        "command_name": _COMMAND_NAME,
        "std": 0.20,
        "joint_names": (),  # Arms, set per-robot.
        "metric_prefix": "arm",
      },
    ),
    "arm_tracking_error": RewardTermCfg(
      func=mdp.joint_pos_tracking_huber_error,
      weight=-2.0,
      params={
        "command_name": _COMMAND_NAME,
        "beta": 0.25,
        "joint_names": (),  # Arms, set per-robot.
      },
    ),
    "track_arm_velocity": RewardTermCfg(
      func=mdp.track_joint_vel_cauchy,
      weight=1.0,
      params={
        "command_name": _COMMAND_NAME,
        "std": 1.0,
        "joint_names": (),  # Arms, set per-robot.
        "metric_prefix": "arm",
      },
    ),
    "arm_action_limits": RewardTermCfg(
      func=mdp.joint_action_limit_cost,
      weight=-0.5,
      params={
        "action_name": "joint_pos",
        "joint_names": (),  # Arms, set per-robot.
        "max_abs_actions": (),  # Motion-derived, set per-robot.
        "global_clip": 6.0,
        "metric_prefix": "arm",
      },
    ),
    "track_waist": RewardTermCfg(
      func=mdp.track_joint_pos,
      weight=1.0,
      params={"command_name": _COMMAND_NAME, "std": 0.2, "joint_names": ()},
    ),
    "track_joint_velocity": RewardTermCfg(
      func=mdp.track_joint_vel,
      weight=2.0,
      params={"command_name": _COMMAND_NAME, "std": 2.0, "joint_names": None},
    ),
    "track_base_position": RewardTermCfg(
      func=mdp.track_base_position,
      weight=2.0,
      params={"command_name": _COMMAND_NAME, "std": 0.05},
    ),
    "track_base_orientation": RewardTermCfg(
      func=mdp.track_base_orientation,
      weight=1.0,
      params={"command_name": _COMMAND_NAME, "std": 0.3},
    ),
    "track_base_lin_vel": RewardTermCfg(
      func=mdp.track_base_lin_vel,
      weight=1.0,
      params={"command_name": _COMMAND_NAME, "std": 0.3},
    ),
    "track_feet_stationary": RewardTermCfg(
      func=mdp.track_feet_stationary,
      weight=1.5,
      params={
        "sensor_name": "feet_ground_contact",
        "std": 0.10,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "track_feet_anchor": RewardTermCfg(
      func=mdp.track_feet_anchor,
      weight=3.0,
      params={
        "command_name": _COMMAND_NAME,
        "std": 0.04,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.5),
    "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-20.0),
    # Regularization supplements the explicit full-body tracking terms.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.2),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
    "dof_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-1.0e-7),
    "energy": RewardTermCfg(
      func=mdp.energy, weight=-5e-4, params={"asset_cfg": SceneEntityCfg("robot")}
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-3.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # The reference lean peaks near 20 deg, so 70 deg leaves ample squat margin
    # while terminating genuinely unstable poses.
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
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
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=1.0,  # Set from the robot motion metadata.
    is_finite_horizon=True,
  )
