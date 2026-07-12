"""Squat motion-tracking task configuration (DeepMimic-style, plain PPO).

Factory for a base stand->squat->stand tracking task: the policy tracks a
lower-body + base-height reference driven by a phase clock, with the upper body
left free. Robot-specific configs call the factory and customize (see
``config/phybot_c2``). This mirrors ``walk_env_cfg.make_velocity_env_cfg`` and
mjlab's ``tracking_env_cfg.make_tracking_env_cfg``.
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
      # Never resample mid-episode; RSI happens on reset, the clip loops in-episode.
      resampling_time_range=(1.0e9, 1.0e9),
      motion_files=(),  # Set per-robot (standing clip + squat clip).
      joint_names=(),  # Set per-robot (the tracked policy joints).
      sampling_mode="uniform",
      # Init-state randomization aligned to walk's reset_base, kept RSI-reward-safe:
      # x/y are untracked; yaw is safe now that track_base_orientation is
      # yaw-invariant; z / roll / pitch position stay off (base height and tilt are
      # tightly tracked). Joint noise is additive (not walk's x0.8-1.2 scale, which
      # would override the RSI reference pose and collapse the tracking reward).
      pose_range={"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-1.0, 1.0)},
      velocity_range={
        "x": (-0.2, 0.2),
        "y": (-0.2, 0.2),
        "z": (-0.1, 0.1),
        "roll": (-0.2, 0.2),
        "pitch": (-0.2, 0.2),
        "yaw": (-0.2, 0.2),
      },
      joint_position_range=(-0.1, 0.1),
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
    # Track the legs, arms, and base; the waist is left free (reference ~ 0).
    "track_legs": RewardTermCfg(
      func=mdp.track_joint_pos,
      weight=3.0,
      params={"command_name": _COMMAND_NAME, "std": 0.3, "joint_names": ()},  # legs, set per-robot.
    ),
    # Arms follow the raise/lower choreography; looser std than legs so the policy
    # can still deviate them slightly for balance.
    "track_arms": RewardTermCfg(
      func=mdp.track_joint_pos,
      weight=1.5,
      params={"command_name": _COMMAND_NAME, "std": 0.4, "joint_names": ()},  # arms, set per-robot.
    ),
    "track_base_height": RewardTermCfg(
      func=mdp.track_base_height,
      weight=2.0,
      params={"command_name": _COMMAND_NAME, "std": 0.05},
    ),
    "track_base_orientation": RewardTermCfg(
      func=mdp.track_base_orientation,
      weight=1.0,
      params={"command_name": _COMMAND_NAME, "std": 0.3},
    ),
    "alive": RewardTermCfg(func=mdp.is_alive, weight=1.0),
    "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-20.0),
    # Regularization (also the only thing shaping the free upper body).
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
    "dof_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-1.5e-8),
    "energy": RewardTermCfg(
      func=mdp.energy, weight=-5e-4, params={"asset_cfg": SceneEntityCfg("robot")}
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # Relaxed vs walk (70 deg): a deep squat legitimately leans the torso forward.
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(85.0)},
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
    episode_length_s=15.0,
  )
