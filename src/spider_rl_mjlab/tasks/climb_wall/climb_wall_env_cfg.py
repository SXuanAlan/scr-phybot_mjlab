"""Wall-climbing task configuration.

Factory function to create a base climb-wall task config.
Robot-specific configurations call the factory and customize as needed.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from spider_rl_mjlab.tasks.climb_wall import mdp


def make_climb_wall_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base wall-climbing task configuration."""

  num_envs = 128
  env_spacing = 4.0

  ##
  # Observations
  ##

  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),  # (num_envs, 3) [vx, vy, vz]
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),  # (num_envs, 3) [wx, wy, wz]
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),  # (num_envs, 3) [gx, gy, gz]
        "joint_pos_rel": ObservationTermCfg(func=mdp.joint_pos_rel),  # (num_envs, num_dof)
        "joint_vel_rel": ObservationTermCfg(func=mdp.joint_vel_rel),  # (num_envs, num_dof)
        "last_action": ObservationTermCfg(func=mdp.last_action),  # (num_envs, num_actions)
      },
      concatenate_terms=True,  # Concatenate all terms into a single vector.
      enable_corruption=False, # Enable observation corruption (e.g., adding noise).
    ),
  }

  ##
  # Actions
  ##

  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    ),
  }

  ##
  # Rewards
  ##

  robot_cfg = SceneEntityCfg("robot")

  rewards = {
    # Primary: climb higher
    "base_height": RewardTermCfg(
      func=mdp.base_height,
      weight=5.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Orientation: stay upright
    "torso_orientation": RewardTermCfg(
      func=mdp.torso_orientation,
      weight=40.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Penalize body angular velocity
    "torso_ang_vel_xy": RewardTermCfg(
      func=mdp.torso_ang_vel_xy,
      weight=-2.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Penalize lateral drift
    "lin_vel_xy": RewardTermCfg(
      func=mdp.lin_vel_xy,
      weight=-2.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Smoothness: penalize jerky actions
    "action_rate": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-0.6,
    ),
    # Energy: penalize large torques
    "joint_torques": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-1e-6,
    ),
    # Joint limit penalties
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
    ),
  }

  ##
  # Events
  ##

  events = {
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default,
      mode="reset",
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (-0.1, 0.1),
      },
    ),
    "reset_robot_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (0.0, 0.0),
          "y": (-0.3, 0.3),
          "z": (0.0, 0.0),
        },
        "velocity_range": {
          "x": (0.0, 0.0),
          "y": (0.0, 0.0),
          "z": (0.0, 0.0),
          "roll": (0.0, 0.0),
          "pitch": (0.0, 0.0),
          "yaw": (0.0, 0.0),
        },
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_height": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.3},
    ),
  }

  ##
  # Build config
  ##

  return ManagerBasedRlEnvCfg(
    decimation=10,
    episode_length_s=10.0,
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=env_spacing,
      terrain=TerrainEntityCfg(terrain_type="plane"),
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.002, ccd_iterations=200),
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(1.5, 0.0, 2.0),
      distance=8.0,
      elevation=-20.0,
      azimuth=135.0,
    ),
    observations=observations,
    actions=actions,
    rewards=rewards,
    events=events,
    terminations=terminations,
  )
