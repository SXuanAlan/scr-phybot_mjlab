"""Microspine adhesion test task configuration.

Factory function to create a base microspine test config.
Uses a single forearm+hand against a wall. Gravity disabled; 10N in +x for 1s.
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

from spider_rl_mjlab.tasks.microspine_test import mdp


def make_microspine_test_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base microspine adhesion test configuration."""

  num_envs = 128
  env_spacing = 3.0

  ##
  # Observations
  ##

  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
        "joint_pos_rel": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
        "joint_vel_rel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
        "last_action": ObservationTermCfg(func=envs_mdp.last_action),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    ),
  }

  ##
  # Rewards
  ##

  robot_cfg = SceneEntityCfg("robot")

  rewards = {
    # Keep the hand on the wall.
    "hand_on_wall": RewardTermCfg(
      func=mdp.hand_on_wall,
      weight=10.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Maintain height against 300N pull.
    "hand_height": RewardTermCfg(
      func=mdp.hand_height,
      weight=5.0,
      params={"asset_cfg": robot_cfg},
    ),
    # Smoothness: penalize jerky actions.
    "action_rate": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-0.6,
    ),
    # Energy: penalize large torques.
    "joint_torques": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-1e-6,
    ),
    # Joint limit penalties.
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
          "y": (0.0, 0.0),
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
    # External force is handled by MicrospineController via xfrc_applied.
    # See env_cfgs.py for force params (total_force, total_force_after).
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_height": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0},
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
      mujoco=MujocoCfg(timestep=0.0005, iterations=100, cone="elliptic", ccd_iterations=2000, gravity=(0.0, 0.0, 0.0), impratio=5.0),
      njmax=1000,
      nconmax=500,
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(0.5, 0.0, 1.0),
      distance=3.0,
      elevation=-20.0,
      azimuth=135.0,
    ),
    observations=observations,
    actions=actions,
    rewards=rewards,
    events=events,
    terminations=terminations,
  )
