"""Wall-climbing task configuration.

Factory function to create a base climb-wall task config.
Robot-specific configurations call the factory and customize as needed.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from scr_phybot_mjab.tasks.climb_wall import mdp


def make_climb_wall_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base wall-climbing task configuration."""

  num_envs = 4096
  env_spacing = 8.0
  velocity_command_name = "climb_velocity"

  ##
  # Observations
  ##

  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "joint_pos_rel": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel_rel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "last_action": ObservationTermCfg(func=mdp.last_action),
        "prev_action": ObservationTermCfg(func=mdp.prev_action),       # a_{t-1}
        "prev_prev_action": ObservationTermCfg(func=mdp.prev_prev_action),  # a_{t-2}
        "gait_clock": ObservationTermCfg(func=mdp.gait_clock),
        # Per-hand latch state (left, right). Lets the policy coordinate
        # motion with grip transitions.
        "hand_engaged_state": ObservationTermCfg(func=mdp.hand_engaged_state),
        # Robot z above spawn — critic needs this to value climbing.
        "height_above_spawn": ObservationTermCfg(func=mdp.height_above_spawn),
        # Commanded world-frame velocity (vx, vy, vz). The command manager
        # owns the target so obs and reward stay synchronized.
        "velocity_command": ObservationTermCfg(
          func=mdp.velocity_command,
          params={"command_name": velocity_command_name},
        ),
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
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    velocity_command_name: mdp.ClimbVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 8.0),
      ranges=mdp.ClimbVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.0),
        lin_vel_y=(0.0, 0.0),
        lin_vel_z=(0.08, 0.08),  # Currently fixed to 0.08
      ),
    )
  }

  ##
  # Robot-specific configs own concrete rewards and curriculum.
  ##

  rewards = {}
  curriculum = {}

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
  }

  ##
  # Terminations
  ##

  # ====================================================================
  # Terminations — G1-walking minimal set adapted for climbing.
  # G1 has: ``time_out`` + ``base_contact`` (torso touches ground).
  # Climb analog: ``time_out`` + ``base_height`` (robot fell off the wall
  # below the spawn floor). ``bad_state`` stays purely as NaN / physics
  # divergence safety.
  # ====================================================================
  robot_cfg = SceneEntityCfg("robot")
  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    # Climb analog of G1's ``base_contact``: the robot has lost grip and
    # fallen below the spawn floor. No terrain collision is active, so
    # without this termination a fallen robot floats into void.
    "base_height": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.3},
    ),
    # NaN / magnitude divergence guard. Not in G1 (IsaacLab has a global
    # NaN sink elsewhere); mjlab does not, so we keep this defensively.
    "bad_state": TerminationTermCfg(
      func=mdp.bad_state,
      params={"asset_cfg": robot_cfg, "max_velocity": 20.0},
    ),
  }

  ##
  # Build config
  ##

  return ManagerBasedRlEnvCfg(
    decimation=10,
    episode_length_s=30.0,
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=env_spacing,
      terrain=TerrainEntityCfg(terrain_type="plane"),
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.002, cone="elliptic", integrator="implicitfast", impratio=10.0, ccd_iterations=100),
      njmax=250,
      nconmax=50,
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(1.5, 0.0, 4.0),
      distance=12.0,
      elevation=-20.0,
      azimuth=135.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    curriculum=curriculum,
    events=events,
    terminations=terminations,
  )
