"""Friction model validation task configuration.

Factory function to create a friction test config:
- A fixed slope (adjustable angle) in the scene
- A block resting on the slope with gravity
- A configurable normal force pressing the block into the slope

Use this to verify MuJoCo's friction model against analytical predictions:
  - Static friction threshold: F_tangential <= mu * F_normal
  - On a slope with extra normal force N_ext:
      gravity tangential component = m*g*sin(theta)
      total normal force = m*g*cos(theta) + N_ext
      block slides when: m*g*sin(theta) > mu * (m*g*cos(theta) + N_ext)
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from spider_rl_mjlab.tasks.friction_test import mdp


def make_friction_test_env_cfg(
  slope_angle_deg: float = 30.0,
  normal_force: float = 10.0,
  friction_coef: float = 0.5,
  block_mass: float = 1.0,
) -> ManagerBasedRlEnvCfg:
  """Create friction model validation test configuration.

  Args:
    slope_angle_deg: Slope inclination angle in degrees.
    normal_force: External normal force pressing block into slope [N].
    friction_coef: Friction coefficient for the block-slope contact pair.
    block_mass: Mass of the block [kg].
  """

  num_envs = 1
  env_spacing = 5.0

  ##
  # Observations — minimal, just track block position and velocity.
  ##

  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "base_lin_vel": ObservationTermCfg(
          func=envs_mdp.base_lin_vel,
          params={"asset_cfg": SceneEntityCfg("block")},
        ),
        "projected_gravity": ObservationTermCfg(
          func=envs_mdp.projected_gravity,
          params={"asset_cfg": SceneEntityCfg("block")},
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions — dummy (no actuators on the block, but the framework requires an action space).
  ##

  actions = {}

  ##
  # Rewards — none needed for physics validation.
  ##

  rewards = {}

  ##
  # Events
  ##

  events = {
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default,
      mode="reset",
    ),
    # Apply constant normal force pressing block into slope.
    "normal_force": EventTermCfg(
      func=mdp.apply_normal_force,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("block"),
        "force_magnitude": normal_force,
        "slope_angle_deg": slope_angle_deg,
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
  }

  ##
  # Build config
  ##

  return ManagerBasedRlEnvCfg(
    decimation=4,
    episode_length_s=10.0,
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=env_spacing,
      terrain=TerrainEntityCfg(terrain_type="plane"),
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.001,
        iterations=200,
        cone="elliptic",
        ccd_iterations=2000,
        gravity=(0.0, 0.0, -9.81),
        impratio=1.0,
      ),
      njmax=1000,
      nconmax=500,
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(1.0, 0.0, 0.5),
      distance=4.0,
      elevation=-25.0,
      azimuth=150.0,
    ),
    observations=observations,
    actions=actions,
    rewards=rewards,
    events=events,
    terminations=terminations,
  )
