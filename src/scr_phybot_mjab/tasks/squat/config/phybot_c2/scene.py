"""The squat scene: the standing robot on a flat floor, the simulation and the viewer."""

from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import get_phybot_c2_robot_cfg
from scr_phybot_mjab.tasks.common.phybot_c2 import TORSO_BODY, ground_contact_sensors, real_pd_robot


def squat_robot_cfg() -> EntityCfg:
  """The C2 at its standing keyframe, the clip's first frame, with the deployed PD damping."""
  return real_pd_robot(get_phybot_c2_robot_cfg())


def squat_scene(robot: EntityCfg) -> SceneCfg:
  return SceneCfg(
    num_envs=1,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": robot},
    sensors=ground_contact_sensors(),
  )


def squat_sim() -> SimulationCfg:
  """0.005 s physics steps; with decimation 4 the policy runs at the clip's 50 fps."""
  return SimulationCfg(
    nconmax=35,
    njmax=1500,
    mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
  )


def squat_viewer() -> ViewerConfig:
  return ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name=TORSO_BODY,
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
  )
