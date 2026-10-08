"""The walking scene: the robot on the C2 floor, the simulation and the viewer."""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  configure_phybot_c2_floor_geom,
  configure_phybot_c2_terrain_cfg,
  get_phybot_c2_robot_cfg,
)
from scr_phybot_mjab.tasks.common.phybot_c2 import TORSO_BODY, ground_contact_sensors, real_pd_robot


def walk_robot_cfg(real_pd_damping: bool) -> EntityCfg:
  """The C2 at its standing keyframe, with the deployed PD table's damping if ``real_pd_damping``."""
  robot = get_phybot_c2_robot_cfg()
  return real_pd_robot(robot) if real_pd_damping else robot


def _c2_floor(spec: mujoco.MjSpec) -> None:
  """The C2 floor geom (size, friction, material), colliding with the robot."""
  configure_phybot_c2_floor_geom(spec, enable_contact=True)


def walk_scene(robot: EntityCfg) -> SceneCfg:
  terrain = TerrainEntityCfg(terrain_type="plane")
  configure_phybot_c2_terrain_cfg(terrain)
  return SceneCfg(
    num_envs=1,
    extent=2.0,
    terrain=terrain,
    entities={"robot": robot},
    sensors=ground_contact_sensors(),
    spec_fn=_c2_floor,
  )


def walk_sim() -> SimulationCfg:
  return SimulationCfg(
    nconmax=35,
    njmax=1500,
    mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
  )


def walk_viewer() -> ViewerConfig:
  return ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name=TORSO_BODY,
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
  )
