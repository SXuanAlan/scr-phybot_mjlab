"""The brick scene: the robot, the brick, contact sensors, scene edits and the simulation."""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, SensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from scr_phybot_mjab.assets.brick import BrickCfg
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  configure_phybot_c2_floor_geom,
  configure_phybot_c2_terrain_cfg,
  get_phybot_c2_robot_cfg,
)
from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.contact import get_sphere_geom_names
from scr_phybot_mjab.microspine.contact_sensor import WrenchContactSensorCfg
from scr_phybot_mjab.microspine.spec import add_attachment_equalities, add_contact_pairs
from scr_phybot_mjab.tasks.brick.geometry import brick_robot_collision
from scr_phybot_mjab.tasks.common.phybot_c2 import real_pd_robot

from .microspine import HANDS


def brick_robot_cfg() -> EntityCfg:
  """The C2 with the brick task's collision set and the deployed PD damping."""
  robot = get_phybot_c2_robot_cfg()
  robot.collisions = (brick_robot_collision(),)
  return real_pd_robot(robot)


def brick_scene(brick: BrickCfg, mc: MicrospineCfg, robot: EntityCfg) -> SceneCfg:
  terrain = TerrainEntityCfg(terrain_type="plane")
  configure_phybot_c2_terrain_cfg(terrain)
  hands = HANDS

  def spec_fn(spec: mujoco.MjSpec):
    configure_phybot_c2_floor_geom(spec)
    add_contact_pairs(spec, mc)
    add_attachment_equalities(spec, mc)
    for hand in hands:
      # The separate four face proxies own brick compression; retain the
      # physical backing plate against ground and other robot bodies.
      spec.add_exclude(bodyname1=f"robot/{hand}", bodyname2="brick/brick")
    spec.worldbody.add_site(
      name="brick_goal",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=[brick.start_xy[0] + brick.goal_offset_xy[0], brick.start_xy[1] + brick.goal_offset_xy[1], 0.002],
      size=[brick.size[0] / 2 + 0.03, brick.size[1] / 2 + 0.03, 0.002],
      rgba=[0.2, 0.8, 0.3, 0.25],
    )

  return SceneCfg(
    num_envs=1024,
    env_spacing=3.0,
    terrain=terrain,
    entities={"robot": robot, "brick": brick.entity()},
    sensors=brick_sensors(mc),
    spec_fn=spec_fn,
  )


def brick_sensors(mc: MicrospineCfg) -> tuple[SensorCfg, ...]:
  proxy_names = get_sphere_geom_names(mc)
  return (
    WrenchContactSensorCfg(
      name="microspine_brick_contact",
      primary=ContactMatch(mode="geom", pattern=proxy_names, entity="robot"),
      secondary=ContactMatch(mode="geom", pattern="brick_geom", entity="brick"),
      history_length=10,
      num_slots=1,
      reference_sites={name: f"robot/{name.split('_c_')[0]}_contact_center" for name in proxy_names},
    ),
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(mode="subtree", pattern=r"^(left|right)_ankle_roll$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="brick_ground_contact",
      primary=ContactMatch(mode="geom", pattern="brick_geom", entity="brick"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="illegal_ground_contact",
      primary=ContactMatch(mode="body", pattern=r"^(?!(?:left|right)_(?:ankle_roll|foot_spine.*)$).*$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      history_length=10,
    ),
    ContactSensorCfg(
      name="illegal_brick_contact",
      primary=ContactMatch(mode="body", pattern=r"^(?!.*hand_spine).*$", entity="robot"),
      secondary=ContactMatch(mode="geom", pattern="brick_geom", entity="brick"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      history_length=10,
    ),
  )


def brick_sim() -> SimulationCfg:
  return SimulationCfg(
    mujoco=MujocoCfg(timestep=0.002, integrator="implicitfast", cone="elliptic", impratio=10.0),
    njmax=640,
    nconmax=256,
    contact_sensor_maxmatch=512,
  )


def brick_viewer() -> ViewerConfig:
  return ViewerConfig(body_name="torso", distance=2.5, elevation=-15.0, azimuth=135.0)
