"""The climbing scene and simulation: robot and wall, contact sensors, scene edits, buffers."""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, SensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  configure_phybot_c2_floor_geom,
  configure_phybot_c2_terrain_cfg,
)
from scr_phybot_mjab.assets.wall import get_wall_cfg
from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.contact import get_sphere_geom_names
from scr_phybot_mjab.microspine.contact_sensor import WrenchContactSensorCfg
from scr_phybot_mjab.microspine.spec import (
  ToeContact,
  add_arm_wall_contact_pairs,
  add_attachment_equalities,
  add_contact_pairs,
  add_pad_wall_contact_excludes,
  add_toe_contact_pairs,
  toe_geom_name,
)

# Sensor names are manager keys, public like the command names in commands.py.
MICROSPINE_CONTACT_SENSOR = "microspine_wall_contact"
TOE_CONTACT_SENSOR = "climb_toe_contact"
UNDESIRED_CONTACT_SENSOR = "climb_undesired_contact"

DECIMATION = 10
"""Physics steps per policy step (0.002 s timestep, 50 Hz policy)."""


def climb_scene(robot_cfg: EntityCfg, microspine_cfg: MicrospineCfg, toe_contacts: tuple[ToeContact, ...]) -> SceneCfg:
  terrain = TerrainEntityCfg(terrain_type="plane")
  configure_phybot_c2_terrain_cfg(terrain)

  def _spec_fn(spec: mujoco.MjSpec) -> None:
    add_contact_pairs(spec, microspine_cfg)
    add_pad_wall_contact_excludes(spec, microspine_cfg)
    add_toe_contact_pairs(spec, microspine_cfg, toe_contacts)
    if microspine_cfg.arm_wall_contact_geoms:
      add_arm_wall_contact_pairs(spec, microspine_cfg)
    add_attachment_equalities(spec, microspine_cfg)
    # A real floor: a fall lands instead of dropping into the void. The
    # wall's base line sits on this plane at every wall angle, so the floor
    # and the tilted wall coexist without sweeping the spawn region.
    configure_phybot_c2_floor_geom(spec, enable_contact=True)

  return SceneCfg(
    num_envs=2048,
    env_spacing=8.0,
    terrain=terrain,
    # The wall collides as an analytic plane (its finite box is visual), so
    # the robot's collision model stays the same as walk's, with no convex
    # CCD substitutions.
    entities={"robot": robot_cfg, "wall": get_wall_cfg(analytic_collision_plane=True)},
    sensors=climb_sensors(microspine_cfg, toe_contacts),
    spec_fn=_spec_fn,
  )


def climb_sensors(microspine_cfg: MicrospineCfg, toe_contacts: tuple[ToeContact, ...]) -> tuple[SensorCfg, ...]:
  proxies = get_sphere_geom_names(microspine_cfg)
  return (
    WrenchContactSensorCfg(
      name=MICROSPINE_CONTACT_SENSOR,
      primary=ContactMatch(mode="geom", pattern=proxies, entity="robot"),
      secondary=ContactMatch(mode="geom", pattern="wall_geom", entity="wall"),
      # A flat box can generate several contacts: sum their forces instead of
      # keeping only the strongest point. Netforce is in world coordinates;
      # the affordance projects it onto the live wall normal.
      fields=("found", "force", "normal", "tangent", "torque", "pos"),
      reduce="netforce",
      num_slots=1,
      global_frame=False,
      history_length=DECIMATION,
      reference_sites={name: f"robot/{name.rsplit('_c_', 1)[0]}_contact_center" for name in proxies},
    ),
    # Foot-toe brace force, for the scheduled toe-brace shaping.
    ContactSensorCfg(
      name=TOE_CONTACT_SENSOR,
      primary=ContactMatch(mode="geom", pattern=tuple(toe_geom_name(toe) for toe in toe_contacts), entity="robot"),
      secondary=ContactMatch(mode="geom", pattern="wall_geom", entity="wall"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      global_frame=False,
    ),
    ContactSensorCfg(
      name=UNDESIRED_CONTACT_SENSOR,
      primary=ContactMatch(mode="body", pattern=".*", entity="robot", exclude=(".*_spine.*", ".*_ankle_roll")),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      history_length=DECIMATION,
    ),
  )


def climb_sim() -> SimulationCfg:
  return SimulationCfg(
    mujoco=MujocoCfg(timestep=0.002, cone="elliptic", integrator="implicitfast", impratio=10.0, ccd_iterations=100),
    # Four flat boxes per pad can produce 64 simultaneous pad-wall contacts
    # before any floor or toe contact, and a four-world 60-degree reference
    # rollout reached 513 contacts in total; this leaves headroom for
    # ordinary body and floor contacts after a loss of grip.
    nconmax=256,
    njmax=512,
    contact_sensor_maxmatch=500,
  )


def climb_viewer() -> ViewerConfig:
  """A fixed world camera looking at the wall from the side."""
  return ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    lookat=(1.5, 0.0, 1.2),
    distance=5.0,
    elevation=-20.0,
    azimuth=135.0,
  )
