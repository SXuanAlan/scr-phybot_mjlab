"""The squat scene: the standing robot on a flat floor, its contact sensors, the simulation and the viewer."""

from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_STANDING_JOINT_POS, get_phybot_c2_robot_cfg
from scr_phybot_mjab.tasks.common.phybot_c2 import real_pd_robot

TORSO_BODY = "torso"
_FEET_ROOT_BODIES = r"^(left_ankle_roll|right_ankle_roll)$"
_ILLEGAL_GROUND_BODIES = (
  r"^(left_knee|right_knee"
  r"|left_shoulder_roll|right_shoulder_roll"
  r"|left_shoulder_yaw|right_shoulder_yaw"
  r"|left_elbow_pitch|right_elbow_pitch"
  r"|torso|torso_upper)$"
)


def squat_robot_cfg() -> EntityCfg:
  """The C2 standing in the clip's first frame, with the deployed PD damping."""
  robot = get_phybot_c2_robot_cfg()
  robot.init_state = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.726),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos=PHYBOT_C2_STANDING_JOINT_POS,
    joint_vel={".*": 0.0},
  )
  return real_pd_robot(robot)


def squat_scene(robot: EntityCfg) -> SceneCfg:
  return SceneCfg(
    num_envs=1,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": robot},
    sensors=(
      # The stationary-feet reward reads it.
      ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="subtree", pattern=_FEET_ROOT_BODIES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
      ),
      # The self-collision cost counts the substeps of its force history.
      ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern=TORSO_BODY, entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern=TORSO_BODY, entity="robot"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        history_length=4,
      ),
      # A knee, an arm or the torso on the floor ends the episode.
      ContactSensorCfg(
        name="illegal_ground_contact",
        primary=ContactMatch(mode="body", pattern=_ILLEGAL_GROUND_BODIES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=4,
      ),
    ),
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
