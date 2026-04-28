"""PHYBot C2 velocity-tracking environments.

This intentionally follows MJLab's official Unitree G1 velocity pipeline and
only swaps in the PHYBot C2 asset plus robot-specific body/site/joint names.
"""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from spider_rl_mjlab.tasks.walk import mdp
from spider_rl_mjlab.tasks.walk.mdp import UniformVelocityCommandCfg
from spider_rl_mjlab.tasks.walk.walk_env_cfg import make_velocity_env_cfg

from spider_rl_mjlab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_ACTION_SCALE,
  get_phybot_c2_robot_cfg,
  get_spec as get_phybot_c2_spec,
)

_TORSO_BODY = "torso"
_RAYCAST_BODY = "torso"
_FEET_ROOT_BODIES = r"^(left_ankle_roll|right_ankle_roll)$"
_FOOT_SITE_NAMES = ("left_foot", "right_foot")
_FOOT_FRICTION_GEOMS = (r"^(left|right)_ankle_roll_collision$",)
_BROKEN_SPINE_BODIES = (
  "left_foot_spine",
  "right_foot_spine",
  "left_hand_spine",
  "right_hand_spine",
)
_BROKEN_SPINE_GEOM_PREFIXES = (
  "left_foot_spine",
  "right_foot_spine",
  "left_hand_spine",
  "right_hand_spine",
)

_STANDING_JOINT_POS: dict[str, float] = {
  "left_hip_pitch": -0.12,
  "left_hip_roll": 0.0,
  "left_hip_yaw": 0.0,
  "left_knee": 0.24,
  "left_ankle_pitch": -0.12,
  "left_ankle_roll": 0.0,
  "right_hip_pitch": -0.12,
  "right_hip_roll": 0.0,
  "right_hip_yaw": 0.0,
  "right_knee": 0.24,
  "right_ankle_pitch": -0.12,
  "right_ankle_roll": 0.0,
  "waist_yaw": 0.0,
  "neck_yaw": 0.0,
  "neck_pitch": 0.0,
  "left_shoulder_pitch": 0.0,
  "left_shoulder_roll": 0.0,
  "left_shoulder_yaw": 0.0,
  "left_elbow_pitch": 0.0,
  "right_shoulder_pitch": 0.0,
  "right_shoulder_roll": 0.0,
  "right_shoulder_yaw": 0.0,
  "right_elbow_pitch": 0.0,
}

_WALK_STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.70),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos=_STANDING_JOINT_POS,
  joint_vel={".*": 0.0},
)


def _get_walk_spec():
  spec = get_phybot_c2_spec()

  # The raw asset includes its own floor; MJLab tasks provide terrain separately.
  # Keeping both can make contact sensors miss ground contacts against "terrain".
  for geom in list(spec.geoms):
    if geom.name == "floor":
      spec.delete(geom)

  # MJLab uses the first keyframe as init_state. Remove the XML keyframe so the
  # task-level standing keyframe is the only reset/default pose.
  for key in list(spec.keys):
    spec.delete(key)

  # The converted asset has hand/foot spine bodies parented as knee/elbow
  # siblings with placeholder 1 kg inertials. They are useful for climbing
  # tasks, but for flat walking they create misleading contacts and inertia.
  for body in spec.bodies:
    if body.name in _BROKEN_SPINE_BODIES:
      body.mass = 0.05
      body.ipos = [0.0, 0.0, 0.0]
      body.inertia = [1.0e-4, 1.0e-4, 1.0e-4]

  for geom in spec.geoms:
    if any(geom.name.startswith(prefix) for prefix in _BROKEN_SPINE_GEOM_PREFIXES):
      geom.contype = 0
      geom.conaffinity = 0

  # Official velocity rewards use foot sites for height/slip terms. Use the
  # ankle-roll bodies as feet for walking so contact, height and slip all refer
  # to the same articulated subtree.
  for body_name, site_name in (
    ("left_ankle_roll", "left_foot"),
    ("right_ankle_roll", "right_foot"),
  ):
    body = next(body for body in spec.bodies if body.name == body_name)
    body.add_site(name=site_name, pos=[0.04, 0.0, -0.04], size=[0.02])

  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="site_imu",
  )
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="site_imu",
  )
  spec.add_sensor(
    name="imu_lin_acc",
    type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="site_imu",
  )
  spec.add_sensor(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname=_TORSO_BODY,
  )

  return spec


def _get_phybot_c2_velocity_robot_cfg():
  robot_cfg = get_phybot_c2_robot_cfg()
  robot_cfg.init_state = _WALK_STANDING_KEYFRAME
  robot_cfg.spec_fn = _get_walk_spec
  return robot_cfg


def phybot_c2_velocity_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PHYBot C2 rough-terrain velocity config using MJLab's G1 pipeline."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": _get_phybot_c2_velocity_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = _RAYCAST_BODY

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=_FEET_ROOT_BODIES,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=_TORSO_BODY, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=_TORSO_BODY, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PHYBOT_C2_ACTION_SCALE

  cfg.viewer.body_name = _TORSO_BODY

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.8

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = _FOOT_SITE_NAMES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_FRICTION_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = (_TORSO_BODY,)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.45,
    r".*hip_roll.*": 0.22,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.55,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.16,
    r".*waist_yaw.*": 0.25,
    r".*neck.*": 0.1,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.7,
    r".*hip_roll.*": 0.3,
    r".*hip_yaw.*": 0.3,
    r".*knee.*": 0.8,
    r".*ankle_pitch.*": 0.45,
    r".*ankle_roll.*": 0.22,
    r".*waist_yaw.*": 0.35,
    r".*neck.*": 0.2,
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = (_TORSO_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (_TORSO_BODY,)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = _FOOT_SITE_NAMES

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["pose"].weight = 0.6
  cfg.rewards["action_rate_l2"].weight = -0.04
  cfg.rewards["air_time"].weight = 0.15
  cfg.rewards["foot_clearance"].weight = -1.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def phybot_c2_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PHYBot C2 flat-terrain velocity config using MJLab's G1 pipeline."""
  cfg = phybot_c2_velocity_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
