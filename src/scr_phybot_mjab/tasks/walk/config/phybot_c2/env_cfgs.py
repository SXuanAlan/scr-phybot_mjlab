"""PHYBot C2 velocity-tracking environments.

This intentionally follows MJLab's official Unitree G1 velocity pipeline and
only swaps in the PHYBot C2 asset plus robot-specific body/site/joint names.
"""

from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from scr_phybot_mjab.tasks.walk import mdp
from scr_phybot_mjab.tasks.walk.mdp import UniformVelocityCommandCfg
from scr_phybot_mjab.tasks.walk.walk_env_cfg import make_velocity_env_cfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_ACTION_SCALE,
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_STANDING_JOINT_POS,
  configure_phybot_c2_floor_geom,
  configure_phybot_c2_terrain_cfg,
  get_phybot_c2_robot_cfg,
)

_TORSO_BODY = "torso"
_FEET_ROOT_BODIES = r"^(left_ankle_roll|right_ankle_roll)$"
# Bodies that must never touch the ground (official PhyBot Lab terminates on
# knee/shoulder/elbow/base/waist contact). Only bodies with active collision
# geoms are listed; waist_yaw/shoulder_pitch collisions are commented out in
# the MJCF and cannot generate contacts.
_ILLEGAL_GROUND_BODIES = (
  r"^(left_knee|right_knee"
  r"|left_shoulder_roll|right_shoulder_roll"
  r"|left_shoulder_yaw|right_shoulder_yaw"
  r"|left_elbow_pitch|right_elbow_pitch"
  r"|torso|torso_upper)$"
)
_FOOT_SITE_NAMES = ("left_foot", "right_foot")
_COLLISION_GEOMS = (r".*_collision$",)
_BASE_QUAT_SENSOR = "robot/baselink-quat"
_BASE_LIN_VEL_SENSOR = "robot/baselink-velocity"
_BASE_ANG_VEL_SENSOR = "robot/baselink-gyro"

_WALK_STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.70),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos=PHYBOT_C2_STANDING_JOINT_POS,
  joint_vel={".*": 0.0},
)


def phybot_c2_velocity_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PHYBot C2 rough-terrain velocity config using MJLab's G1 pipeline."""
  cfg = make_velocity_env_cfg()

  ##
  # Robot
  ##

  robot_cfg = get_phybot_c2_robot_cfg()
  robot_cfg.init_state = _WALK_STANDING_KEYFRAME
  cfg.scene.entities = {"robot": robot_cfg}

  ##
  # Observations
  ##

  actor_terms = cfg.observations["actor"].terms
  actor_terms["base_lin_vel"].params = {"sensor_name": _BASE_LIN_VEL_SENSOR}
  actor_terms["base_ang_vel"].params = {"sensor_name": _BASE_ANG_VEL_SENSOR}
  actor_terms["projected_gravity"].func = mdp.projected_gravity_from_framequat_sensor
  actor_terms["projected_gravity"].params = {"sensor_name": _BASE_QUAT_SENSOR}

  # Keep policy action dimensions and joint observations in the same joint order.
  for term_name in ("joint_pos", "joint_vel"):
    actor_terms[term_name].params = {
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=PHYBOT_C2_POLICY_JOINT_NAMES, preserve_order=True
      )
    }

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = _FOOT_SITE_NAMES

  ##
  # Actions
  ##

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = PHYBOT_C2_POLICY_JOINT_NAMES
  joint_pos_action.preserve_order = True
  joint_pos_action.scale = PHYBOT_C2_ACTION_SCALE

  ##
  # Viewer and command visualization
  ##

  cfg.viewer.body_name = _TORSO_BODY
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.8

  ##
  # Contact sensors
  ##

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    ContactSensorCfg(
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
    ),
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern=_TORSO_BODY, entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern=_TORSO_BODY, entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
    ContactSensorCfg(
      name="illegal_ground_contact",
      primary=ContactMatch(
        mode="body",
        pattern=_ILLEGAL_GROUND_BODIES,
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      history_length=4,
    ),
  )
  cfg.terminations["self_collision"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "self_collision", "force_threshold": 10.0},
  )
  cfg.terminations["illegal_ground_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "illegal_ground_contact", "force_threshold": 1.0},
  )

  ##
  # Terrain and Domain Randomization
  ##

  mdp.set_rough_terrain_curriculum(cfg, enabled=True)

  # base_height uses absolute world z, which is invalid on generated rough
  # terrain. The flat config re-enables it after switching to a plane.
  cfg.rewards["base_height"].weight = 0.0

  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "yaw": (-1.0, 1.0),
  }
  reset_base.params["velocity_range"] = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (-0.1, 0.1),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.2, 0.2),
  }

  reset_joints = cfg.events["reset_robot_joints"]
  reset_joints.params["position_range"] = (0.8, 1.2)
  reset_joints.params["velocity_range"] = (0.0, 0.0)

  # More frequent pushes with vertical and angular components (MJLab G1 style)
  # force the policy to learn firm footing instead of a compliant crouch.
  push_robot = cfg.events["push_robot"]
  push_robot.interval_range_s = (1.0, 3.0)
  push_robot.params["velocity_range"] = {
    "x": (-0.8, 0.8),
    "y": (-0.8, 0.8),
    "z": (-0.4, 0.4),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
  }

  body_friction = cfg.events["body_friction"]
  body_friction.params["asset_cfg"].geom_names = _COLLISION_GEOMS
  # MJLab exposes MuJoCo geom friction here; official C2 also randomizes
  # dynamic friction/restitution buckets, which do not have a direct DR hook.
  body_friction.params["ranges"] = (0.6, 1.0)

  base_mass = cfg.events["base_mass"]
  base_mass.params["asset_cfg"].body_names = (_TORSO_BODY,)
  base_mass.params["ranges"] = (-5.0, 5.0)

  ##
  # Play mode
  ##

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    mdp.configure_play_rough_terrain(cfg)

  return cfg


def phybot_c2_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PHYBot C2 flat-terrain velocity config using MJLab's G1 pipeline."""
  cfg = phybot_c2_velocity_rough_env_cfg(play=play)

  mdp.configure_flat_terrain(
    cfg,
    terrain_cfg_fn=configure_phybot_c2_terrain_cfg,
    spec_fn=lambda spec: configure_phybot_c2_floor_geom(spec, enable_contact=True),
  )

  cfg.rewards["base_height"].weight = 0.5

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
