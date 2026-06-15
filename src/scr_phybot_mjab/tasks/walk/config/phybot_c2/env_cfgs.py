"""PHYBot C2 velocity-tracking environments.

This intentionally follows MJLab's official Unitree G1 velocity pipeline and
only swaps in the PHYBot C2 asset plus robot-specific body/site/joint names.
"""

from __future__ import annotations

import copy

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg
from scr_phybot_mjab.tasks.walk import mdp
from scr_phybot_mjab.tasks.walk.mdp import PresetVelocityCommandCfg, UniformVelocityCommandCfg
from scr_phybot_mjab.tasks.walk.walk_env_cfg import make_velocity_env_cfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_ACTION_SCALE,
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_REAL_PD_DAMPING_BY_GROUP,
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
_BASE_QUAT_SENSOR = "robot/baselink-quat"
_BASE_LIN_VEL_SENSOR = "robot/baselink-velocity"
_BASE_ANG_VEL_SENSOR = "robot/baselink-gyro"
_FEET_GROUND_CONTACT_SENSOR = "feet_ground_contact"

# Walk-specific collision config. The shared PHYBOT_C2_COLLISION carries
# microspine climbing values (foot sliding friction 30, condim 6) which let the
# policy balance on a heel point without ever slipping; on real ground
# (mu ~ 0.5-1.0) that gait falls over. The climb task keeps the shared config.
_FOOT_COLLISION_GEOMS = r"^(left|right)_ankle_roll_collision$"
_SPINE_PLATE_GEOMS = r"^(left|right)_(foot|hand)_spine_collision$"
PHYBOT_C2_WALK_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  # Spine plates are climbing hardware; the foot plate box bottom sits below
  # the shoe sole and must not collide with the ground while walking.
  contype={_SPINE_PLATE_GEOMS: 0, ".*": 1},
  conaffinity={_SPINE_PLATE_GEOMS: 0, ".*": 1},
  condim={".*": 3},
  friction={
    _FOOT_COLLISION_GEOMS: (0.7, 0.005, 0.0001),
    ".*": (1.0, 0.1, 0.1),
  },
  # Feet keep priority so their friction/solref win against the terrain plane.
  priority={_FOOT_COLLISION_GEOMS: 1},
  # The XML default solref timeconst (0.001) violates MuJoCo's stability
  # guideline (>= 2*timestep = 0.01 at dt=0.005) and caused visible foot
  # penetration. Critically damped at the guideline minimum.
  solref={".*": (0.01, 1.0)},
)

_WALK_STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.726),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos=PHYBOT_C2_STANDING_JOINT_POS,
  joint_vel={".*": 0.0},
)


def _apply_real_pd_actuator_damping(cfg: ManagerBasedRlEnvCfg) -> None:
  robot_cfg = copy.deepcopy(cfg.scene.entities["robot"])
  damping_by_target_expr = {
    (f".*_{group}",): damping for group, damping in PHYBOT_C2_REAL_PD_DAMPING_BY_GROUP.items()
  }

  updated = set()
  for actuator in robot_cfg.articulation.actuators:
    target_expr = tuple(actuator.target_names_expr)
    if target_expr in damping_by_target_expr:
      actuator.damping = damping_by_target_expr[target_expr]
      updated.add(target_expr)

  missing = set(damping_by_target_expr) - updated
  if missing:
    raise ValueError(f"Missing PHYBot C2 real-PD actuator damping targets: {sorted(missing)}")

  cfg.scene.entities["robot"] = robot_cfg


def phybot_c2_velocity_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PHYBot C2 rough-terrain velocity config using MJLab's G1 pipeline."""
  cfg = make_velocity_env_cfg()

  ##
  # Robot
  ##

  robot_cfg = get_phybot_c2_robot_cfg()
  robot_cfg.init_state = _WALK_STANDING_KEYFRAME
  robot_cfg.collisions = (PHYBOT_C2_WALK_COLLISION,)
  cfg.scene.entities = {"robot": robot_cfg}

  ##
  # Observations
  ##

  actor_terms = cfg.observations["actor"].terms
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

  critic_terms = cfg.observations["critic"].terms
  critic_terms["base_lin_vel"].params = {"sensor_name": _BASE_LIN_VEL_SENSOR}
  critic_terms["foot_height"].params["asset_cfg"].site_names = _FOOT_SITE_NAMES

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

  cfg.rewards["feet_flat"].params["asset_cfg"].body_names = (_FEET_ROOT_BODIES,)
  # Penalize only policy joints when standing; the PD-held neck is not ours.
  cfg.rewards["stand_still"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=PHYBOT_C2_POLICY_JOINT_NAMES
  )

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
  # No self-collision termination (matching the official Isaac config): leg
  # grazes while standing/stepping otherwise reset the env within 1-2 s, which
  # starves the policy of standing experience (three 12k runs converged to
  # leg-contact standing because of it). Self collisions are deterred by the
  # self_collisions reward instead; falls still terminate via fell_over /
  # illegal_ground_contact.
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
  # Exclude the feet: they get their own (wider) friction DR below.
  body_friction.params["asset_cfg"].geom_names = (
    r"^(?!(left|right)_ankle_roll_collision$).*_collision$",
  )
  # MJLab exposes MuJoCo geom friction here; official C2 also randomizes
  # dynamic friction/restitution buckets, which do not have a direct DR hook.
  body_friction.params["ranges"] = (0.6, 1.0)

  # Foot-ground sliding friction DR. Defined after body_friction so it wins
  # even if the negative lookahead above ever fails to exclude the feet.
  cfg.events["foot_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=(_FOOT_COLLISION_GEOMS,)),
      "operation": "abs",
      "ranges": (0.4, 1.0),
      "shared_random": True,
    },
  )

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


def phybot_c2_walk_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PHYBot C2 flat-terrain velocity config with AMP style reward."""
  cfg = phybot_c2_walk_env_cfg(play=play)

  amp_asset_cfg = SceneEntityCfg(
    "robot",
    joint_names=PHYBOT_C2_POLICY_JOINT_NAMES,
    preserve_order=True,
  )
  cfg.observations["amp"] = ObservationGroupCfg(
    terms={
      "joint_pos": ObservationTermCfg(
        func=mdp.joint_pos,
        params={"asset_cfg": amp_asset_cfg},
      ),
      "joint_vel": ObservationTermCfg(
        func=mdp.joint_vel,
        params={"asset_cfg": amp_asset_cfg},
      ),
      "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
      "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=None,
  )

  for reward_name in (
    "pose",
    "upright",
    "body_ang_vel",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "feet_flat",
    "soft_landing",
    "base_height",
  ):
    cfg.rewards.pop(reward_name, None)

  cfg.rewards["track_linear_velocity"].weight = 25.0
  cfg.rewards["track_angular_velocity"].weight = 10.0
  cfg.rewards["dof_pos_limits"].weight = -5.0
  cfg.rewards["action_rate_l2"].weight = -0.3
  cfg.rewards["stand_still"].weight = -0.05
  cfg.rewards["stand_feet_contact"].weight = 0.3
  cfg.rewards["termination_penalty"].weight = -20.0
  cfg.rewards["feet_distance"].weight = -5.0
  cfg.rewards["foot_slip"].weight = -0.1
  cfg.rewards["self_collisions"].weight = -3.0

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.2)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  if not play and "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-0.5, 0.5)},
      {
        "step": 5000 * 24,
        "lin_vel_x": (-1.0, 1.2),
        "lin_vel_y": (-0.5, 0.5),
        "ang_vel_z": (-0.7, 0.7),
      },
    ]

  return cfg


def phybot_c2_walk_preset_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a preset-gait AMP task with strict zero-command foot locking."""
  cfg = phybot_c2_walk_env_cfg(play=play)

  amp_asset_cfg = SceneEntityCfg(
    "robot",
    joint_names=PHYBOT_C2_POLICY_JOINT_NAMES,
    preserve_order=True,
  )
  cfg.observations["amp"] = ObservationGroupCfg(
    terms={
      "joint_pos": ObservationTermCfg(
        func=mdp.joint_pos,
        params={"asset_cfg": amp_asset_cfg},
      ),
      "joint_vel": ObservationTermCfg(
        func=mdp.joint_vel,
        params={"asset_cfg": amp_asset_cfg},
      ),
      "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
      "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=None,
  )

  cfg.commands["twist"] = PresetVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(4.0, 8.0),
    rel_standing_envs=0.0,
    rel_heading_envs=0.0,
    heading_command=False,
    debug_vis=True,
    ranges=PresetVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 0.6),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(-0.35, 0.35),
      heading=None,
    ),
    presets=(
      PresetVelocityCommandCfg.Preset("stand", 0.0, 0.0, 0.0, 45.0),
      PresetVelocityCommandCfg.Preset("slow_forward", 0.35, 0.0, 0.0, 25.0),
      PresetVelocityCommandCfg.Preset("walk_forward", 0.60, 0.0, 0.0, 20.0),
      PresetVelocityCommandCfg.Preset("turn_left", 0.20, 0.0, 0.35, 5.0),
      PresetVelocityCommandCfg.Preset("turn_right", 0.20, 0.0, -0.35, 5.0),
    ),
    slow_speed=0.35,
    walk_speed=0.60,
    turn_forward_speed=0.20,
    turn_speed=0.35,
  )

  # Preset task: no velocity range curriculum and no random pushes until the
  # stand-with-fixed-feet behavior is reliable.
  cfg.curriculum = {}
  cfg.events.pop("push_robot", None)

  cfg.rewards["alive"] = RewardTermCfg(func=mdp.is_alive, weight=1.0)
  cfg.rewards["track_linear_velocity"].weight = 0.0
  cfg.rewards["track_angular_velocity"].weight = 0.0
  cfg.rewards["upright"].weight = 2.0
  cfg.rewards["pose"].weight = 2.0
  cfg.rewards["base_height"].weight = 1.0
  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["dof_pos_limits"].weight = -5.0
  cfg.rewards["action_rate_l2"].weight = -0.25
  cfg.rewards["termination_penalty"].weight = -20.0
  cfg.rewards["air_time"].weight = 0.05
  cfg.rewards["air_time"].params["command_threshold"] = 0.1
  cfg.rewards["foot_clearance"].weight = -0.5
  cfg.rewards["foot_swing_height"].weight = -0.15
  cfg.rewards["feet_flat"].weight = 0.8
  cfg.rewards["feet_distance"].weight = -5.0
  cfg.rewards["stand_still"].weight = -0.05
  cfg.rewards["stand_feet_contact"].weight = 2.0
  cfg.rewards["foot_slip"].weight = -0.2
  cfg.rewards["self_collisions"].weight = -3.0

  cfg.rewards["preset_forward_velocity"] = RewardTermCfg(
    func=mdp.preset_forward_velocity_band,
    weight=4.0,
    params={"command_name": "twist", "command_threshold": 0.05},
  )
  cfg.rewards["preset_yaw_velocity"] = RewardTermCfg(
    func=mdp.preset_yaw_velocity_band,
    weight=2.0,
    params={"command_name": "twist", "command_threshold": 0.05},
  )
  cfg.rewards["stand_feet_stationary"] = RewardTermCfg(
    func=mdp.feet_stationary_when_standing,
    weight=-0.5,
    params={
      "command_name": "twist",
      "sensor_name": _FEET_GROUND_CONTACT_SENSOR,
      "command_threshold": 0.05,
      "missing_contact_cost": 0.5,
      "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITE_NAMES),
    },
  )
  cfg.rewards["stand_base_motion"] = RewardTermCfg(
    func=mdp.base_motion_when_standing,
    weight=-0.05,
    params={"command_name": "twist", "command_threshold": 0.05},
  )

  if not play:
    cfg.curriculum = {
      "stand_feet_stationary_weight": CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
          "reward_name": "stand_feet_stationary",
          "weight_stages": [
            {"step": 0, "weight": -0.5},
            {"step": 2000 * 24, "weight": -1.5},
            {"step": 6000 * 24, "weight": -3.0},
            {"step": 9000 * 24, "weight": -4.0},
          ],
        },
      ),
      "stand_still_weight": CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
          "reward_name": "stand_still",
          "weight_stages": [
            {"step": 0, "weight": -0.05},
            {"step": 4000 * 24, "weight": -0.15},
            {"step": 8000 * 24, "weight": -0.25},
          ],
        },
      ),
    }

  if play:
    # Keep play unlimited, clean, and deterministic.
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

  return cfg


def phybot_c2_walk_preset_amp_realpd_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Preset AMP task with actor obs unchanged and sim PD matched to mini deploy."""
  cfg = phybot_c2_walk_preset_amp_env_cfg(play=play)
  _apply_real_pd_actuator_damping(cfg)
  cfg.rewards["action_rate_l2"].weight = -0.25
  return cfg
