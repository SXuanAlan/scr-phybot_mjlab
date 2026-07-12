"""PHYBot C2 squat motion-tracking environment.

Follows the PHYBot C2 walk pipeline (same asset, sensors, real-PD damping and
ground collision) but swaps the velocity command for the phase-driven squat
reference and the velocity-tracking rewards for lower-body + base tracking.
"""

from __future__ import annotations

import copy
from pathlib import Path

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_ACTION_SCALE,
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_REAL_PD_DAMPING_BY_GROUP,
  PHYBOT_C2_STANDING_JOINT_POS,
  get_phybot_c2_robot_cfg,
)
from scr_phybot_mjab.tasks.squat import mdp
from scr_phybot_mjab.tasks.squat.mdp import SquatMotionCommandCfg
from scr_phybot_mjab.tasks.squat.squat_env_cfg import make_squat_env_cfg
from scr_phybot_mjab.tasks.walk.config.phybot_c2.env_cfgs import PHYBOT_C2_WALK_COLLISION

_TORSO_BODY = "torso"
_FEET_ROOT_BODIES = r"^(left_ankle_roll|right_ankle_roll)$"
_ILLEGAL_GROUND_BODIES = (
  r"^(left_knee|right_knee"
  r"|left_shoulder_roll|right_shoulder_roll"
  r"|left_shoulder_yaw|right_shoulder_yaw"
  r"|left_elbow_pitch|right_elbow_pitch"
  r"|torso|torso_upper)$"
)
_FOOT_COLLISION_GEOMS = r"^(left|right)_ankle_roll_collision$"
_BASE_QUAT_SENSOR = "robot/baselink-quat"
_BASE_LIN_VEL_SENSOR = "robot/baselink-velocity"
_BASE_ANG_VEL_SENSOR = "robot/baselink-gyro"

# Legs = the first 12 policy joints (2 x 6); arms = the last 8 (index 13:21, after
# waist_yaw at 12). Both are tracked; the waist is left free (reference ~ 0).
_LEG_JOINT_NAMES = PHYBOT_C2_POLICY_JOINT_NAMES[:12]
_ARM_JOINT_NAMES = PHYBOT_C2_POLICY_JOINT_NAMES[13:21]

_MOTIONS_DIR = Path(__file__).resolve().parents[4] / "assets" / "motions"
# The tracked timeline is a standing clip followed by the squat transition, so the
# standing dwell comes from the standing clip (not padded into the squat clip).
_SQUAT_MOTION_FILES = (
  str(_MOTIONS_DIR / "c2_stand_zero.npy"),
  str(_MOTIONS_DIR / "squat.npy"),
)

_SQUAT_STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.726),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos=PHYBOT_C2_STANDING_JOINT_POS,
  joint_vel={".*": 0.0},
)


def _apply_real_pd_actuator_damping(cfg: ManagerBasedRlEnvCfg) -> None:
  """Match hip/knee damping to the deployed mini PD (kd 5 -> 10)."""
  robot_cfg = copy.deepcopy(cfg.scene.entities["robot"])
  damping_by_target_expr = {
    (f".*_{group}",): damping
    for group, damping in PHYBOT_C2_REAL_PD_DAMPING_BY_GROUP.items()
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


def phybot_c2_squat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the PHYBot C2 flat-terrain squat tracking configuration."""
  cfg = make_squat_env_cfg()

  ##
  # Robot
  ##

  robot_cfg = get_phybot_c2_robot_cfg()
  robot_cfg.init_state = _SQUAT_STANDING_KEYFRAME
  robot_cfg.collisions = (PHYBOT_C2_WALK_COLLISION,)
  cfg.scene.entities = {"robot": robot_cfg}

  ##
  # Observations
  ##

  actor_terms = cfg.observations["actor"].terms
  actor_terms["base_ang_vel"].params = {"sensor_name": _BASE_ANG_VEL_SENSOR}
  actor_terms["projected_gravity"].func = mdp.projected_gravity_from_framequat_sensor
  actor_terms["projected_gravity"].params = {"sensor_name": _BASE_QUAT_SENSOR}
  for term_name in ("joint_pos", "joint_vel"):
    actor_terms[term_name].params = {
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=PHYBOT_C2_POLICY_JOINT_NAMES, preserve_order=True
      )
    }
  cfg.observations["critic"].terms["base_lin_vel"].params = {"sensor_name": _BASE_LIN_VEL_SENSOR}

  ##
  # Actions
  ##

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = PHYBOT_C2_POLICY_JOINT_NAMES
  joint_pos_action.preserve_order = True
  joint_pos_action.scale = PHYBOT_C2_ACTION_SCALE

  ##
  # Command and rewards
  ##

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, SquatMotionCommandCfg)
  motion_cmd.motion_files = _SQUAT_MOTION_FILES
  motion_cmd.joint_names = tuple(PHYBOT_C2_POLICY_JOINT_NAMES)

  cfg.rewards["track_legs"].params["joint_names"] = _LEG_JOINT_NAMES
  cfg.rewards["track_arms"].params["joint_names"] = _ARM_JOINT_NAMES

  ##
  # Contact sensors and terminations
  ##

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
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
      primary=ContactMatch(mode="body", pattern=_ILLEGAL_GROUND_BODIES, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      history_length=4,
    ),
  )
  cfg.terminations["illegal_ground_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "illegal_ground_contact", "force_threshold": 1.0},
  )

  ##
  # Domain randomization
  ##

  # Pushes aligned to the walk task (more frequent and stronger than a gentle
  # squat default). These fire mid-episode, so they do not touch the RSI reward
  # scale; a push near the squat bottom is a real robustness test.
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
  # Friction DR split like walk: non-foot body geoms 0.6-1.0, feet 0.4-1.0.
  body_friction = cfg.events["body_friction"]
  body_friction.params["asset_cfg"].geom_names = (
    r"^(?!(left|right)_ankle_roll_collision$).*_collision$",
  )
  body_friction.params["ranges"] = (0.6, 1.0)
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
  # Encoder bias: the policy observes joint angles offset by a fixed per-episode
  # bias (sim2real sensor error). The tracking reward reads the TRUE joint_pos, so
  # this does not affect the reward scale -- only what the policy sees.
  cfg.events["encoder_bias"] = EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
  )

  ##
  # Viewer
  ##

  cfg.viewer.body_name = _TORSO_BODY

  ##
  # Sim to real: match hip/knee PD to the deployed mini gains.
  ##

  _apply_real_pd_actuator_damping(cfg)

  ##
  # Play mode
  ##

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.sampling_mode = "start"
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)

  return cfg
