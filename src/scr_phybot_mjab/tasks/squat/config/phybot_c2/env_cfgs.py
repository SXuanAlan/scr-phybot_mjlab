"""PHYBot C2 squat motion-tracking environment.

Follows the PHYBot C2 walk pipeline (same asset, sensors, real-PD damping and
ground collision) but swaps the velocity command for the phase-driven squat
reference and the velocity-tracking rewards for lower-body + base tracking.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_ACTION_SCALE,
  PHYBOT_C2_FOOT_SITES,
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

# All 21 policy joints are motion-tracked. The neck is not policy-controlled and
# is intentionally absent from both this list and squat.npy.
_LEG_JOINT_NAMES = PHYBOT_C2_POLICY_JOINT_NAMES[:12]
_WAIST_JOINT_NAMES = (PHYBOT_C2_POLICY_JOINT_NAMES[12],)
_ARM_JOINT_NAMES = PHYBOT_C2_POLICY_JOINT_NAMES[13:21]
_FOOT_SITE_NAMES = tuple(site for _, site in PHYBOT_C2_FOOT_SITES)
assert _LEG_JOINT_NAMES + _WAIST_JOINT_NAMES + _ARM_JOINT_NAMES == tuple(
  PHYBOT_C2_POLICY_JOINT_NAMES
)

_MOTIONS_DIR = Path(__file__).resolve().parents[4] / "assets" / "motions"
_SQUAT_MOTION_FILE = _MOTIONS_DIR / "squat.npy"
_SQUAT_MOTION_FILES = (str(_SQUAT_MOTION_FILE),)


def _motion_duration_s(path: Path, control_dt: float = 0.02) -> float:
  """Read an exact frame-count duration and require one frame per control step."""
  data = np.load(path, allow_pickle=True).item()
  frames = int(np.asarray(data["joint_positions"]).shape[0])
  fps = float(np.asarray(data["fps"]).reshape(-1)[0])
  duration_s = frames / fps
  control_steps = duration_s / control_dt
  if not np.isclose(control_steps, round(control_steps), atol=1e-9):
    raise ValueError(
      f"Squat motion duration {duration_s}s is not an integer number of "
      f"{control_dt}s control steps."
    )
  if not np.isclose(fps, 1.0 / control_dt, atol=1e-6):
    raise ValueError(f"Squat motion fps {fps} must equal control fps {1.0 / control_dt}.")
  return duration_s


def _motion_action_limits(
  path: Path,
  joint_names: tuple[str, ...],
  action_scale: float,
  margin: float = 0.75,
  minimum: float = 1.0,
) -> tuple[float, ...]:
  """Build raw-action envelopes that contain the complete arm reference."""
  data = np.load(path, allow_pickle=True).item()
  clip_joint_names = list(data["joints_list"])
  joint_ids = [clip_joint_names.index(name) for name in joint_names]
  joint_pos = np.asarray(data["joint_positions"], dtype=np.float32)[:, joint_ids]
  default = np.asarray(
    [PHYBOT_C2_STANDING_JOINT_POS[name] for name in joint_names],
    dtype=np.float32,
  )
  reference_action = (joint_pos - default) / action_scale
  limits = np.maximum(np.max(np.abs(reference_action), axis=0) + margin, minimum)
  return tuple(float(limit) for limit in limits)


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
  motion_cmd.foot_site_names = _FOOT_SITE_NAMES
  cfg.episode_length_s = _motion_duration_s(_SQUAT_MOTION_FILE)

  cfg.rewards["track_legs"].params["joint_names"] = _LEG_JOINT_NAMES
  for reward_name in (
    "track_waist",
    "waist_tracking_error",
    "track_waist_velocity",
    "waist_action_limits",
    "waist_reference_action",
  ):
    cfg.rewards[reward_name].params["joint_names"] = _WAIST_JOINT_NAMES
  # waist_yaw is identically zero in squat.npy.  The inner tube leaves 0.075
  # rad of target correction for balance (0.30 raw * 0.25-rad action scale),
  # while the outer envelope prevents another global-clip exploit.
  cfg.rewards["waist_reference_action"].params["max_abs_deviation"] = (0.30,)
  cfg.rewards["waist_action_limits"].params["max_abs_actions"] = (0.75,)
  for reward_name in (
    "track_arms",
    "arm_tracking_error",
    "track_arm_velocity",
    "arm_action_limits",
    "arm_reference_action",
  ):
    cfg.rewards[reward_name].params["joint_names"] = _ARM_JOINT_NAMES
  left_arm_names = _ARM_JOINT_NAMES[:4]
  right_arm_names = _ARM_JOINT_NAMES[4:]
  for reward_name in ("arm_position_symmetry", "arm_action_symmetry"):
    cfg.rewards[reward_name].params["left_joint_names"] = left_arm_names
    cfg.rewards[reward_name].params["right_joint_names"] = right_arm_names
  cfg.rewards["arm_reference_action"].params["max_abs_deviation"] = (
    (0.75,) * len(_ARM_JOINT_NAMES)
  )
  cfg.rewards["arm_action_limits"].params["max_abs_actions"] = (
    _motion_action_limits(
      _SQUAT_MOTION_FILE,
      _ARM_JOINT_NAMES,
      PHYBOT_C2_ACTION_SCALE,
    )
  )
  for reward_name in (
    "track_feet_stationary",
    "foot_slip_error",
    "track_feet_anchor",
    "foot_anchor_error",
  ):
    cfg.rewards[reward_name].params["asset_cfg"] = SceneEntityCfg(
      "robot", site_names=_FOOT_SITE_NAMES, preserve_order=True
    )
  cfg.rewards["feet_distance"] = RewardTermCfg(
    func=mdp.feet_lateral_distance,
    weight=-5.0,
    params={
      "min_dist": 0.15,
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=_FOOT_SITE_NAMES, preserve_order=True
      ),
    },
  )

  ##
  # Contact sensors and terminations
  ##

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(mode="subtree", pattern=_FEET_ROOT_BODIES, entity="robot"),
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
      reduce="maxforce",
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

  # Do not train recovery steps in a planted-foot squat task. Strong periodic
  # velocity pushes taught the old policy to stomp while preserving joint angles.
  cfg.events.pop("push_robot", None)
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
    cfg.observations["actor"].enable_corruption = False
    motion_cmd.sampling_mode = "start"
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    body_friction.params["ranges"] = (1.0, 1.0)
    cfg.events["foot_friction"].params["ranges"] = (0.7, 0.7)
    base_mass.params["ranges"] = (0.0, 0.0)
    cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)

  return cfg
