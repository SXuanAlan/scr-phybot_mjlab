"""Squat rewards: joint tracking by group, base tracking, planted feet and regularization."""

from __future__ import annotations

from mjlab.envs import mdp as base_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from scr_phybot_mjab.assets.motions import SQUAT_MOTION
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_ACTION_SCALE
from scr_phybot_mjab.tasks.common.phybot_c2 import FEET_GROUND_CONTACT_SENSOR, FOOT_SITE_NAMES, SELF_COLLISION_SENSOR
from scr_phybot_mjab.tasks.common.rewards import energy, feet_lateral_distance, self_collision_cost
from scr_phybot_mjab.tasks.squat import mdp

from .commands import MOTION_COMMAND
from .motion import ARM_JOINT_NAMES, LEG_JOINT_NAMES, WAIST_JOINT_NAMES, motion_action_limits

_LEFT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[:4]
_RIGHT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[4:]


def _feet() -> SceneEntityCfg:
  return SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES, preserve_order=True)


def squat_rewards() -> dict[str, RewardTermCfg]:
  return {
    "track_legs": RewardTermCfg(
      func=mdp.track_joint_pos,
      weight=3.0,
      params={"command_name": MOTION_COMMAND, "std": 0.3, "joint_names": LEG_JOINT_NAMES},
    ),
    # The arm joints get independent dense scores: one exponential over all
    # eight saturates near zero and lets the policy sacrifice its arms for the
    # planted-foot and base rewards.
    "track_arms": RewardTermCfg(
      func=mdp.track_joint_pos_cauchy,
      weight=3.0,
      params={
        "command_name": MOTION_COMMAND,
        "std": 0.20,
        "joint_names": ARM_JOINT_NAMES,
        "metric_prefix": "arm",
        "worst_joint_weight": 0.5,
      },
    ),
    "arm_tracking_error": RewardTermCfg(
      func=mdp.joint_pos_tracking_huber_error,
      weight=-2.0,
      params={
        "command_name": MOTION_COMMAND,
        "beta": 0.25,
        "joint_names": ARM_JOINT_NAMES,
        "reduction": "sum",
      },
    ),
    "track_arm_velocity": RewardTermCfg(
      func=mdp.track_joint_vel_cauchy,
      weight=1.0,
      params={
        "command_name": MOTION_COMMAND,
        "std": 1.0,
        "joint_names": ARM_JOINT_NAMES,
        "metric_prefix": "arm",
      },
    ),
    # Raw-action envelopes that contain the whole arm reference.
    "arm_action_limits": RewardTermCfg(
      func=mdp.joint_action_limit_cost,
      weight=-0.5,
      params={
        "action_name": "joint_pos",
        "joint_names": ARM_JOINT_NAMES,
        "max_abs_actions": motion_action_limits(SQUAT_MOTION, ARM_JOINT_NAMES, PHYBOT_C2_ACTION_SCALE),
        "metric_prefix": "arm",
        "reduction": "sum",
      },
    ),
    "arm_reference_action": RewardTermCfg(
      func=mdp.joint_action_reference_tube_cost,
      weight=-0.5,
      params={
        "command_name": MOTION_COMMAND,
        "action_name": "joint_pos",
        "joint_names": ARM_JOINT_NAMES,
        "max_abs_deviation": (0.75,) * len(ARM_JOINT_NAMES),
        "beta": 0.5,
        "metric_prefix": "arm",
      },
    ),
    "arm_position_symmetry": RewardTermCfg(
      func=mdp.joint_pair_symmetry_huber_error,
      weight=-2.0,
      params={
        "command_name": MOTION_COMMAND,
        "left_joint_names": _LEFT_ARM_JOINT_NAMES,
        "right_joint_names": _RIGHT_ARM_JOINT_NAMES,
        "beta": 0.10,
        "metric_prefix": "arm",
      },
    ),
    "arm_action_symmetry": RewardTermCfg(
      func=mdp.joint_action_pair_symmetry_huber_error,
      weight=-0.5,
      params={
        "command_name": MOTION_COMMAND,
        "action_name": "joint_pos",
        "left_joint_names": _LEFT_ARM_JOINT_NAMES,
        "right_joint_names": _RIGHT_ARM_JOINT_NAMES,
        "beta": 0.25,
        "metric_prefix": "arm",
      },
    ),
    # The clip holds waist_yaw at zero. A single exponential saturates after a
    # modest error, which lets PPO sacrifice the waist and park its action at
    # the global clip as a cheap balance actuator; a dense position/velocity
    # score and non-saturating position/action costs keep a restoring gradient
    # at any yaw error.
    "track_waist": RewardTermCfg(
      func=mdp.track_joint_pos_cauchy,
      weight=2.0,
      params={
        "command_name": MOTION_COMMAND,
        "std": 0.08,
        "joint_names": WAIST_JOINT_NAMES,
        "metric_prefix": "waist",
      },
    ),
    "waist_tracking_error": RewardTermCfg(
      func=mdp.joint_pos_tracking_huber_error,
      weight=-3.0,
      params={
        "command_name": MOTION_COMMAND,
        "beta": 0.10,
        "joint_names": WAIST_JOINT_NAMES,
        "reduction": "sum",
      },
    ),
    "track_waist_velocity": RewardTermCfg(
      func=mdp.track_joint_vel_cauchy,
      weight=1.0,
      params={
        "command_name": MOTION_COMMAND,
        "std": 0.5,
        "joint_names": WAIST_JOINT_NAMES,
        "metric_prefix": "waist",
      },
    ),
    # The inner tube (waist_reference_action) leaves 0.075 rad of target
    # correction for balance (0.30 raw x the 0.25-rad action scale); this outer
    # envelope prevents another global-clip exploit.
    "waist_action_limits": RewardTermCfg(
      func=mdp.joint_action_limit_cost,
      weight=-0.5,
      params={
        "action_name": "joint_pos",
        "joint_names": WAIST_JOINT_NAMES,
        "max_abs_actions": (0.75,),
        "metric_prefix": "waist",
        "reduction": "sum",
      },
    ),
    "waist_reference_action": RewardTermCfg(
      func=mdp.joint_action_reference_tube_cost,
      weight=-1.0,
      params={
        "command_name": MOTION_COMMAND,
        "action_name": "joint_pos",
        "joint_names": WAIST_JOINT_NAMES,
        "max_abs_deviation": (0.30,),
        "beta": 0.25,
        "metric_prefix": "waist",
      },
    ),
    "track_joint_velocity": RewardTermCfg(
      func=mdp.track_joint_vel,
      weight=2.0,
      params={"command_name": MOTION_COMMAND, "std": 2.0, "joint_names": None},
    ),
    "track_base_position": RewardTermCfg(
      func=mdp.track_base_position,
      weight=2.0,
      params={"command_name": MOTION_COMMAND, "std": 0.05},
    ),
    "track_base_orientation": RewardTermCfg(
      func=mdp.track_base_orientation,
      weight=1.0,
      params={"command_name": MOTION_COMMAND, "std": 0.3},
    ),
    "track_base_lin_vel": RewardTermCfg(
      func=mdp.track_base_lin_vel,
      weight=1.0,
      params={"command_name": MOTION_COMMAND, "std": 0.3},
    ),
    "track_feet_stationary": RewardTermCfg(
      func=mdp.track_feet_stationary,
      weight=3.0,
      params={"sensor_name": FEET_GROUND_CONTACT_SENSOR, "std": 0.04, "asset_cfg": _feet()},
    ),
    "foot_slip_error": RewardTermCfg(
      func=mdp.feet_xy_velocity_huber_error,
      weight=-4.0,
      params={"beta": 0.05, "asset_cfg": _feet()},
    ),
    "track_feet_anchor": RewardTermCfg(
      func=mdp.track_feet_anchor,
      weight=5.0,
      params={"command_name": MOTION_COMMAND, "std": 0.02, "asset_cfg": _feet()},
    ),
    "foot_anchor_error": RewardTermCfg(
      func=mdp.feet_xy_anchor_huber_error,
      weight=-5.0,
      params={"command_name": MOTION_COMMAND, "beta": 0.02, "asset_cfg": _feet()},
    ),
    "alive": RewardTermCfg(func=base_mdp.is_alive, weight=0.5),
    "termination_penalty": RewardTermCfg(func=base_mdp.is_terminated, weight=-20.0),
    # Regularization supplements the explicit full-body tracking terms.
    "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.2),
    "dof_pos_limits": RewardTermCfg(func=base_mdp.joint_pos_limits, weight=-5.0),
    "dof_acc_l2": RewardTermCfg(func=base_mdp.joint_acc_l2, weight=-1.0e-7),
    "energy": RewardTermCfg(func=energy, weight=-5e-4, params={"asset_cfg": SceneEntityCfg("robot")}),
    "self_collisions": RewardTermCfg(
      func=self_collision_cost,
      weight=-3.0,
      params={"sensor_name": SELF_COLLISION_SENSOR, "force_threshold": 10.0},
    ),
    # The clip's feet stand about 0.296 m apart: 0.26 leaves roughly 3.6 cm of
    # lateral compliance and penalizes inward drift well before the shoes meet.
    "feet_distance": RewardTermCfg(
      func=feet_lateral_distance,
      weight=-5.0,
      params={"min_dist": 0.26, "asset_cfg": _feet()},
    ),
  }
