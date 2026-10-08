"""Walking rewards: each term once, with its weight in every variant that uses it."""

from __future__ import annotations

import math

from mjlab.envs import mdp as base_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_BODY_JOINT_NAMES,
  PHYBOT_C2_POLICY_JOINT_NAMES,
)
from scr_phybot_mjab.tasks.common.phybot_c2 import (
  FEET_GROUND_CONTACT_SENSOR,
  FEET_ROOT_BODIES,
  FOOT_SITE_NAMES,
  SELF_COLLISION_SENSOR,
  TORSO_BODY,
  policy_joint_cfg,
)
from scr_phybot_mjab.tasks.common.rewards import energy, feet_lateral_distance, self_collision_cost
from scr_phybot_mjab.tasks.walk import mdp

from .commands import TWIST_COMMAND
from .variants import WalkVariant


def walk_rewards(variant: WalkVariant) -> dict[str, RewardTermCfg]:
  stand = variant == "stand"

  def weight(**by_variant: float) -> float | None:
    """The term's weight in this variant; a variant left out does not use the term."""
    return by_variant.get(variant)

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=weight(walk=2.0, stand=2.0, amp=0.0),
      params={"command_name": TWIST_COMMAND, "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=weight(walk=2.0, stand=2.0, amp=0.0),
      params={"command_name": TWIST_COMMAND, "std": math.sqrt(0.5)},
    ),
    "upright": RewardTermCfg(
      func=mdp.flat_orientation,
      weight=weight(walk=1.0, stand=1.0, amp=2.0),
      params={"std": math.sqrt(0.2), "asset_cfg": SceneEntityCfg("robot", body_names=(TORSO_BODY,))},
    ),
    # Without AMP this is walk's only gait-style prior, with tight leg stds
    # (MJLab G1 values) that keep the legs near the default pose.
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=weight(walk=1.0, stand=2.0, amp=2.0),
      params=_pose_params(stand),
    ),
    # Anti-crouch: the measured zero-action settled base height in the
    # real-robot zero pose (ankle_pitch -0.295, keyframe z 0.726, PD sag
    # included). Absolute world z, valid on the flat floor only.
    "base_height": RewardTermCfg(
      func=mdp.base_height,
      weight=weight(walk=0.5, stand=0.5, amp=1.0),
      params={"target_height": 0.713, "std": 0.1},
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=weight(walk=-0.05, stand=-0.05, amp=-0.08),
      params={"asset_cfg": SceneEntityCfg("robot", body_names=(TORSO_BODY,))},
    ),
    "dof_pos_limits": RewardTermCfg(func=base_mdp.joint_pos_limits, weight=weight(walk=-1.0, stand=-1.0, amp=-5.0)),
    "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=weight(walk=-0.04, stand=-0.04, amp=-0.25)),
    # The weights below follow the official PhyBot Lab LiteRewardCfg.
    "energy": RewardTermCfg(func=energy, weight=-5e-4, params={"asset_cfg": SceneEntityCfg("robot")}),
    # The official AMP value -5e-4 makes heel balancing profitable without the
    # AMP style prior: zero ankle torque is worth about 30% of the tracking reward.
    "ankle_torque": RewardTermCfg(
      func=base_mdp.joint_torques_l2,
      weight=-5e-5,
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*_ankle_pitch", ".*_ankle_roll"))},
    ),
    "dof_acc_l2": RewardTermCfg(func=base_mdp.joint_acc_l2, weight=-1.5e-8),
    "termination_penalty": RewardTermCfg(func=base_mdp.is_terminated, weight=-20.0),
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=weight(walk=0.15, amp=0.05),
      params={
        "sensor_name": FEET_GROUND_CONTACT_SENSOR,
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": TWIST_COMMAND,
        "command_threshold": 0.1 if variant == "amp" else 0.5,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=weight(walk=-1.0, amp=-0.5),
      params={
        "target_height": 0.1,
        "command_name": TWIST_COMMAND,
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=weight(walk=-0.25, amp=-0.15),
      params={
        "sensor_name": FEET_GROUND_CONTACT_SENSOR,
        "target_height": 0.1,
        "command_name": TWIST_COMMAND,
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      },
    ),
    # Flat-foot contact discourages heel-point balancing.
    "feet_flat": RewardTermCfg(
      func=mdp.feet_flat_orientation,
      weight=weight(walk=0.5, stand=0.5, amp=0.8),
      params={
        "std": math.sqrt(0.1),
        "sensor_name": FEET_GROUND_CONTACT_SENSOR,
        "asset_cfg": SceneEntityCfg("robot", body_names=(FEET_ROOT_BODIES,)),
      },
    ),
    # Stance width: the feet stay at least 0.15 m apart laterally (the shoe
    # shells touch at about 0.10 m; the default stance is 0.297 m, so a normal
    # gait is unaffected).
    "feet_distance": RewardTermCfg(
      func=feet_lateral_distance,
      weight=-5.0,
      params={"min_dist": 0.15, "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES)},
    ),
    # A gentle calm-stance preference only. Static standing is impossible at
    # the deployed ankle gains (2x50 N*m/rad << m*g*h ~ 271), so the policy
    # must balance actively (weight shifts, stepping in place); a weight of
    # -0.3 dwarfed the -20 fall penalty and the policy froze and tipped over.
    # Only the policy joints count: the PD-held neck is not the policy's.
    "stand_still": RewardTermCfg(
      func=mdp.stand_still_joint_vel,
      weight=-0.05,
      params={
        "command_name": TWIST_COMMAND,
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", joint_names=PHYBOT_C2_POLICY_JOINT_NAMES),
        # stand recovers for three seconds before calm is asked of it.
        **({"start_time_s": 3.0} if stand else {}),
      },
    ),
    "stand_feet_contact": RewardTermCfg(
      func=mdp.feet_contact_when_standing,
      weight=weight(walk=0.3, stand=1.0, amp=2.0),
      params={
        "command_name": TWIST_COMMAND,
        "sensor_name": FEET_GROUND_CONTACT_SENSOR,
        "command_threshold": 0.05,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=weight(walk=-0.1, amp=-0.2),
      params={
        "sensor_name": FEET_GROUND_CONTACT_SENSOR,
        "command_name": TWIST_COMMAND,
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=weight(walk=-1e-5, amp=-1e-5),
      params={
        "sensor_name": FEET_GROUND_CONTACT_SENSOR,
        "command_name": TWIST_COMMAND,
        "command_threshold": 0.05,
      },
    ),
    # The only self-collision deterrent. There is no self-collision
    # termination (as in the official Isaac config): leg grazes while standing
    # or stepping would reset the env within 1-2 s and starve the policy of
    # standing experience.
    "self_collisions": RewardTermCfg(
      func=self_collision_cost,
      weight=weight(walk=-3.0, stand=-5.0, amp=-3.0),
      params={"sensor_name": SELF_COLLISION_SENSOR, "force_threshold": 10.0},
    ),
    "alive": RewardTermCfg(func=base_mdp.is_alive, weight=weight(amp=1.0)),
    # The preset bands reward the commanded gait's speed range; AMP supplies the style.
    "preset_forward_velocity": RewardTermCfg(
      func=mdp.preset_forward_velocity_band,
      weight=weight(amp=4.0),
      params={"command_name": TWIST_COMMAND, "command_threshold": 0.05},
    ),
    "preset_yaw_velocity": RewardTermCfg(
      func=mdp.preset_yaw_velocity_band,
      weight=weight(amp=2.0),
      params={"command_name": TWIST_COMMAND, "command_threshold": 0.05},
    ),
    # Under the zero command the planted feet must not move (amp).
    "stand_feet_stationary": RewardTermCfg(
      func=mdp.feet_stationary_when_standing,
      weight=weight(amp=-0.5),
      params={
        "command_name": TWIST_COMMAND,
        "sensor_name": FEET_GROUND_CONTACT_SENSOR,
        "command_threshold": 0.05,
        "missing_contact_cost": 0.5,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      },
    ),
    # stand recovers for three seconds, then freezes its current footholds.
    "stand_feet_lock": RewardTermCfg(
      func=mdp.feet_lock_after_recovery,
      weight=weight(stand=-0.25),
      params={
        "sensor_name": FEET_GROUND_CONTACT_SENSOR,
        "recovery_time_s": 3.0,
        "speed_scale": 0.10,
        "drift_scale": 0.05,
        "max_scaled_cost": 4.0,
        "missing_contact_cost": 4.0,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES, preserve_order=True),
      },
    ),
    "stand_base_motion": RewardTermCfg(
      func=mdp.base_motion_when_standing,
      weight=weight(stand=-0.05, amp=-0.05),
      params={
        "command_name": TWIST_COMMAND,
        "command_threshold": 0.05,
        **({"start_time_s": 3.0} if stand else {}),
      },
    ),
  }
  return {name: term for name, term in rewards.items() if term.weight is not None}


def _pose_params(stand: bool) -> dict:
  """Posture stds per joint group and gait speed.

  walk and amp weigh the body joints in model order, matching the std
  tables (".*" would also pull in the passive pad gimbals). stand weighs the
  policy joints, with a looser standing std and no neck entries.
  """
  std_walking = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.2,
    r".*neck.*": 0.1,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
  }
  std_running = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    r".*waist_yaw.*": 0.3,
    r".*neck.*": 0.2,
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
  }
  if stand:
    del std_walking[r".*neck.*"], std_running[r".*neck.*"]
  return {
    "asset_cfg": policy_joint_cfg() if stand else SceneEntityCfg("robot", joint_names=PHYBOT_C2_BODY_JOINT_NAMES),
    "command_name": TWIST_COMMAND,
    "std_standing": {".*": 0.12 if stand else 0.05},
    "std_walking": std_walking,
    "std_running": std_running,
    "walking_threshold": 0.05,
    "running_threshold": 1.5,
  }
