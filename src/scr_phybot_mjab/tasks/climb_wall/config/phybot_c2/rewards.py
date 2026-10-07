"""Climbing rewards: BeyondMimic body tracking adapted to climbing, plus the microspine contract.

Pose kernels are gated by schedule adherence, a reference-height ghost
replaces global position tracking, regrips get landing and approach shaping,
and the microspine terms price support loss and loaded releases.
"""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from scr_phybot_mjab.tasks.climb_wall import mdp
from scr_phybot_mjab.tasks.climb_wall.climb_wall_env_cfg import (
  MICROSPINE_AFFORDANCE_COMMAND,
  MICROSPINE_GRIP_EVENT,
  MOTION_COMMAND,
  UNDESIRED_CONTACT_SENSOR,
)
from scr_phybot_mjab.tasks.common.phybot_c2 import policy_joint_cfg


def climb_rewards() -> dict[str, RewardTermCfg]:
  return {**_tracking_rewards(), **_microspine_rewards(), **_regularization_rewards()}


def _tracking_rewards() -> dict[str, RewardTermCfg]:
  # Pose income is gated by schedule adherence: the 0.25 floor keeps early
  # learning fed, and the grace is 0.3 of the violated window, so a
  # weld-through forfeits ~70% of the gate credit regardless of clip timing.
  pose_params = {
    "command_name": MOTION_COMMAND,
    "engage_gate_floor": 0.25,
    "engage_gate_grace_fraction": 0.3,
  }
  return {
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_body_position_error_exp, weight=1.0, params={**pose_params, "std": 0.1}
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_body_orientation_error_exp, weight=1.0, params={**pose_params, "std": 0.3}
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_body_linear_velocity_error_exp, weight=1.0, params={"command_name": MOTION_COMMAND, "std": 1.0}
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": MOTION_COMMAND, "std": 3.14},
    ),
    "motion_root_ori": RewardTermCfg(
      func=mdp.motion_root_ori_error_exp,
      weight=0.5,
      params={
        "command_name": MOTION_COMMAND,
        "std": 0.3,
        "engage_gate_floor": 0.25,
        "engage_gate_grace_fraction": 0.3,
      },
    ),
    # Climb income follows the reference: an exp kernel on (robot climb -
    # reference climb) since settling, decaying when the robot falls behind
    # or runs ahead. std 0.025 m is 1.25 step heights of the 20 mm clip.
    "motion_ref_height": RewardTermCfg(
      func=mdp.motion_ref_height_tracking(),
      weight=2.5,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "motion_command_name": MOTION_COMMAND,
        "std": 0.025,
        "settle_time_s": 0.5,
      },
    ),
    # Pays each scheduled latch once by where it lands on the anchor ladder;
    # relatching inside the same window pays nothing.
    "motion_anchor_landing": RewardTermCfg(
      func=mdp.motion_anchor_landing(),
      weight=40.0,
      params={"command_name": MOTION_COMMAND, "std": 0.03},
    ),
    # Dense progress toward ``ref_anchor_pos_w`` while the EE is scheduled off
    # and released; the landing bonus alone is too sparse for exploration to
    # find. Ceiling 0.5/s per swinging EE, under the mismatch bill.
    "motion_swing_approach": RewardTermCfg(
      func=mdp.motion_swing_approach,
      weight=0.5,
      params={"command_name": MOTION_COMMAND, "std": 0.05},
    ),
    # Bills each pad whose latch state disagrees with the schedule; the bill
    # ramps over the violated window's outer 30%, the same countdown the
    # policy observes.
    "motion_engage_mismatch": RewardTermCfg(
      func=mdp.motion_engage_mismatch,
      weight=-1.0,
      params={"command_name": MOTION_COMMAND, "grace_fraction": 0.3},
    ),
  }


def _microspine_rewards() -> dict[str, RewardTermCfg]:
  """Two, three and four anchors are all acceptable; only too little support is
  penalized. Hold failures carry one-shot costs."""
  return {
    "microspine_support_loss": RewardTermCfg(
      func=mdp.microspine_support_loss,
      weight=-4.0,
      params={
        "event_term_name": MICROSPINE_GRIP_EVENT,
        "minimum_supports": 2,
      },
    ),
    # One-shot costs of hold failures; microspine_release_load prices the
    # unit loads of every release.
    "unsafe_microspine_release": RewardTermCfg(
      func=mdp.unsafe_microspine_release_penalty,
      weight=-2.5,
      params={
        "command_name": MICROSPINE_AFFORDANCE_COMMAND,
        "minimum_supports": 2,
        "settle_time_s": 0.5,
      },
    ),
    # Uses the active-unit loads the hold judge captured. Near-zero shear and
    # lateral load earns one scheduled event bonus; shear-loaded releases cost
    # more than equally loaded lateral ones, in either sign.
    "microspine_release_load": RewardTermCfg(
      func=mdp.microspine_release_load_reward(),
      weight=1.0,
      params={
        "command_name": MICROSPINE_AFFORDANCE_COMMAND,
        "motion_command_name": MOTION_COMMAND,
        "shear_scale_n": 2.0,
        "lateral_scale_n": 2.0,
        "shear_cost": 0.25,
        "lateral_cost": 0.1,
        "max_load_cost": 4.0,
        "minimum_supports": 3,
        "release_lead_time_s": 0.25,
        "settle_time_s": 0.5,
      },
    ),
  }


def _regularization_rewards() -> dict[str, RewardTermCfg]:
  return {
    # Prices a hand buzzing through the 60 ms engage debounce, with headroom
    # left for fast regrip reaches (walk uses -0.25).
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    # The target clamp is the safety guard; this gives the policy a signal to
    # stay away from it and from the physical joint stops.
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": policy_joint_cfg()},
    ),
    # The reward manager scales by dt, so -500 is a -10 one-shot failure cost.
    "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-500.0),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_body_contacts,
      weight=-0.1,
      params={"sensor_name": UNDESIRED_CONTACT_SENSOR, "threshold": 1.0},
    ),
  }
