"""The brick reward table."""

from __future__ import annotations

from mjlab.envs import mdp as base_mdp
from mjlab.managers.reward_manager import RewardTermCfg

from scr_phybot_mjab.tasks.brick import mdp


def brick_rewards() -> dict[str, RewardTermCfg]:
  return {
    "reference": RewardTermCfg(func=mdp.tracking, weight=1.0),
    "object": RewardTermCfg(func=mdp.object_tracking, weight=2.0),
    "milestone": RewardTermCfg(func=mdp.milestone, weight=2.0),
    "success": RewardTermCfg(func=mdp.completion, weight=20.0),
    "grip": RewardTermCfg(func=mdp.grip_quality, weight=0.5),
    "squeeze": RewardTermCfg(func=mdp.squeeze, weight=-0.2),
    "overload": RewardTermCfg(func=mdp.overload, weight=-2.0),
    "slip": RewardTermCfg(func=mdp.slip, weight=-1.0),
    "foot_drift": RewardTermCfg(func=mdp.foot_drift, weight=-100.0),
    "action_rate": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.005),
    # A failure costs 20 in total: the per-step weight is divided by the 0.02 s step.
    "failure": RewardTermCfg(func=mdp.failed, weight=-20.0 / 0.02),
  }
