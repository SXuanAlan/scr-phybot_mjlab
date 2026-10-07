"""The transition reward table: BeyondMimic tracking, one engage-progress potential, a dwell wage and the rip bill."""

from __future__ import annotations

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg

from scr_phybot_mjab.tasks.climb_transition80 import mdp
from scr_phybot_mjab.tasks.climb_wall.climb_wall_env_cfg import MICROSPINE_AFFORDANCE_COMMAND, MOTION_COMMAND
from scr_phybot_mjab.tasks.common.phybot_c2 import policy_joint_cfg

from .schedules import ANCHOR_RADIUS_STAGES, RIP_PENALTY_STAGES, SEED_SETTLE_EXEMPT_S, stage0


def transition80_rewards() -> dict[str, RewardTermCfg]:
  # Climb's schedule-adherence discount on pose income: while a pad's latch
  # state disagrees with the chart, the anchor kernels pay ``floor`` of their
  # value, ramped over a fixed 0.5 s near scheduled edges. The body kernels
  # take the same parameters but do not apply them.
  gate = {"engage_gate_floor": 0.10, "engage_gate_grace_s": 0.5, "engage_gate_grace_fraction": None}
  return {
    "motion_anchor_pos": RewardTermCfg(
      func=mdp.motion_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": MOTION_COMMAND, "std": 0.3, **gate},
    ),
    "motion_anchor_ori": RewardTermCfg(
      func=mdp.motion_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": MOTION_COMMAND, "std": 0.4, **gate},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.transition_body_position_error_exp,
      weight=1.0,
      params={"command_name": MOTION_COMMAND, "std": 0.3, **gate},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.transition_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": MOTION_COMMAND, "std": 0.4, **gate},
    ),
    # The pad face centers, never discounted: reaching the reference face is
    # the path to schedule compliance. The narrow std pays the last
    # centimeters to contact; at 0.2 a pad hovering 3 cm off the wall kept
    # 98 % of this income.
    "motion_pad_pos": RewardTermCfg(
      func=mdp.motion_pad_center_error_exp,
      weight=1.0,
      params={"command_name": MOTION_COMMAND, "std": 0.08},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.transition_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": MOTION_COMMAND, "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.transition_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": MOTION_COMMAND, "std": 3.14},
    ),
    # The potential Phi prices a touch above a hover and a weld above a
    # touch. Reach + approach stays below the touch level and touch + load +
    # press equals the weld level 0.9, so Phi rises monotonically along
    # reach, approach, touch, press and weld.
    "engage_progress": RewardTermCfg(
      func=mdp.engage_progress,
      weight=3.0,
      params={
        "command_name": MOTION_COMMAND,
        "affordance_command_name": MICROSPINE_AFFORDANCE_COMMAND,
        "edge_lead_time_s": 2.0,
        # Wide enough to slope from where a parked hand waits.
        "reach_radius_m": 0.5,
        "touch_radius_m": 0.15,
        "reach_share": 0.25,
        "touch_level": 0.45,
        "load_share": 0.10,
        # The last 10 cm of wall-normal approach and the press itself.
        "approach_share": 0.19,
        "approach_radius_m": 0.10,
        "press_share": 0.35,
        "press_ref_n": 20.0,
        "anchor_radius_m": stage0(ANCHOR_RADIUS_STAGES),
      },
    ),
    # Paid on scheduled, welded pads as clip(hold_margin / full_margin, 0, 1):
    # a load-sharing preference, never rent and never a reason not to weld.
    "hold_margin_wage": RewardTermCfg(
      func=mdp.scheduled_hold_margin_wage,
      weight=0.5,
      params={
        "command_name": MOTION_COMMAND,
        "affordance_command_name": MICROSPINE_AFFORDANCE_COMMAND,
        "full_margin": 0.3,
      },
    ),
    # A torn scheduled anchor is billed per pad until re-welded instead of
    # ending the episode; the contact law still ends one never re-welded.
    "anchor_rip_penalty": RewardTermCfg(
      func=mdp.scheduled_anchor_rip_penalty(),
      weight=-1.0,
      params={
        "command_name": MOTION_COMMAND,
        "affordance_command_name": MICROSPINE_AFFORDANCE_COMMAND,
        "weight_per_pad": stage0(RIP_PENALTY_STAGES),
        "release_lead_time_s": 0.5,
        "seed_settle_exempt_s": SEED_SETTLE_EXEMPT_S,
      },
    ),
    # Half of climb's -0.1: the full weight taxes exploration variance and
    # collapses the action std before any engage skill forms.
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": policy_joint_cfg()},
    ),
  }
