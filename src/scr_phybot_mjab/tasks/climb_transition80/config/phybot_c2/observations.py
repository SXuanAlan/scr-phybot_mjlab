"""Transition observations. Term order is the checkpoint and export layout contract.

The actor keeps climb's proprioception, latch lanes and schedule chart, and
adds BeyondMimic reference lanes, the pad faces and root pose in the world
frame, and the wall's face corners. The critic adds the tracked body poses
and privileged microspine state.
"""

from __future__ import annotations

import dataclasses

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg

from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.tasks.climb_transition80 import mdp
from scr_phybot_mjab.tasks.climb_wall import mdp as climb_mdp
from scr_phybot_mjab.tasks.climb_wall.climb_wall_env_cfg import MICROSPINE_AFFORDANCE_COMMAND, MOTION_COMMAND
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.observations import (
  add_noise,
  latch_terms,
  motion_command_term,
  proprio_terms,
)

REMOVED_OBS = (
  "height_above_spawn",
  "velocity_command",
  "microspine_disengage_time",
  "microspine_cooldown",
  "wall_base_position",
)
"""Climb lanes left out: on a mount at a pinned 80-degree wall each is
constant or another lane's copy (root_pose_world gives the height, the face
corners the wall). The motion_command chart also drops its six constant
lanes, 16 -> 10."""

ACTOR_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "projected_gravity",
  "joint_pos_rel",
  "joint_vel_rel",
  "last_action",
  "prev_action",
  "prev_prev_action",
  "microspine_engaged_state",
  "microspine_contact_fraction",
  "motion_command",
  "motion_ref_joint_pos_rel",
  "motion_ref_joint_vel",
  "motion_anchor_pos_b",
  "motion_anchor_ori_b",
  "motion_pad_center_relative",
  "motion_anchor_relative",
  "microspine_pad_center_world",
  "root_pose_world",
  "wall_face_corners",
)
CRITIC_TAIL = (
  "robot_body_pos_b",
  "robot_body_ori_b",
  "microspine_hold_margin",
  "microspine_anchor_load",
  "microspine_forced_release",
  "microspine_pad_normal_force",
  "microspine_unit_preload_fraction",
)
"""The critic's lanes after its copies of the actor lanes."""

OBS_DIMS = {
  "base_lin_vel": 3,
  "base_ang_vel": 3,
  "projected_gravity": 3,
  "joint_pos_rel": 21,
  "joint_vel_rel": 21,
  "last_action": 21,
  "prev_action": 21,
  "prev_prev_action": 21,
  "microspine_engaged_state": 4,
  "microspine_contact_fraction": 4,
  "motion_command": 10,
  "motion_ref_joint_pos_rel": 21,
  "motion_ref_joint_vel": 21,
  "motion_anchor_pos_b": 3,
  "motion_anchor_ori_b": 6,
  "motion_pad_center_relative": 12,
  "motion_anchor_relative": 12,
  "microspine_pad_center_world": 12,
  "root_pose_world": 7,
  "wall_face_corners": 12,
  "robot_body_pos_b": 51,
  "robot_body_ori_b": 102,
  "microspine_hold_margin": 4,
  "microspine_anchor_load": 12,
  "microspine_forced_release": 4,
  "microspine_pad_normal_force": 4,
  "microspine_unit_preload_fraction": 4,
}
ACTOR_DIM = sum(OBS_DIMS[name] for name in ACTOR_TERMS)  # 238
CRITIC_DIM = ACTOR_DIM + sum(OBS_DIMS[name] for name in CRITIC_TAIL)  # 419


def transition80_observations(
  microspine_cfg: MicrospineCfg, actor_noise: dict[str, UniformNoiseCfg] | None
) -> dict[str, ObservationGroupCfg]:
  """The actor group, corrupted with ``actor_noise`` when given, and its privileged critic."""

  def reference(func) -> ObservationTermCfg:
    return ObservationTermCfg(func=func, params={"command_name": MOTION_COMMAND})

  actor_terms = {
    **proprio_terms(),
    **latch_terms(microspine_cfg),
    "motion_command": motion_command_term(),
    "motion_ref_joint_pos_rel": reference(mdp.motion_ref_joint_pos_rel),
    "motion_ref_joint_vel": reference(mdp.motion_ref_joint_vel),
    "motion_anchor_pos_b": reference(mdp.motion_anchor_pos_b),
    "motion_anchor_ori_b": reference(mdp.motion_anchor_ori_b),
    "motion_pad_center_relative": reference(mdp.motion_pad_center_relative),
    "motion_anchor_relative": reference(mdp.motion_anchor_relative),
    "microspine_pad_center_world": ObservationTermCfg(
      func=mdp.microspine_pad_center_world, params={"command_name": MICROSPINE_AFFORDANCE_COMMAND}
    ),
    "root_pose_world": ObservationTermCfg(func=mdp.root_pose_world),
    "wall_face_corners": ObservationTermCfg(func=mdp.wall_face_corners),
  }
  add_noise(actor_terms, actor_noise)
  # Noise-free copies that share the actor lanes' params.
  critic_terms = {name: dataclasses.replace(term, noise=None) for name, term in actor_terms.items()}
  critic_terms |= {
    "robot_body_pos_b": reference(mdp.robot_body_pos_b),
    "robot_body_ori_b": reference(mdp.robot_body_ori_b),
    **climb_mdp.privileged_microspine_terms(MICROSPINE_AFFORDANCE_COMMAND),
    "microspine_pad_normal_force": ObservationTermCfg(
      func=climb_mdp.microspine_pad_normal_force, params={"command_name": MICROSPINE_AFFORDANCE_COMMAND}, scale=0.01
    ),
    "microspine_unit_preload_fraction": ObservationTermCfg(
      func=mdp.microspine_unit_preload_fraction, params={"command_name": MICROSPINE_AFFORDANCE_COMMAND}
    ),
  }
  return {
    "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True, enable_corruption=actor_noise is not None),
    "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False),
  }
