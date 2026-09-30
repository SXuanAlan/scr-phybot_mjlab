from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

# Avoid MJLab entry-point discovery re-importing a partially initialized PHYBot
# constants module during this direct contract-test import.
import mjlab  # noqa: F401
import numpy as np
import torch

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_FOOT_SITES,
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_STANDING_JOINT_POS,
)
from scr_phybot_mjab.tasks.squat.config.phybot_c2.env_cfgs import (
  phybot_c2_squat_env_cfg,
)
from scr_phybot_mjab.tasks.squat.config.phybot_c2.rl_cfg import (
  phybot_c2_squat_ppo_runner_cfg,
)
from scr_phybot_mjab.tasks.squat.mdp import (
  SquatMotionCommand,
  SquatMotionCommandCfg,
  SquatMotionLoader,
)

_REPO = Path(__file__).resolve().parents[1]
_MOTION = _REPO / "src/scr_phybot_mjab/assets/motions/squat.npy"


class SquatMotionContractTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.motion = np.load(_MOTION, allow_pickle=True).item()

  def test_motion_starts_and_ends_at_zero_command_pose(self) -> None:
    joint_names = tuple(self.motion["joints_list"])
    joint_pos = np.asarray(self.motion["joint_positions"])
    zero_pos = np.asarray([PHYBOT_C2_STANDING_JOINT_POS[name] for name in joint_names])

    self.assertEqual(joint_names, tuple(PHYBOT_C2_POLICY_JOINT_NAMES))
    self.assertNotIn("neck_yaw", joint_names)
    self.assertNotIn("neck_pitch", joint_names)
    np.testing.assert_allclose(joint_pos[0], zero_pos, atol=2.0e-8)
    np.testing.assert_allclose(joint_pos[-1], zero_pos, atol=2.0e-8)
    np.testing.assert_allclose(
      self.motion["root_position"][[0, -1]],
      np.tile((0.0, 0.0, 0.726), (2, 1)),
    )
    np.testing.assert_allclose(
      self.motion["root_quaternion"][[0, -1]],
      np.tile((0.0, 0.0, 0.0, 1.0), (2, 1)),
    )

  def test_motion_has_three_second_standing_holds_and_no_seam(self) -> None:
    joint_pos = np.asarray(self.motion["joint_positions"])
    zero_pos = np.asarray([PHYBOT_C2_STANDING_JOINT_POS[name] for name in self.motion["joints_list"]])
    fps = float(self.motion["fps"])
    stand_frames = int(round(3.0 * fps))

    self.assertEqual(fps, 50.0)
    self.assertEqual(len(joint_pos), 650)
    stand_pose = np.broadcast_to(zero_pos, (stand_frames, len(zero_pos)))
    np.testing.assert_allclose(joint_pos[:stand_frames], stand_pose, atol=2.0e-6)
    np.testing.assert_allclose(joint_pos[-stand_frames:], stand_pose, atol=2.0e-6)
    self.assertGreater(
      float(np.max(np.abs(joint_pos[stand_frames:-stand_frames] - zero_pos))),
      0.5,
    )

    max_frame_delta = float(np.max(np.abs(np.diff(joint_pos, axis=0))))
    self.assertLess(max_frame_delta, 0.03)

    loader = SquatMotionLoader(
      (str(_MOTION),),
      tuple(PHYBOT_C2_POLICY_JOINT_NAMES),
      tuple(site for _, site in PHYBOT_C2_FOOT_SITES),
      device="cpu",
    )
    np.testing.assert_allclose(loader.joint_vel[[0, -1]].numpy(), 0.0)
    np.testing.assert_allclose(loader.root_lin_vel[[0, -1]].numpy(), 0.0)

  def test_episode_duration_and_joint_tracking_match_motion(self) -> None:
    cfg = phybot_c2_squat_env_cfg()
    play_cfg = phybot_c2_squat_env_cfg(play=True)
    command = cfg.commands["motion"]
    self.assertIsInstance(command, SquatMotionCommandCfg)

    frames = len(self.motion["joint_positions"])
    fps = float(self.motion["fps"])
    self.assertEqual(command.motion_files, (str(_MOTION),))
    self.assertEqual(command.sampling_mode, "start")
    self.assertEqual(command.foot_site_names, tuple(site for _, site in PHYBOT_C2_FOOT_SITES))
    self.assertEqual(cfg.episode_length_s, frames / fps)
    self.assertEqual(play_cfg.episode_length_s, frames / fps)
    self.assertEqual(frames, cfg.episode_length_s / (cfg.sim.mujoco.timestep * cfg.decimation))

    tracked = (
      tuple(cfg.rewards["track_legs"].params["joint_names"])
      + tuple(cfg.rewards["track_waist"].params["joint_names"])
      + tuple(cfg.rewards["track_arms"].params["joint_names"])
    )
    self.assertEqual(tracked, tuple(PHYBOT_C2_POLICY_JOINT_NAMES))
    self.assertEqual(len(set(tracked)), len(tracked))

  def test_reference_feet_are_fixed_for_the_complete_clip(self) -> None:
    foot_names = tuple(self.motion["foot_sites_list"])
    foot_positions = np.asarray(self.motion["foot_positions"])
    frames = len(self.motion["joint_positions"])

    self.assertEqual(foot_names, tuple(site for _, site in PHYBOT_C2_FOOT_SITES))
    self.assertEqual(foot_positions.shape, (frames, 2, 3))
    self.assertTrue(np.isfinite(foot_positions).all())
    np.testing.assert_allclose(foot_positions, np.broadcast_to(foot_positions[0], foot_positions.shape), atol=2e-4)

    cfg = phybot_c2_squat_env_cfg()
    self.assertEqual(cfg.rewards["track_feet_anchor"].weight, 5.0)
    self.assertEqual(cfg.rewards["track_feet_anchor"].params["std"], 0.02)
    self.assertEqual(cfg.rewards["track_feet_stationary"].weight, 3.0)
    self.assertEqual(cfg.rewards["track_feet_stationary"].params["std"], 0.04)
    self.assertEqual(cfg.rewards["foot_slip_error"].weight, -4.0)
    self.assertEqual(cfg.rewards["foot_slip_error"].params["beta"], 0.05)
    self.assertEqual(cfg.rewards["foot_anchor_error"].weight, -5.0)
    self.assertEqual(cfg.rewards["foot_anchor_error"].params["beta"], 0.02)
    self.assertEqual(cfg.rewards["self_collisions"].weight, -3.0)
    self.assertEqual(cfg.rewards["feet_distance"].params["min_dist"], 0.26)
    reference_separation = abs(float(foot_positions[0, 0, 1] - foot_positions[0, 1, 1]))
    self.assertGreater(reference_separation, 0.29)
    self.assertLess(cfg.rewards["feet_distance"].params["min_dist"], reference_separation)
    self.assertFalse(cfg.events["foot_friction"].params["shared_random"])

    self_collision = next(s for s in cfg.scene.sensors if s.name == "self_collision")
    self.assertEqual(self_collision.reduce, "maxforce")
    self.assertEqual(self_collision.num_slots, 1)
    self.assertEqual(self_collision.history_length, 4)

  def test_runner_does_not_clip_actions(self) -> None:
    self.assertIsNone(phybot_c2_squat_ppo_runner_cfg().clip_actions)

  def test_training_reset_covers_task_switch_state_but_play_is_exact(self) -> None:
    cfg = phybot_c2_squat_env_cfg()
    play_cfg = phybot_c2_squat_env_cfg(play=True)
    command = cfg.commands["motion"]
    play_command = play_cfg.commands["motion"]
    self.assertIsInstance(command, SquatMotionCommandCfg)
    self.assertIsInstance(play_command, SquatMotionCommandCfg)

    self.assertEqual(
      command.pose_range,
      {
        "z": (-0.005, 0.005),
        "roll": (-0.025, 0.025),
        "pitch": (-0.025, 0.025),
      },
    )
    self.assertEqual(
      command.velocity_range,
      {
        "x": (-0.075, 0.075),
        "y": (-0.075, 0.075),
        "z": (-0.025, 0.025),
        "roll": (-0.05, 0.05),
        "pitch": (-0.05, 0.05),
        "yaw": (-0.05, 0.05),
      },
    )
    self.assertEqual(command.joint_position_range, (-0.025, 0.025))
    self.assertEqual(command.joint_velocity_range, (-0.05, 0.05))

    self.assertEqual(play_command.pose_range, {})
    self.assertEqual(play_command.velocity_range, {})
    self.assertEqual(play_command.joint_position_range, (0.0, 0.0))
    self.assertEqual(play_command.joint_velocity_range, (0.0, 0.0))

  def test_phase_clock_ignores_runner_episode_length_randomization(self) -> None:
    command = object.__new__(SquatMotionCommand)
    command.cfg = SimpleNamespace(sampling_mode="start")
    command.motion = SimpleNamespace(time_step_total=650)
    command.time_steps = torch.tensor([0, 312, 649])
    command._env = SimpleNamespace(episode_length_buf=torch.tensor([477, 0, 12]))

    command._update_command()

    # Env 0 models RSL-RL randomizing the episode counter after a frame-zero
    # reset: phase advances to one instead of jumping to frame 478. Env 1 has
    # just reset and must stay at zero; env 2 verifies cyclic wrapping.
    torch.testing.assert_close(command.time_steps, torch.tensor([1, 0, 0]))

  def test_waist_reference_is_zero_and_has_independent_guards(self) -> None:
    motion_names = list(self.motion["joints_list"])
    waist_names = (PHYBOT_C2_POLICY_JOINT_NAMES[12],)
    waist_id = motion_names.index(waist_names[0])
    waist_reference = np.asarray(self.motion["joint_positions"])[:, waist_id]
    self.assertEqual(waist_names, ("waist_yaw",))
    np.testing.assert_allclose(
      waist_reference,
      PHYBOT_C2_STANDING_JOINT_POS[waist_names[0]],
      atol=1.0e-8,
    )

    cfg = phybot_c2_squat_env_cfg()
    for reward_name in (
      "track_waist",
      "waist_tracking_error",
      "track_waist_velocity",
      "waist_action_limits",
      "waist_reference_action",
    ):
      self.assertEqual(cfg.rewards[reward_name].params["joint_names"], waist_names)

    self.assertEqual(cfg.rewards["track_waist"].weight, 2.0)
    self.assertEqual(cfg.rewards["track_waist"].params["std"], 0.08)
    self.assertEqual(cfg.rewards["track_waist"].params["metric_prefix"], "waist")
    self.assertEqual(cfg.rewards["waist_tracking_error"].weight, -3.0)
    self.assertEqual(cfg.rewards["waist_tracking_error"].params["beta"], 0.10)
    self.assertEqual(cfg.rewards["waist_tracking_error"].params["reduction"], "sum")
    self.assertEqual(cfg.rewards["track_waist_velocity"].weight, 1.0)
    self.assertEqual(cfg.rewards["track_waist_velocity"].params["std"], 0.5)
    self.assertEqual(cfg.rewards["waist_action_limits"].weight, -0.5)
    self.assertEqual(cfg.rewards["waist_action_limits"].params["max_abs_actions"], (0.75,))
    self.assertEqual(cfg.rewards["waist_action_limits"].params["reduction"], "sum")
    self.assertEqual(cfg.rewards["waist_reference_action"].weight, -1.0)
    self.assertEqual(cfg.rewards["waist_reference_action"].params["beta"], 0.25)
    self.assertEqual(
      cfg.rewards["waist_reference_action"].params["max_abs_deviation"],
      (0.30,),
    )

    action_scale = 0.25
    self.assertEqual(0.30 * action_scale, 0.075)
    self.assertEqual(0.75 * action_scale, 0.1875)

  def test_arm_rewards_are_dense_and_reference_actions_have_safe_limits(self) -> None:
    cfg = phybot_c2_squat_env_cfg()
    arm_names = tuple(cfg.rewards["track_arms"].params["joint_names"])
    self.assertEqual(arm_names, tuple(PHYBOT_C2_POLICY_JOINT_NAMES[13:21]))
    self.assertEqual(cfg.rewards["track_arms"].weight, 3.0)
    self.assertEqual(cfg.rewards["track_arms"].params["std"], 0.20)
    self.assertEqual(cfg.rewards["track_arms"].params["worst_joint_weight"], 0.5)
    self.assertEqual(cfg.rewards["arm_tracking_error"].weight, -2.0)
    self.assertEqual(cfg.rewards["arm_tracking_error"].params["beta"], 0.25)
    self.assertEqual(cfg.rewards["arm_tracking_error"].params["reduction"], "sum")
    self.assertEqual(cfg.rewards["track_arm_velocity"].weight, 1.0)
    self.assertEqual(cfg.rewards["arm_action_limits"].weight, -0.5)
    self.assertEqual(cfg.rewards["arm_action_limits"].params["reduction"], "sum")
    self.assertEqual(cfg.rewards["arm_reference_action"].weight, -0.5)
    self.assertEqual(cfg.rewards["arm_reference_action"].params["beta"], 0.5)
    self.assertEqual(cfg.rewards["arm_position_symmetry"].weight, -2.0)
    self.assertEqual(cfg.rewards["arm_position_symmetry"].params["beta"], 0.10)
    self.assertEqual(cfg.rewards["arm_action_symmetry"].weight, -0.5)
    self.assertEqual(cfg.rewards["arm_action_symmetry"].params["beta"], 0.25)

    left_arm_names = arm_names[:4]
    right_arm_names = arm_names[4:]
    for reward_name in ("arm_position_symmetry", "arm_action_symmetry"):
      self.assertEqual(cfg.rewards[reward_name].params["left_joint_names"], left_arm_names)
      self.assertEqual(cfg.rewards[reward_name].params["right_joint_names"], right_arm_names)
    np.testing.assert_allclose(
      np.asarray(self.motion["joint_positions"])[
        :, [list(self.motion["joints_list"]).index(name) for name in left_arm_names]
      ],
      np.asarray(self.motion["joint_positions"])[
        :, [list(self.motion["joints_list"]).index(name) for name in right_arm_names]
      ],
      atol=1.0e-8,
    )

    limits = np.asarray(cfg.rewards["arm_action_limits"].params["max_abs_actions"])
    motion_names = list(self.motion["joints_list"])
    arm_ids = [motion_names.index(name) for name in arm_names]
    arm_default = np.asarray([PHYBOT_C2_STANDING_JOINT_POS[name] for name in arm_names])
    reference_actions = (np.asarray(self.motion["joint_positions"])[:, arm_ids] - arm_default) / 0.25
    self.assertEqual(limits.shape, (len(arm_names),))
    self.assertEqual(
      cfg.rewards["arm_reference_action"].params["max_abs_deviation"],
      (0.75,) * len(arm_names),
    )
    self.assertTrue(np.all(np.max(np.abs(reference_actions), axis=0) <= limits))


if __name__ == "__main__":
  unittest.main()
