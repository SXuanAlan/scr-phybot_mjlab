from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

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
from scr_phybot_mjab.tasks.squat.mdp import SquatMotionCommandCfg, SquatMotionLoader


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

  def test_motion_has_no_discontinuous_stand_clip_seam(self) -> None:
    joint_pos = np.asarray(self.motion["joint_positions"])
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
    self.assertEqual(
      command.foot_site_names, tuple(site for _, site in PHYBOT_C2_FOOT_SITES)
    )
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
    np.testing.assert_allclose(
      foot_positions, np.broadcast_to(foot_positions[0], foot_positions.shape), atol=2e-4
    )

    cfg = phybot_c2_squat_env_cfg()
    self.assertEqual(cfg.rewards["track_feet_anchor"].weight, 3.0)
    self.assertEqual(cfg.rewards["track_feet_anchor"].params["std"], 0.04)
    self.assertEqual(cfg.rewards["track_feet_stationary"].params["std"], 0.10)
    self.assertEqual(cfg.rewards["self_collisions"].weight, -3.0)
    self.assertEqual(cfg.rewards["feet_distance"].params["min_dist"], 0.15)

    self_collision = next(s for s in cfg.scene.sensors if s.name == "self_collision")
    self.assertEqual(self_collision.reduce, "maxforce")
    self.assertEqual(self_collision.num_slots, 1)
    self.assertEqual(self_collision.history_length, 4)

  def test_reference_action_fits_configured_clip(self) -> None:
    joint_pos = np.asarray(self.motion["joint_positions"])
    zero_pos = np.asarray(
      [PHYBOT_C2_STANDING_JOINT_POS[name] for name in PHYBOT_C2_POLICY_JOINT_NAMES]
    )
    max_reference_action = float(np.max(np.abs((joint_pos - zero_pos) / 0.25)))
    self.assertLessEqual(max_reference_action, phybot_c2_squat_ppo_runner_cfg().clip_actions)


if __name__ == "__main__":
  unittest.main()
