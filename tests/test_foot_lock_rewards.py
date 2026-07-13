from __future__ import annotations

import types
import unittest

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from scr_phybot_mjab.tasks.squat.mdp.rewards import (
  feet_xy_anchor_huber_error,
  feet_xy_velocity_huber_error,
  joint_action_limit_cost,
  joint_action_pair_symmetry_huber_error,
  joint_action_reference_tube_cost,
  joint_pair_symmetry_huber_error,
  joint_pos_tracking_huber_error,
  track_feet_anchor,
  track_feet_stationary,
  track_joint_pos_cauchy,
  track_joint_vel_cauchy,
)
from scr_phybot_mjab.tasks.walk.mdp.rewards import feet_lock_after_recovery


def _ns(**kwargs):
  return types.SimpleNamespace(**kwargs)


class _CommandManager:
  def __init__(self, command):
    self.command = command

  def get_term(self, _name):
    return self.command


class _ActionManager:
  def __init__(self, action_term):
    self.action_term = action_term
    self.total_action_dim = action_term.raw_action.shape[1]
    self.prev_action = torch.zeros_like(action_term.raw_action)

  def get_term(self, _name):
    return self.action_term


class FootRewardTest(unittest.TestCase):
  def _stand_env(self):
    robot = _ns(
      data=_ns(
        site_pos_w=torch.tensor(
          [[[0.0, 0.15, 0.0], [0.0, -0.15, 0.0]]], dtype=torch.float32
        ),
        site_lin_vel_w=torch.zeros(1, 2, 3),
      )
    )
    sensor = _ns(data=_ns(found=torch.ones(1, 2)))
    return _ns(
      scene={"robot": robot, "feet": sensor},
      device="cpu",
      step_dt=0.02,
      episode_length_buf=torch.zeros(1, dtype=torch.long),
      extras={"log": {}},
    )

  def test_stand_anchor_tracks_then_freezes_and_resets(self) -> None:
    env = self._stand_env()
    cfg = SceneEntityCfg("robot", site_ids=[0, 1])
    reward = feet_lock_after_recovery()
    params = {
      "sensor_name": "feet",
      "recovery_time_s": 3.0,
      "speed_scale": 0.10,
      "drift_scale": 0.05,
      "max_scaled_cost": 4.0,
      "missing_contact_cost": 4.0,
      "asset_cfg": cfg,
    }

    env.episode_length_buf[:] = 149
    env.scene["robot"].data.site_pos_w[:, :, 0] = 0.4
    self.assertEqual(float(reward(env, **params)), 0.0)

    env.episode_length_buf[:] = 150
    env.scene["robot"].data.site_pos_w[:, :, 0] = 0.5
    self.assertEqual(float(reward(env, **params)), 0.0)

    env.episode_length_buf[:] = 151
    env.scene["robot"].data.site_pos_w[:, 0, 0] += 0.05
    env.scene["robot"].data.site_lin_vel_w[:, 0, 0] = 0.10
    self.assertAlmostEqual(float(reward(env, **params)), 2.0, places=5)

    reward.reset(torch.tensor([0]))
    env.episode_length_buf[:] = 0
    env.scene["robot"].data.site_lin_vel_w.zero_()
    self.assertEqual(float(reward(env, **params)), 0.0)

  def test_squat_anchor_and_stationary_formulas(self) -> None:
    actual = torch.tensor(
      [[[0.01, 0.0, 0.0], [0.01, 0.30, 0.0]]], dtype=torch.float32
    )
    reference = torch.tensor(
      [[[0.0, 0.0, 0.0], [0.0, 0.30, 0.0]]], dtype=torch.float32
    )
    robot = _ns(data=_ns(site_pos_w=actual, site_lin_vel_w=torch.zeros(1, 2, 3)))
    command = _ns(robot=robot, foot_position_ref_w=reference)
    sensor = _ns(data=_ns(found=torch.ones(1, 2)))
    env = _ns(
      command_manager=_CommandManager(command),
      scene={"robot": robot, "feet": sensor},
      extras={"log": {}},
    )
    cfg = SceneEntityCfg("robot", site_ids=[0, 1])

    anchor = track_feet_anchor(env, "motion", 0.04, cfg)
    self.assertAlmostEqual(float(anchor), 16.0 / 17.0, places=6)
    stationary = track_feet_stationary(env, "feet", 0.10, cfg)
    self.assertEqual(float(stationary), 1.0)

    robot.data.site_lin_vel_w[:, 0, 0] = 0.10
    stationary = track_feet_stationary(env, "feet", 0.10, cfg)
    self.assertAlmostEqual(float(stationary), 0.75, places=6)
    slip = feet_xy_velocity_huber_error(env, 0.05, cfg)
    self.assertAlmostEqual(float(slip), 0.0375, places=6)
    anchor_cost = feet_xy_anchor_huber_error(env, "motion", 0.02, cfg)
    self.assertAlmostEqual(float(anchor_cost), 0.0025, places=6)

    sensor.data.found[:, 1] = 0
    self.assertEqual(float(track_feet_stationary(env, "feet", 0.10, cfg)), 0.0)

  def test_dense_arm_tracking_and_action_guard_formulas(self) -> None:
    robot = _ns(
      data=_ns(
        joint_pos=torch.tensor([[0.0, 0.4]]),
        joint_vel=torch.tensor([[0.0, 1.0]]),
        encoder_bias=torch.zeros(1, 2),
      )
    )
    command = _ns(
      cfg=_ns(joint_names=("arm_a", "arm_b")),
      joint_ids=torch.tensor([0, 1]),
      joint_pos_ref=torch.zeros(1, 2),
      joint_vel_ref=torch.zeros(1, 2),
      robot=robot,
    )
    action_term = _ns(
      target_names=["arm_a", "arm_b"],
      raw_action=torch.tensor([[1.2, 3.0]]),
      scale=1.0,
      offset=torch.zeros(1, 2),
    )
    env = _ns(
      command_manager=_CommandManager(command),
      action_manager=_ActionManager(action_term),
      device="cpu",
      extras={"log": {}},
    )
    joints = ("arm_a", "arm_b")

    position_score = track_joint_pos_cauchy(
      env, "motion", 0.2, joints, "arm"
    )
    self.assertAlmostEqual(float(position_score), 0.6, places=6)
    self.assertAlmostEqual(
      float(env.extras["log"]["Metrics/motion/arm_arm_a_pos_mae"]),
      0.0,
    )
    self.assertAlmostEqual(
      float(env.extras["log"]["Metrics/motion/arm_arm_b_pos_mae"]),
      0.4,
    )
    huber = joint_pos_tracking_huber_error(env, "motion", 0.25, joints)
    self.assertAlmostEqual(float(huber), 0.1375, places=6)
    velocity_score = track_joint_vel_cauchy(
      env, "motion", 1.0, joints, "arm"
    )
    self.assertAlmostEqual(float(velocity_score), 0.75, places=6)
    action_cost = joint_action_limit_cost(
      env,
      "joint_pos",
      joints,
      (1.0, 2.0),
      6.0,
      "arm",
    )
    self.assertAlmostEqual(float(action_cost), 0.52, places=6)
    self.assertAlmostEqual(
      float(env.extras["log"]["Metrics/motion/arm_action_limit_fraction"]),
      1.0,
    )
    reference_action_cost = joint_action_reference_tube_cost(
      env,
      "motion",
      "joint_pos",
      joints,
      (1.0, 2.0),
      0.5,
      "arm",
    )
    self.assertAlmostEqual(float(reference_action_cost), 0.79, places=6)
    position_symmetry = joint_pair_symmetry_huber_error(
      env, "motion", ("arm_a",), ("arm_b",), 0.25, "arm"
    )
    self.assertAlmostEqual(float(position_symmetry), 0.275, places=6)
    action_symmetry = joint_action_pair_symmetry_huber_error(
      env,
      "motion",
      "joint_pos",
      ("arm_a",),
      ("arm_b",),
      0.5,
      "arm",
    )
    self.assertAlmostEqual(float(action_symmetry), 1.55, places=6)


if __name__ == "__main__":
  unittest.main()
