from __future__ import annotations

import types
import unittest

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from scr_phybot_mjab.tasks.squat.mdp.rewards import (
  track_feet_anchor,
  track_feet_stationary,
)
from scr_phybot_mjab.tasks.walk.mdp.rewards import feet_lock_after_recovery


def _ns(**kwargs):
  return types.SimpleNamespace(**kwargs)


class _CommandManager:
  def __init__(self, command):
    self.command = command

  def get_term(self, _name):
    return self.command


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
    self.assertAlmostEqual(float(anchor), float(torch.exp(torch.tensor(-0.125))), places=6)
    stationary = track_feet_stationary(env, "feet", 0.10, cfg)
    self.assertEqual(float(stationary), 1.0)
    sensor.data.found[:, 1] = 0
    self.assertEqual(float(track_feet_stationary(env, "feet", 0.10, cfg)), 0.0)


if __name__ == "__main__":
  unittest.main()
