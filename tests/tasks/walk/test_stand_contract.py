from __future__ import annotations

import unittest

from mjlab.envs.mdp import reset_joints_by_offset

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_POLICY_JOINT_NAMES,
)
from scr_phybot_mjab.tasks.walk.config.phybot_c2.env_cfgs import phybot_c2_walk_env_cfg
from scr_phybot_mjab.tasks.walk.mdp.rewards import feet_lock_after_recovery


class StandContractTest(unittest.TestCase):
  def test_training_has_collision_safe_recovery_reset_then_strict_lock(self) -> None:
    cfg = phybot_c2_walk_env_cfg(variant="stand")
    self.assertNotIn("push_robot", cfg.events)

    reset_legs = cfg.events["reset_robot_joints"]
    reset_upper = cfg.events["reset_robot_upper_joints"]
    self.assertIs(reset_legs.func, reset_joints_by_offset)
    self.assertIs(reset_upper.func, reset_joints_by_offset)
    self.assertEqual(reset_legs.params["position_range"], (-0.05, 0.05))
    self.assertEqual(reset_legs.params["velocity_range"], (-0.05, 0.05))
    self.assertEqual(reset_upper.params["position_range"], (-0.15, 0.15))
    self.assertEqual(reset_upper.params["velocity_range"], (-0.10, 0.10))
    tracked_reset_joints = tuple(reset_legs.params["asset_cfg"].joint_names) + tuple(
      reset_upper.params["asset_cfg"].joint_names
    )
    self.assertEqual(tracked_reset_joints, PHYBOT_C2_POLICY_JOINT_NAMES)

    self.assertNotIn("foot_slip", cfg.rewards)
    self.assertIs(cfg.rewards["stand_feet_lock"].func, feet_lock_after_recovery)
    self.assertEqual(cfg.rewards["stand_feet_lock"].weight, -0.25)
    self.assertEqual(cfg.rewards["stand_feet_lock"].params["recovery_time_s"], 3.0)
    self.assertEqual(cfg.rewards["stand_feet_contact"].weight, 1.0)
    self.assertEqual(cfg.rewards["self_collisions"].weight, -5.0)
    self.assertEqual(cfg.rewards["pose"].weight, 2.0)
    self.assertEqual(cfg.rewards["pose"].params["std_standing"], {".*": 0.12})
    self.assertEqual(
      tuple(cfg.rewards["pose"].params["asset_cfg"].joint_names),
      PHYBOT_C2_POLICY_JOINT_NAMES,
    )

    stages = cfg.curriculum["stand_feet_lock_weight"].params["weight_stages"]
    self.assertEqual([s["weight"] for s in stages], [-0.25, -0.50, -0.75])

  def test_play_reset_is_deterministic(self) -> None:
    cfg = phybot_c2_walk_env_cfg(play=True, variant="stand")
    reset_joints = cfg.events["reset_robot_joints"]
    reset_upper_joints = cfg.events["reset_robot_upper_joints"]
    reset_base = cfg.events["reset_base"]
    self.assertEqual(reset_joints.params["position_range"], (0.0, 0.0))
    self.assertEqual(reset_joints.params["velocity_range"], (0.0, 0.0))
    self.assertEqual(reset_upper_joints.params["position_range"], (0.0, 0.0))
    self.assertEqual(reset_upper_joints.params["velocity_range"], (0.0, 0.0))
    self.assertEqual(reset_base.params["pose_range"], {})
    self.assertEqual(reset_base.params["velocity_range"], {})
    self.assertEqual(cfg.events["body_friction"].params["ranges"], (1.0, 1.0))
    self.assertEqual(cfg.events["foot_friction"].params["ranges"], (0.7, 0.7))
    self.assertEqual(cfg.events["base_mass"].params["ranges"], (0.0, 0.0))
    self.assertEqual(cfg.curriculum, {})

  def test_walk_and_stand_self_collision_sensor_selects_strongest_contact(self) -> None:
    for cfg in (phybot_c2_walk_env_cfg(), phybot_c2_walk_env_cfg(variant="stand")):
      sensor = next(s for s in cfg.scene.sensors if s.name == "self_collision")
      self.assertEqual(sensor.reduce, "maxforce")
      self.assertEqual(sensor.num_slots, 1)
      self.assertEqual(sensor.history_length, 4)


if __name__ == "__main__":
  unittest.main()
