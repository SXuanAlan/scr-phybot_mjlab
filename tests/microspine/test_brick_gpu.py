"""Opt-in Warp integration; PHYBOT_BRICK_TEST_DEVICE=cpu supports CPU validation."""

import os

import mjlab  # noqa: F401
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv

from scr_phybot_mjab.microspine.surfaces import rotate
from scr_phybot_mjab.tasks.brick.env_cfg import brick_env_cfg
from scr_phybot_mjab.tasks.brick.geometry import active_unit_count

pytestmark = pytest.mark.skipif(
  os.environ.get("PHYBOT_BRICK_GPU_TEST") != "1", reason="Opt-in MuJoCo Warp GPU integration"
)


@pytest.mark.parametrize(
  "stage,randomized,enabled",
  [
    ("full", False, True),
    ("hold", False, True),
    ("engage", True, True),
    ("place", False, True),
    ("full", True, False),
  ],
)
@torch.inference_mode()
def test_parallel_task_reset_and_finite_physics(stage, randomized, enabled):
  cfg = brick_env_cfg(play=False, stage=stage, randomize=randomized, grip_enabled=enabled)
  cfg.scene.num_envs, cfg.auto_reset, cfg.seed = 4, False, 17
  env = ManagerBasedRlEnv(cfg, device=os.environ.get("PHYBOT_BRICK_TEST_DEVICE", "cuda:0"))
  try:
    obs, _ = env.reset()
    command = env.command_manager.get_term("brick_task")
    affordance = env.command_manager.get_term("microspine_affordance")
    executor = env.event_manager.get_term_cfg("microspine_grip").func
    expected = enabled and stage in ("hold", "place")
    assert bool(affordance.engaged.all()) == expected
    assert bool(env.sim.data.eq_active[:, executor._eq_ids].all()) == expected
    assert obs["actor"].shape == (4, 1170)
    assert obs["critic"].shape == (4, 156 + 2 * active_unit_count(command.cfg.unit_layout))
    torch.testing.assert_close(command.robot.data.joint_pos[:, command.joint_ids], command.joint_pos_ref)
    action_term = env.action_manager.get_term("joint_pos")
    action_term.process_actions(torch.zeros_like(command.joint_pos_ref))
    torch.testing.assert_close(action_term._processed_actions, command.joint_pos_ref)
    action_term.process_actions(torch.full_like(command.joint_pos_ref, 0.1))
    torch.testing.assert_close(action_term._processed_actions, command.joint_pos_ref + 0.025)
    if expected:
      point, quat = executor._surfaces.pose()
      torch.testing.assert_close(
        point + rotate(quat, executor.anchor_pos_local), executor.pad_center_world(), atol=1e-6, rtol=0
      )
    for _ in range(8):
      actions = torch.zeros_like(command.joint_pos_ref)
      obs, reward, terminated, truncated, _ = env.step(actions)
      assert torch.isfinite(reward).all()
      assert all(torch.isfinite(value).all() for value in obs.values())
      assert (env.sim.data.ne <= env.sim.data.njmax).all()
      done = (terminated | truncated).nonzero().flatten()
      if len(done):
        env.reset(env_ids=done)
    saved = {
      key: getattr(command, key)[1:].clone()
      for key in ("phase", "elapsed", "dwell", "measurement", "measurement_velocity", "pose_history", "delay")
    }
    anchors = env.sim.model.eq_data[1:, executor._eq_ids].clone()
    age = executor._engage_age[1:].clone()
    env.reset(env_ids=torch.tensor([0], device=env.device))
    for key, value in saved.items():
      torch.testing.assert_close(getattr(command, key)[1:], value)
    torch.testing.assert_close(env.sim.model.eq_data[1:, executor._eq_ids], anchors)
    torch.testing.assert_close(executor._engage_age[1:], age)
    assert bool(affordance.engaged[0].all()) == expected
    if not enabled:
      assert not affordance.engaged.any()
      assert not env.sim.data.eq_active[:, executor._eq_ids].any()
    if randomized:
      mass = env.sim.model.body_mass[:, env.sim.mj_model.body("brick/brick").id]
      assert ((mass >= 0.5) & (mass <= 1.5)).all()
      assert mass.std() > 0
  finally:
    env.close()
