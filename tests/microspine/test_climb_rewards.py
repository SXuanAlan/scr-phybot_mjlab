from types import SimpleNamespace

import pytest
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from scr_phybot_mjab.tasks.climb_wall.mdp.rewards import (
  bounded_joint_acc_l2,
  climb_height_progress,
  climb_gait_phase_timeout,
  commanded_anchor_advance,
  confirmed_anchor_advance,
  microspine_support_loss,
  productive_gait_cycle,
  signed_climb_height_progress,
  unscheduled_microspine_release,
  unsafe_microspine_release_penalty,
)
from scr_phybot_mjab.tasks.climb_wall.mdp.metrics import microspine_engage_events
from scr_phybot_mjab.tasks.climb_wall.mdp.terminations import no_progress_stall


class FakeCommandManager:
  def __init__(self, command: torch.Tensor, affordance=None):
    self.command = command
    self.affordance = affordance

  def get_command(self, name: str) -> torch.Tensor:
    del name
    return self.command

  def get_term(self, name: str):
    del name
    return self.affordance


class FakeEventManager:
  def __init__(self, physics):
    self.physics = physics

  def get_term_cfg(self, name: str):
    del name
    return SimpleNamespace(func=self.physics)


def make_progress_env(z: float = 1.0):
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, z]]),
      root_link_lin_vel_w=torch.zeros(1, 3),
    )
  )
  return SimpleNamespace(
    scene={"robot": robot},
    command_manager=FakeCommandManager(torch.tensor([[0.0, 0.0, 0.08]])),
    episode_length_buf=torch.zeros(1, dtype=torch.long),
    step_dt=0.02,
    num_envs=1,
    device="cpu",
  )


def test_highwater_progress_follows_settling_then_rewards_only_new_height():
  env = make_progress_env()
  reward = climb_height_progress()
  asset_cfg = SceneEntityCfg("robot")

  assert reward(env, asset_cfg=asset_cfg, command_name="climb").item() == 0.0

  env.episode_length_buf[:] = 25
  env.scene["robot"].data.root_link_pos_w[:, 2] = 0.95
  assert reward(env, asset_cfg=asset_cfg, command_name="climb").item() == 0.0

  env.episode_length_buf[:] = 26
  env.scene["robot"].data.root_link_pos_w[:, 2] += 0.0016
  assert reward(env, asset_cfg=asset_cfg, command_name="climb").item() == pytest.approx(
    1.0, abs=1.0e-4
  )

  env.episode_length_buf[:] = 27
  env.scene["robot"].data.root_link_pos_w[:, 2] -= 0.01
  assert reward(env, asset_cfg=asset_cfg, command_name="climb").item() == 0.0


def test_signed_progress_is_positive_up_and_negative_down():
  env = make_progress_env()
  reward = signed_climb_height_progress()
  asset_cfg = SceneEntityCfg("robot")

  reward(env, asset_cfg=asset_cfg, command_name="climb")
  env.episode_length_buf[:] = 25
  reward(env, asset_cfg=asset_cfg, command_name="climb")

  env.episode_length_buf[:] = 26
  env.scene["robot"].data.root_link_pos_w[:, 2] += 0.0016
  assert reward(env, asset_cfg=asset_cfg, command_name="climb").item() == pytest.approx(1.0)

  env.episode_length_buf[:] = 27
  env.scene["robot"].data.root_link_pos_w[:, 2] -= 0.0016
  assert reward(env, asset_cfg=asset_cfg, command_name="climb").item() == pytest.approx(-1.0)


def test_support_loss_penalizes_only_fewer_than_two_anchors():
  physics = SimpleNamespace(
    engaged=torch.tensor(
      [
        [True, True, False, False],
        [True, False, False, False],
        [True, True, True, True],
      ]
    )
  )
  env = SimpleNamespace(event_manager=FakeEventManager(physics))
  result = microspine_support_loss(env, minimum_supports=2)
  assert result.tolist() == [0.0, 1.0, 0.0]


def test_bounded_joint_acc_regularizer_normalizes_and_caps_impulses():
  robot = SimpleNamespace(
    data=SimpleNamespace(
      joint_acc=torch.tensor(
        [
          [100.0, -100.0],
          [1.0e6, -1.0e6],
          [float("nan"), 0.0],
        ]
      )
    )
  )
  env = SimpleNamespace(scene={"robot": robot})
  asset_cfg = SceneEntityCfg("robot", joint_ids=[0, 1])

  result = bounded_joint_acc_l2(
    env,
    asset_cfg=asset_cfg,
    acceleration_scale=100.0,
    max_cost=10.0,
  )

  assert result.tolist() == pytest.approx([1.0, 10.0, 10.0])


def test_release_cost_distinguishes_controlled_and_failed_release():
  affordance = SimpleNamespace(
    engaged=torch.tensor(
      [
        [True, True, True, False],
        [True, True, True, False],
        [True, False, False, False],
      ]
    ),
    forced_release_this_step=torch.tensor(
      [
        [True, False, False, False],
        [True, False, False, False],
        [True, False, False, False],
      ]
    ),
    release_reason=torch.tensor(
      [
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [4, 0, 0, 0],
      ]
    ),
  )
  env = SimpleNamespace(
    command_manager=FakeCommandManager(
      torch.tensor([[0.0, 0.0, 0.08]]), affordance
    ),
    step_dt=0.02,
    episode_length_buf=torch.full((3,), 100, dtype=torch.long),
  )
  result = unsafe_microspine_release_penalty(env)
  assert result.tolist() == pytest.approx([0.0, 12.5, 100.0])


def test_confirmed_higher_regrip_gets_one_shot_event_reward():
  physics = SimpleNamespace(
    engaged=torch.ones(1, 4, dtype=torch.bool),
    anchor_pos_w=torch.tensor(
      [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]]
    ),
    engage_age=torch.full((1, 4), 20, dtype=torch.long),
  )
  env = SimpleNamespace(
    event_manager=FakeEventManager(physics),
    episode_length_buf=torch.zeros(1, dtype=torch.long),
    step_dt=0.02,
  )
  reward = confirmed_anchor_advance()
  assert reward(env).item() == 0.0

  env.episode_length_buf[:] = 25
  assert reward(env).item() == 0.0

  # Six disengaged steps provide 0.12 s of genuine swing time.
  physics.engaged[:, 0] = False
  for step in range(26, 32):
    env.episode_length_buf[:] = step
    assert reward(env).item() == 0.0

  physics.engaged[:, 0] = True
  physics.anchor_pos_w[:, 0, 2] = 1.08
  physics.engage_age[:, 0] = 1
  env.episode_length_buf[:] = 32
  assert reward(env).item() == 0.0

  for age in range(2, 16):
    physics.engage_age[:, 0] = age
    env.episode_length_buf += 1
    assert reward(env).item() == 0.0

  physics.engage_age[:, 0] = 16
  env.episode_length_buf += 1
  # score=1, divided by dt so the manager's dt scaling yields a +1 event.
  assert reward(env).item() == pytest.approx(50.0)
  env.episode_length_buf += 1
  assert reward(env).item() == 0.0


def test_no_progress_stall_ignores_settling_and_resets_on_one_cm_gain():
  env = make_progress_env()
  termination = no_progress_stall()
  asset_cfg = SceneEntityCfg("robot")
  params = {
    "asset_cfg": asset_cfg,
    "command_name": "climb",
    "settle_time_s": 0.5,
    "stall_time_s": 0.06,
    "minimum_progress": 0.01,
  }

  assert not termination(env, **params).item()
  env.episode_length_buf[:] = 25
  assert not termination(env, **params).item()

  env.episode_length_buf[:] = 26
  assert not termination(env, **params).item()
  env.episode_length_buf[:] = 27
  assert not termination(env, **params).item()
  env.episode_length_buf[:] = 28
  assert termination(env, **params).item()

  env.scene["robot"].data.root_link_pos_w[:, 2] += 0.011
  env.episode_length_buf[:] = 29
  assert not termination(env, **params).item()


def test_gait_event_rewards_are_one_shot_values_after_dt_scaling():
  gait = SimpleNamespace(
    anchor_advance_this_step=torch.tensor([0.75, -0.25]),
    productive_cycle_this_step=torch.tensor([True, False]),
    unscheduled_release_count=torch.tensor([0.0, 2.0]),
    phase_timeout_this_step=torch.tensor([False, True]),
  )
  env = SimpleNamespace(
    command_manager=FakeCommandManager(torch.zeros(2, 3), gait),
    step_dt=0.02,
  )

  assert commanded_anchor_advance(env).tolist() == pytest.approx([37.5, -12.5])
  assert productive_gait_cycle(env).tolist() == pytest.approx([50.0, 0.0])
  assert unscheduled_microspine_release(env).tolist() == pytest.approx([0.0, 100.0])
  assert climb_gait_phase_timeout(env).tolist() == pytest.approx([0.0, 50.0])


def test_engage_event_metric_counts_affordance_regrips_not_reset_anchors():
  affordance = SimpleNamespace(
    engaged_this_step=torch.tensor(
      [[False, True, False, False], [False, False, False, False]]
    )
  )
  env = SimpleNamespace(
    command_manager=FakeCommandManager(torch.zeros(2, 3), affordance),
    step_dt=0.02,
  )

  assert microspine_engage_events(env).tolist() == pytest.approx([50.0, 0.0])
