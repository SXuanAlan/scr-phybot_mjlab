from types import SimpleNamespace

import pytest
import torch

from scr_phybot_mjab.tasks.climb_wall.mdp.curriculums import (
  performance_gated_gravity,
)


class FakeMotion:
  def __init__(self, height_gained: list[float]):
    self.height_gained = torch.tensor(height_gained)


class FakeCommandManager:
  def __init__(self, motion):
    self.terms = {"climb_motion": motion}

  def get_term(self, name):
    return self.terms[name]


def test_performance_curriculum_advances_on_height_gain_success_rate():
  stages = [
    {"angle_deg": 45.0, "min_height_gain": 0.06},
    {"angle_deg": 60.0, "min_height_gain": 0.18},
  ]
  # 3 of 4 episodes gained at least the stage-0 threshold of 0.06 m.
  motion = FakeMotion([0.09, 0.30, 0.061, -0.02])
  env = SimpleNamespace(
    command_manager=FakeCommandManager(motion),
    episode_length_buf=torch.ones(4, dtype=torch.long),
    num_envs=4,
    device="cpu",
  )
  cfg = SimpleNamespace(params={"stages": stages})
  curriculum = performance_gated_gravity(cfg, env)
  applied_gravity = []
  curriculum._set_gravity = lambda env, angle, magnitude: applied_gravity.append(
    (angle, magnitude)
  )

  def evaluate():
    return curriculum(
      env,
      torch.arange(4),
      stages=stages,
      min_evaluated_episodes=4,
      advance_success_rate=0.7,
    )

  # First call: every episode began before this stage was applied (stale) and
  # must not grade it — stale successes once advanced two levels in a single
  # iteration and skipped a rung outright.
  state = evaluate()
  assert state["stage"].item() == 0.0

  # Second call: fresh episodes grade the stage and advance it.
  state = evaluate()
  assert state["stage"].item() == 1.0
  assert state["success_rate"].item() == 0.75
  assert state["angle_deg"].item() == 60.0
  assert state["min_height_gain"].item() == pytest.approx(0.18)
  assert applied_gravity == [(45.0, 9.81), (60.0, 9.81)]

  # Third call: envs went stale again on the advance, so the harder stage is
  # not graded by episodes that started under the easier one.
  state = evaluate()
  assert state["stage"].item() == 1.0


def test_success_is_judged_against_the_current_stage_threshold():
  stages = [
    {"angle_deg": 45.0, "min_height_gain": 0.06},
    {"angle_deg": 90.0, "min_height_gain": 0.18},
  ]
  # All episodes gain 0.10 m: enough for stage 0, not for stage 1.
  motion = FakeMotion([0.10, 0.10])
  env = SimpleNamespace(
    command_manager=FakeCommandManager(motion),
    episode_length_buf=torch.ones(2, dtype=torch.long),
    num_envs=2,
    device="cpu",
  )
  cfg = SimpleNamespace(params={"stages": stages})
  curriculum = performance_gated_gravity(cfg, env)
  curriculum._set_gravity = lambda *args: None

  def evaluate():
    return curriculum(
      env,
      torch.arange(2),
      stages=stages,
      min_evaluated_episodes=2,
      advance_success_rate=1.0,
    )

  evaluate()  # de-stale
  assert evaluate()["stage"].item() == 1.0  # 0.10 >= 0.06 everywhere
  evaluate()  # de-stale under stage 1
  assert evaluate()["stage"].item() == 1.0  # 0.10 < 0.18 -> no advance
  assert curriculum.last_success_rate == 0.0
