from types import SimpleNamespace

import torch

from scr_phybot_mjab.tasks.climb_wall.mdp.curriculums import (
  performance_gated_climb,
)


class FakeGait:
  def __init__(self, successes: list[bool]):
    self.episode_success = torch.tensor(successes)
    self.stage_kwargs = None

  def set_training_stage(self, **kwargs):
    self.stage_kwargs = kwargs


class FakeAffordance:
  def __init__(self):
    self.initial_names = None

  def set_initial_engaged_ee_names(self, names):
    self.initial_names = names


class FakeCommandManager:
  def __init__(self, gait, affordance):
    self.terms = {"climb_gait": gait, "microspine_affordance": affordance}

  def get_term(self, name):
    return self.terms[name]


def test_performance_curriculum_advances_and_applies_next_stage():
  stages = [
    {
      "angle_deg": 45.0,
      "initial_engaged_ee_names": ("lh", "rh", "lf", "rf"),
      "required_productive_regrips": 1,
      "randomize_start_limb": True,
    },
    {
      "angle_deg": 60.0,
      "initial_engaged_ee_names": ("lh", "rh"),
      "required_productive_regrips": 4,
      "randomize_start_limb": False,
    },
  ]
  gait = FakeGait([True, True, True, False])
  affordance = FakeAffordance()
  env = SimpleNamespace(
    command_manager=FakeCommandManager(gait, affordance),
    episode_length_buf=torch.ones(4, dtype=torch.long),
    device="cpu",
  )
  cfg = SimpleNamespace(params={"stages": stages})
  curriculum = performance_gated_climb(cfg, env)
  applied_gravity = []
  curriculum._set_gravity = lambda env, angle, magnitude: applied_gravity.append(
    (angle, magnitude)
  )

  state = curriculum(
    env,
    torch.arange(4),
    stages=stages,
    min_evaluated_episodes=4,
    advance_success_rate=0.7,
  )

  assert state["stage"].item() == 1.0
  assert state["success_rate"].item() == 0.75
  assert affordance.initial_names == ("lh", "rh")
  assert gait.stage_kwargs == {
    "required_productive_regrips": 4,
    "randomize_start_limb": False,
    # Stage omits the optional knobs -> command restores task defaults.
    "step_height": None,
    "gait_ee_names": None,
  }
  assert applied_gravity == [(60.0, 9.81)]
