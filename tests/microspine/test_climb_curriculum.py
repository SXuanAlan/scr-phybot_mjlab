import torch
import pytest

from scr_phybot_mjab.tasks.climb_wall.mdp.curriculums import (
  sample_wall_angles,
  resolve_wall_stage,
)


def test_angle_jitter_samples_around_base_but_never_at_the_vertical_stage():
  torch.manual_seed(0)
  draws = sample_wall_angles(45.0, 5.0, 90.0, 4096)
  assert draws.min().item() >= 40.0
  assert draws.max().item() <= 50.0
  # Actually randomized, not a constant fill.
  assert draws.std().item() > 1.0
  # The vertical stage is exempt: every draw is exactly 90 — jitter there
  # would mix in >90 overhangs, a different task.
  vertical = sample_wall_angles(90.0, 5.0, 90.0, 128)
  assert torch.all(vertical == 90.0)
  # Jitter disabled (play) degenerates to the base at any stage.
  assert torch.all(sample_wall_angles(60.0, 0.0, 90.0, 8) == 60.0)


@pytest.mark.parametrize("iteration,low,high", [
  (0, 40.0, 50.0), (1499, 40.0, 50.0),
  (1500, 50.0, 60.0), (3299, 50.0, 60.0),
  (3300, 60.0, 60.0), (5150, 60.0, 60.0), (12000, 60.0, 60.0),
])
def test_corner_curriculum_advances_but_never_samples_above_60(iteration, low, high):
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Mjlab-Climb-Wall-Phybot-C2-Corners-A60")
  params = cfg.curriculum["wall_angle"].params
  stage = resolve_wall_stage(iteration, params["angle_stages"])
  jitter = stage.get("jitter_deg", params["angle_jitter_deg"])
  assert stage["angle_deg"] - jitter == low
  assert stage["angle_deg"] + jitter == high
  draws = sample_wall_angles(stage["angle_deg"], jitter, 90.0, 4096)
  assert draws.min() >= low
  assert draws.max() <= high
  if low == high:
    assert torch.all(draws == 60.0)
  else:
    assert draws.std() > 1.0
