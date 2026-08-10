import torch

from scr_phybot_mjab.tasks.climb_wall.mdp.curriculums import (
  sample_wall_angles,
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
