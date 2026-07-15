import pytest
import torch

from scr_phybot_mjab.tasks.climb_wall.mdp.actions import clamp_to_joint_limits


def test_clamp_to_joint_limits_is_joint_specific_and_reports_saturation():
  target = torch.tensor([[-2.0, 0.5, 3.0]])
  limits = torch.tensor([[[-1.0, 1.0], [-0.25, 0.25], [0.0, 2.0]]])

  clamped, saturated = clamp_to_joint_limits(target, limits)

  assert clamped.tolist() == [[-1.0, 0.25, 2.0]]
  assert saturated.tolist() == [[True, True, True]]


def test_clamp_to_joint_limits_rejects_mismatched_shapes():
  with pytest.raises(ValueError, match="does not match"):
    clamp_to_joint_limits(torch.zeros(2, 3), torch.zeros(2, 3))
