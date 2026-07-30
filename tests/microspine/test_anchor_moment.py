import torch

from scr_phybot_mjab.microspine.affordance import moment_adjusted_loads
from scr_phybot_mjab.microspine.constraint import (
  weld_rotational_force_to_world_moment,
)


def test_weld_physical_moment_is_half_the_rotational_multiplier_at_lock():
  rotational_force = torch.tensor([[[2.0, -4.0, 6.0]]])
  identity = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])

  moment = weld_rotational_force_to_world_moment(
    rotational_force,
    identity,
    identity,
    torquescale=1.0,
  )

  torch.testing.assert_close(moment, torch.tensor([[[1.0, -2.0, 3.0]]]))


def test_weld_rotational_conversion_includes_quaternion_residual_jacobian():
  # q_err = (cos 30°, 0, 0, sin 30°). For lambda along +x,
  # mujoco_warp uses J_rot = 0.5 * (w I - skew(v)), hence
  # J_rot.T lambda = 0.5 * (w*lambda + v x lambda).
  body = torch.tensor([[[0.8660254, 0.0, 0.0, 0.5]]])
  anchor = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
  rotational_force = torch.tensor([[[2.0, 0.0, 0.0]]])

  moment = weld_rotational_force_to_world_moment(
    rotational_force,
    body,
    anchor,
    torquescale=1.0,
  )

  torch.testing.assert_close(
    moment, torch.tensor([[[0.8660254, 0.5, 0.0]]]), atol=1.0e-6, rtol=1.0e-6
  )


def test_moments_fold_into_worst_edge_forces_with_compression_credit():
  f_t = torch.tensor([[80.0]])
  f_l = torch.tensor([[-5.0]])
  f_n = torch.tensor([[-30.0]])
  m_t = torch.tensor([[0.10]])
  m_l = torch.tensor([[-0.20]])
  m_n = torch.tensor([[0.15]])

  effective = moment_adjusted_loads(
    f_t,
    f_l,
    f_n,
    m_t,
    m_l,
    m_n,
    pad_span_rake_m=0.04,
    pad_span_lateral_m=0.08,
    moment_arm_fraction=0.25,
  )

  # Arms are 10 mm rake and 20 mm lateral:
  # normal = -30 + .20/.01 + .10/.02 = -5 N (still compressed);
  # lateral = |-5| + .15/.01 = 20 N.
  torch.testing.assert_close(effective[0], torch.tensor([[80.0]]))
  torch.testing.assert_close(effective[1], torch.tensor([[20.0]]))
  torch.testing.assert_close(effective[2], torch.tensor([[-5.0]]))
