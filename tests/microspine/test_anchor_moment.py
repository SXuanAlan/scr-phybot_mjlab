import torch

from scr_phybot_mjab.microspine.anchor_loads import weld_rotational_force_to_world_moment
from scr_phybot_mjab.microspine.wrench import distribute_wrench_to_units


def _unit_grid(rows: int, cols: int, span_r: float, span_l: float):
  xs = torch.linspace(-0.5, 0.5, rows) * span_r
  ys = torch.linspace(-0.5, 0.5, cols) * span_l
  gx, gy = torch.meshgrid(xs, ys, indexing="ij")
  x, y = gx.reshape(-1), gy.reshape(-1)
  return x, y, float((x**2).sum()), float((y**2).sum())


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

  torch.testing.assert_close(moment, torch.tensor([[[0.8660254, 0.5, 0.0]]]), atol=1.0e-6, rtol=1.0e-6)


def test_per_unit_distribution_reproduces_the_pad_wrench():
  x, y, sxx, syy = _unit_grid(3, 4, 0.0747, 0.096)
  f_t = torch.tensor([[80.0]])
  f_l = torch.tensor([[-5.0]])
  f_n = torch.tensor([[12.0]])
  m_t = torch.tensor([[0.40]])
  m_l = torch.tensor([[-0.70]])
  m_n = torch.tensor([[0.25]])

  a, b, c = distribute_wrench_to_units(f_t, f_l, f_n, m_t, m_l, m_n, x, y, sxx, syy, sxx + syy)

  # Force totals recover the pad wrench...
  torch.testing.assert_close(a.sum(dim=-1), f_t)
  torch.testing.assert_close(b.sum(dim=-1), f_l)
  torch.testing.assert_close(c.sum(dim=-1), f_n)
  # ...and so do the engage-frame cross products (m_t = sum y*c,
  # m_l = -sum x*c, m_n = sum x*b - y*a).
  torch.testing.assert_close((y * c).sum(dim=-1), m_t)
  torch.testing.assert_close(-(x * c).sum(dim=-1), m_l)
  torch.testing.assert_close((x * b - y * a).sum(dim=-1), m_n)


def test_per_unit_two_row_grid_reproduces_the_worst_edge_fold():
  # A 2x2 grid is the layout the legacy fold is exact for: the worst
  # unit's normal load, scaled back to pad scale (x4), must equal
  # f_n + |M_l|/(span/2).
  span_r = 0.08
  x, y, sxx, syy = _unit_grid(2, 2, span_r, 0.096)
  f_n = torch.tensor([[8.0]])
  m_l = torch.tensor([[-0.20]])
  zero = torch.zeros_like(f_n)

  _, _, c = distribute_wrench_to_units(zero, zero, f_n, zero, m_l, zero, x, y, sxx, syy, sxx + syy)

  folded = f_n + m_l.abs() / (span_r / 2.0)
  torch.testing.assert_close(c.max(dim=-1).values * 4.0, folded)


def test_per_unit_three_row_grid_bills_the_discrete_row_arm():
  # N evenly spaced rows under pure bending: pad-scale worst-unit load
  # = M / arm with arm = span*(N+1)/(6*(N-1)); three rows -> span/3.
  span_r = 0.09
  x, y, sxx, syy = _unit_grid(3, 2, span_r, 0.05)
  m_l = torch.tensor([[0.30]])
  zero = torch.zeros_like(m_l)

  _, _, c = distribute_wrench_to_units(zero, zero, zero, zero, m_l, zero, x, y, sxx, syy, sxx + syy)

  n_units = x.shape[-1]
  expected = m_l / (span_r / 3.0)
  torch.testing.assert_close(c.max(dim=-1).values * n_units, expected)
