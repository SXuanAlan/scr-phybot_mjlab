"""Unit tests for the segment-routed KL distillation machinery."""

import math

import pytest
import torch
from tensordict import TensorDict

from scr_phybot_mjab.rl.kl_distill import (
  DistillBuffer,
  DistillCfg,
  KLDistillPPO,
  gaussian_kl,
)


def test_gaussian_kl_closed_form():
  mean = torch.zeros(3, 4)
  std = torch.ones(3, 4)
  # Identical distributions: zero.
  assert torch.allclose(gaussian_kl(mean, std, mean, std), torch.zeros(3))
  # Known value: KL(N(0,1) || N(1,1)) = 0.5 per dim.
  shifted = torch.ones(3, 4)
  assert torch.allclose(
    gaussian_kl(mean, std, shifted, std), torch.full((3,), 2.0)
  )
  # Teacher confidence weighting: a wider teacher tolerates the same mean
  # gap more cheaply.
  wide = torch.full((3, 4), 2.0)
  assert bool(
    (gaussian_kl(mean, std, shifted, wide) < gaussian_kl(mean, std, shifted, std))
    .all()
  )


def test_distill_buffer_take_clears():
  buf = DistillBuffer()
  assert buf.take() == (None, None, None)
  for _ in range(3):
    buf.append(torch.zeros(4, 5), torch.zeros(4, dtype=torch.bool), torch.ones(4))
  obs, route, fade = buf.take()
  assert obs.shape == (12, 5) and route.shape == (12,) and fade.shape == (12,)
  assert buf.take() == (None, None, None)


def test_lambda_anneal_schedule():
  cfg = DistillCfg(lambda0=1.0, lambda_final=0.1, anneal_iters=100)
  assert cfg.lambda_at(0) == pytest.approx(1.0)
  assert cfg.lambda_at(50) == pytest.approx(0.55)
  assert cfg.lambda_at(100) == pytest.approx(0.1)
  assert cfg.lambda_at(10_000) == pytest.approx(0.1)  # floor, not zero-cross


class _FakeTeacher:
  """Minimal actor stand-in: constant mean per teacher, unit std."""

  def __init__(self, value: float, dims: int):
    self._value = value
    self._dims = dims
    self._params = None

  def __call__(self, td: TensorDict, stochastic_output: bool = True):
    n = td.batch_size[0]
    mean = torch.full((n, self._dims), self._value)
    self._params = (mean, torch.ones_like(mean))

  @property
  def output_distribution_params(self):
    return self._params


def test_teacher_routing_by_segment():
  alg = object.__new__(KLDistillPPO)
  alg._teachers = (_FakeTeacher(-1.0, 4), _FakeTeacher(2.0, 4))
  alg._num_actions = 4
  obs = torch.zeros(6, 8)
  route = torch.tensor([False, True, False, True, True, False])
  t_mean, t_std = alg._teacher_params(obs, route)
  # Mount rows (route False) answered by teacher 0, cycle rows by teacher 1.
  assert torch.allclose(t_mean[~route], torch.full((3, 4), -1.0))
  assert torch.allclose(t_mean[route], torch.full((3, 4), 2.0))
  assert torch.allclose(t_std, torch.ones(6, 4))
  # Single-segment batches route cleanly too (the empty-side guard).
  t_mean, _ = alg._teacher_params(obs[:2], torch.tensor([True, True]))
  assert torch.allclose(t_mean, torch.full((2, 4), 2.0))


def test_fade_zeroes_the_seam_contribution():
  # The loss weighting is (per_sample * fade).mean(): fade rows at 0 make
  # exactly zero contribution regardless of teacher disagreement.
  s_mean = torch.zeros(4, 2)
  s_std = torch.ones(4, 2)
  t_mean = torch.ones(4, 2) * 10.0  # wildly disagreeing teacher
  t_std = torch.ones(4, 2)
  per = gaussian_kl(s_mean, s_std, t_mean, t_std)
  fade = torch.tensor([1.0, 0.0, 1.0, 0.0])
  weighted = per * fade
  assert weighted[1] == 0.0 and weighted[3] == 0.0
  assert weighted[0] == pytest.approx(per[0].item())
  assert math.isfinite(float(weighted.mean()))
