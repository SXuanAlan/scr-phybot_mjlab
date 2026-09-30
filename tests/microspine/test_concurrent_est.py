"""Unit tests for the concurrent pi+phi (合跑) machinery."""

import pytest
import torch

from scr_phybot_mjab.rl.concurrent_est import (
  ConcurrentCfg,
  ConcurrentEstPPO,
  EstBuffer,
  PhiNet,
  ScheduleState,
)


def test_scheduled_sampling_anneal():
  cfg = ConcurrentCfg(sched_iters=100)
  assert cfg.p_truth_at(0) == pytest.approx(1.0)
  assert cfg.p_truth_at(50) == pytest.approx(0.5)
  assert cfg.p_truth_at(100) == pytest.approx(0.0)
  assert cfg.p_truth_at(10_000) == pytest.approx(0.0)  # floor, no wrap
  # Degenerate schedule: straight to ê.
  assert ConcurrentCfg(sched_iters=0).p_truth_at(0) == pytest.approx(0.0)


def test_pi_freeze_window():
  cfg = ConcurrentCfg(freeze_pi_iters=3)
  assert [cfg.pi_frozen_at(i) for i in range(5)] == [
    True,
    True,
    True,
    False,
    False,
  ]
  assert not ConcurrentCfg(freeze_pi_iters=0).pi_frozen_at(0)


def test_est_buffer_take_clears():
  buf = EstBuffer()
  assert buf.take() == (None, None)
  for _ in range(3):
    buf.append(torch.zeros(4, 89), torch.ones(4, 9))
  student, target = buf.take()
  assert student.shape == (12, 89) and target.shape == (12, 9)
  assert buf.take() == (None, None)


def test_phinet_matches_template_contract():
  """State-dict keys and forward semantics must match the inline Phi of
  scripts/train_estimator.py (the shipped-artifact compatibility gate)."""
  phi = PhiNet(keep_dim=80, tgt_dim=9, hidden=(32, 16))
  keys = set(phi.state_dict().keys())
  # Template attribute names: norm.* (EmpiricalNormalization buffers) and
  # mlp.* (Sequential Linear/ELU stack).
  assert any(k.startswith("norm.") for k in keys)
  assert any(k.startswith("mlp.") for k in keys)
  assert all(k.startswith(("norm.", "mlp.")) for k in keys)
  # Full student vector in, internal keep_dim slice (no-contact arm).
  out = phi(torch.randn(5, 89))
  assert out.shape == (5, 9)
  # TorchScript-able — the artifact format is a scripted module.
  scripted = torch.jit.script(phi.eval())
  out2 = scripted(torch.randn(5, 89))
  assert out2.shape == (5, 9)


def test_phi_pass_trains_and_reports():
  """The regression pass runs standalone on a stub algorithm object and
  actually reduces the loss on a learnable mapping."""
  torch.manual_seed(0)
  alg = object.__new__(ConcurrentEstPPO)
  phi = PhiNet(keep_dim=12, tgt_dim=3, hidden=(32,))
  cfg = ConcurrentCfg(phi_epochs=50, phi_minibatches=2, phi_lr=1e-2)
  buf = EstBuffer()
  state = ScheduleState()
  alg._est_buffer = buf
  alg._est_cfg = cfg
  alg._phi = phi
  alg._phi_opt = torch.optim.Adam(phi.parameters(), lr=cfg.phi_lr)
  alg._sched = state
  alg._est_iter = 0

  student = torch.randn(256, 12)
  target = student[:, :3] * 2.0 + 1.0  # learnable linear map
  buf.append(student, target)
  first = alg._phi_pass()
  assert "est/loss" in first and first["est/pi_frozen"] == 1.0
  # Buffer drained; a second pass with fresh data shows a lower loss.
  buf.append(student, target)
  second = alg._phi_pass()
  assert second["est/loss"] < first["est/loss"]
  # Empty buffer degrades gracefully.
  empty = alg._phi_pass()
  assert empty["est/loss"] == 0.0
