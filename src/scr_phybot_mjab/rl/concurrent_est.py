"""Concurrent policy + estimator training (the L-E 合跑).

One run replaces the four-stage 2a/2b/2c/Phase-3 pipeline
(docs/plans/2026-08-10_unified_mount_climb_policy_plan.md §7): PPO
fine-tunes pi while phi regresses the privileged truth on the SAME
on-policy rollouts, and scheduled sampling anneals the actor's privileged
slices from truth to phi's estimates (Ji et al., RA-L 2022 — from-scratch
concurrent training is the quadruped standard; ours warm-starts pi).

Safeguards against early-garbage-ê feedback, both plan §7 items:

* scheduled sampling — each env row independently keeps the TRUE
  privileged slices with probability ``p_truth`` (1 -> 0 over
  ``sched_iters``); estimate rows also zero the deploy-dropped slices, so
  the anneal moves the whole obs contract from teacher-native to
  deploy-native;
* pi freeze — the first ``freeze_pi_iters`` iterations skip the PPO update
  entirely (phi catches up on a healthy state distribution first; pi is
  warm-started anyway).

The artifact contract is UNCHANGED: ``PhiNet`` state-dict keys and the
TorchScript/meta-json layout match scripts/train_estimator.py's inline
model one-for-one, so the raster ``estimator=`` token, estfreeze and the
C++ packer consume the 合跑 output as-is.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rsl_rl.algorithms import PPO
from rsl_rl.modules import EmpiricalNormalization

__all__ = (
  "ConcurrentCfg",
  "ConcurrentEstPPO",
  "EstBuffer",
  "PhiNet",
  "ScheduleState",
)


class PhiNet(nn.Module):
  """phi(student history) -> privileged estimates, template-compatible.

  Attribute names (``norm``, ``mlp``) and forward semantics mirror the
  inline ``Phi`` of scripts/train_estimator.py exactly: full student
  vector in, sliced to ``keep_dim`` INSIDE the module (the no-contact arm
  drops the pad load-cell tail without changing the caller contract).
  """

  def __init__(
    self,
    keep_dim: int,
    tgt_dim: int,
    hidden: tuple[int, ...] = (512, 256),
  ) -> None:
    super().__init__()
    self.keep_dim = keep_dim
    self.norm = EmpiricalNormalization(shape=[keep_dim])
    layers: list[nn.Module] = []
    last = keep_dim
    for h in hidden:
      layers += [nn.Linear(last, h), nn.ELU()]
      last = h
    layers.append(nn.Linear(last, tgt_dim))
    self.mlp = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.mlp(self.norm(x[:, : self.keep_dim]))


class EstBuffer:
  """Rollout-time capture of (student obs, privileged targets) rows."""

  def __init__(self) -> None:
    self._student: list[torch.Tensor] = []
    self._target: list[torch.Tensor] = []

  def append(self, student: torch.Tensor, target: torch.Tensor) -> None:
    self._student.append(student)
    self._target.append(target)

  def take(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    if not self._student:
      return None, None
    out = (torch.cat(self._student, dim=0), torch.cat(self._target, dim=0))
    self._student, self._target = [], []
    return out


@dataclass
class ScheduleState:
  """Shared live state: the algorithm writes at update time, the env
  wrapper reads at rollout time (the next rollout sees the new values)."""

  p_truth: float = 1.0
  pi_frozen: bool = True


@dataclass
class ConcurrentCfg:
  """合跑 knobs (plan §7)."""

  sched_iters: int = 1000
  freeze_pi_iters: int = 100
  phi_lr: float = 3.0e-4
  phi_epochs: int = 2
  phi_minibatches: int = 4

  def p_truth_at(self, iteration: int) -> float:
    if self.sched_iters <= 0:
      return 0.0
    return max(0.0, 1.0 - iteration / self.sched_iters)

  def pi_frozen_at(self, iteration: int) -> bool:
    return iteration < self.freeze_pi_iters


class ConcurrentEstPPO(PPO):
  """PPO whose update also trains phi on the rollout's captured rows.

  Constructed by the stock runner path (``algorithm.class_name`` pointing
  here); estimation state attached afterwards by the training script.
  Unattached, behaves exactly like PPO.
  """

  _est_buffer: EstBuffer | None = None
  _est_cfg: ConcurrentCfg | None = None
  _phi: PhiNet | None = None
  _phi_opt: torch.optim.Optimizer | None = None
  _sched: ScheduleState | None = None
  _est_iter: int = 0

  def attach_estimation(
    self,
    buffer: EstBuffer,
    phi: PhiNet,
    cfg: ConcurrentCfg,
    state: ScheduleState,
  ) -> None:
    self._est_buffer = buffer
    self._est_cfg = cfg
    self._phi = phi
    self._phi_opt = torch.optim.Adam(phi.parameters(), lr=cfg.phi_lr)
    self._sched = state
    self._est_iter = 0
    state.p_truth = cfg.p_truth_at(0)
    state.pi_frozen = cfg.pi_frozen_at(0)

  def update(self) -> dict[str, float]:
    if self._est_buffer is None:
      return super().update()
    assert self._est_cfg is not None and self._sched is not None
    if self._est_cfg.pi_frozen_at(self._est_iter):
      # phi-only iteration: PPO skipped, but the rollout storage must
      # still drain (PPO.update ends with storage.clear()).
      self.storage.clear()
      losses: dict[str, float] = {}
    else:
      losses = super().update()
    losses.update(self._phi_pass())
    self._est_iter += 1
    # Publish the NEXT rollout's schedule.
    self._sched.p_truth = self._est_cfg.p_truth_at(self._est_iter)
    self._sched.pi_frozen = self._est_cfg.pi_frozen_at(self._est_iter)
    return losses

  def _phi_pass(self) -> dict[str, float]:
    assert (
      self._est_buffer is not None
      and self._est_cfg is not None
      and self._phi is not None
      and self._phi_opt is not None
      and self._sched is not None
    )
    student, target = self._est_buffer.take()
    frozen = float(self._est_cfg.pi_frozen_at(self._est_iter))
    if student is None:
      return {
        "est/loss": 0.0,
        "est/p_truth": self._sched.p_truth,
        "est/pi_frozen": frozen,
      }
    cfg = self._est_cfg
    self._phi.train()
    total, count = 0.0, 0
    n = student.shape[0]
    for _ in range(cfg.phi_epochs):
      perm = torch.randperm(n, device=student.device)
      for mb in perm.chunk(cfg.phi_minibatches):
        stu = student[mb]
        pred = self._phi(stu)
        loss = nn.functional.mse_loss(pred, target[mb])
        self._phi_opt.zero_grad()
        loss.backward()
        self._phi_opt.step()
        # Explicit-update pattern of the template: forward never learns
        # the normalizer, update() does (gated by train mode).
        self._phi.norm.update(stu[:, : self._phi.keep_dim].detach())
        total += float(loss.item())
        count += 1
    return {
      "est/loss": total / max(count, 1),
      "est/p_truth": self._sched.p_truth,
      "est/pi_frozen": frozen,
    }
