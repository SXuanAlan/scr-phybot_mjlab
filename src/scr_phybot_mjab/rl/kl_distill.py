"""Segment-routed KL distillation on top of PPO (LadderMan Eq. 3).

The unified mount+climb student optimizes ``L_PPO + lambda(t) * KL(student
|| routed frozen teacher)`` — imitation as an annealed leash, not a
ceiling. Mechanics per docs/plans/2026-08-10_unified_mount_climb_policy_plan.md
§6.2:

* one observation feeds all three networks (the teachers share the
  student's actor contract by construction);
* the teacher is ROUTED by clip segment: mount teacher before the handoff
  frame, cycle teacher after — with a fade band around the seam where no
  teacher has coverage and the KL weight drops to zero (RL owns the seam);
* lambda anneals linearly to a floor, so late training is task-driven.

Implementation shape: a ``DistillBuffer`` is filled by the env wrapper at
rollout time (actor obs + route bit + fade weight, one row per env step);
``KLDistillPPO.update()`` runs the stock PPO update, then a separate
imitation gradient pass over the buffer (two-step approximation of the
joint objective — no rsl_rl internals are copied, so this survives
version drift). With no distillation attached the class IS plain PPO.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl.algorithms import PPO

__all__ = (
  "DistillBuffer",
  "DistillCfg",
  "KLDistillPPO",
  "gaussian_kl",
  "load_frozen_actor",
)


def gaussian_kl(
  s_mean: torch.Tensor,
  s_std: torch.Tensor,
  t_mean: torch.Tensor,
  t_std: torch.Tensor,
) -> torch.Tensor:
  """Per-sample KL(N_s || N_t) for diagonal Gaussians, summed over dims.

  The LadderMan direction — student || teacher: the student is penalized
  for putting mass where the teacher has none, so it stays inside the
  teacher's support while lambda is high.
  """
  var_t = t_std.square()
  per_dim = (
    torch.log(t_std / s_std)
    + (s_std.square() + (s_mean - t_mean).square()) / (2.0 * var_t)
    - 0.5
  )
  return per_dim.sum(dim=-1)


class DistillBuffer:
  """Rollout-time capture: (actor obs, route bit, fade weight) per step.

  Filled by the env wrapper, drained once per PPO update. Rows pair the
  obs each network SEES with the segment that owns it — computed from the
  env's unified clock at capture time, so the update pass needs no access
  to env state.
  """

  def __init__(self) -> None:
    self._obs: list[torch.Tensor] = []
    self._route: list[torch.Tensor] = []
    self._fade: list[torch.Tensor] = []

  def append(
    self, obs: torch.Tensor, route: torch.Tensor, fade: torch.Tensor
  ) -> None:
    self._obs.append(obs)
    self._route.append(route)
    self._fade.append(fade)

  def take(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    None, None, None
  ]:
    if not self._obs:
      return None, None, None
    out = (
      torch.cat(self._obs, dim=0),
      torch.cat(self._route, dim=0),
      torch.cat(self._fade, dim=0),
    )
    self._obs, self._route, self._fade = [], [], []
    return out


@dataclass
class DistillCfg:
  """Distillation knobs (see the plan's §6.3 for the tuning doctrine)."""

  lambda0: float = 1.0
  lambda_final: float = 0.0
  anneal_iters: int = 2000
  epochs: int = 1
  num_minibatches: int = 4
  # "kl" = closed-form Gaussian KL (teacher std as confidence weighting);
  # "mse" = plain mean regression (HOVER-style, the bring-up mode).
  mode: str = "kl"

  def lambda_at(self, iteration: int) -> float:
    if self.anneal_iters <= 0:
      return self.lambda_final
    frac = min(max(iteration / self.anneal_iters, 0.0), 1.0)
    return self.lambda0 + (self.lambda_final - self.lambda0) * frac


class KLDistillPPO(PPO):
  """PPO + annealed segment-routed imitation pass (see module docstring).

  Constructed by the stock runner path (``algorithm.class_name`` pointing
  here); the distillation state is attached afterwards by the training
  script via :meth:`attach_distillation`. Unattached, behaves exactly like
  PPO.
  """

  _distill_buffer: DistillBuffer | None = None
  _distill_cfg: DistillCfg | None = None
  _teachers: tuple[nn.Module, nn.Module] | None = None
  _distill_iter: int = 0
  _num_actions: int = 0

  def attach_distillation(
    self,
    buffer: DistillBuffer,
    mount_teacher: nn.Module,
    cycle_teacher: nn.Module,
    cfg: DistillCfg,
    num_actions: int,
  ) -> None:
    self._distill_buffer = buffer
    self._distill_cfg = cfg
    self._teachers = (mount_teacher, cycle_teacher)
    self._distill_iter = 0
    self._num_actions = num_actions

  def update(self) -> dict[str, float]:
    losses = super().update()
    if self._distill_buffer is not None:
      losses.update(self._distill_pass())
    return losses

  @torch.no_grad()
  def _teacher_params(
    self, obs: torch.Tensor, route: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Routed teacher (mean, std): mount rows -> teacher 0, cycle -> 1."""
    assert self._teachers is not None
    t_mean = torch.empty(obs.shape[0], self._num_actions, device=obs.device)
    t_std = torch.empty_like(t_mean)
    for flag, teacher in ((False, self._teachers[0]), (True, self._teachers[1])):
      sel = route == flag
      if not bool(sel.any()):
        continue
      td = TensorDict({"actor": obs[sel]}, batch_size=[int(sel.sum())])
      teacher(td, stochastic_output=True)
      mean, std = teacher.output_distribution_params
      t_mean[sel] = mean
      t_std[sel] = std
    return t_mean, t_std

  def _distill_pass(self) -> dict[str, float]:
    assert self._distill_buffer is not None and self._distill_cfg is not None
    cfg = self._distill_cfg
    lam = cfg.lambda_at(self._distill_iter)
    self._distill_iter += 1
    obs, route, fade = self._distill_buffer.take()
    if obs is None or lam <= 0.0:
      return {"distill/lambda": lam, "distill/loss": 0.0}

    total, count = 0.0, 0
    n = obs.shape[0]
    for _ in range(cfg.epochs):
      perm = torch.randperm(n, device=obs.device)
      for mb in perm.chunk(cfg.num_minibatches):
        o, r, f = obs[mb], route[mb], fade[mb]
        t_mean, t_std = self._teacher_params(o, r)
        td = TensorDict({"actor": o}, batch_size=[o.shape[0]])
        self.actor(td, stochastic_output=True)
        s_mean, s_std = self.actor.output_distribution_params
        if cfg.mode == "kl":
          per_sample = gaussian_kl(s_mean, s_std, t_mean, t_std)
        else:
          per_sample = (s_mean - t_mean).square().sum(dim=-1)
        loss = lam * (per_sample * f).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()
        total += float(loss.item())
        count += 1
    return {
      "distill/lambda": lam,
      "distill/loss": total / max(count, 1),
      "distill/fade_share": float(1.0 - fade.mean().item()),
    }


def load_frozen_actor(
  template_actor: nn.Module, checkpoint: str | Path, device: str
) -> nn.Module:
  """Frozen teacher actor from a PPO checkpoint, arch-copied from the
  student's raw actor (identical contract by construction)."""
  teacher = copy.deepcopy(template_actor).to(device)
  loaded = torch.load(str(checkpoint), weights_only=False, map_location=device)
  teacher.load_state_dict(loaded["actor_state_dict"], strict=True)
  teacher.eval()
  for p in teacher.parameters():
    p.requires_grad_(False)
  return teacher
