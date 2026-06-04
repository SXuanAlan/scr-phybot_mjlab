"""Action terms for policy-controlled microspine adhesion intent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MicrospineAdhesionActionCfg(ActionTermCfg):
  """Policy action that requests per-EE microspine adhesion on/off.

  The term does not write to MuJoCo directly. It stores the latest policy
  intent, and ``MicrospineController`` combines that intent with contact and
  geometry gates before changing the physical latch state.
  """

  ee_body_names: tuple[str, ...]
  """End-effector body names, in the same order as ``MicrospineCfg``."""

  activation_bias: float = 0.5
  """Bias added to raw logits before sigmoid. A zero policy action requests ON."""

  activation_threshold: float = 0.5
  """Sigmoid probability threshold for an ON request."""

  def build(self, env: ManagerBasedRlEnv) -> MicrospineAdhesionAction:
    return MicrospineAdhesionAction(self, env)


class MicrospineAdhesionAction(ActionTerm):
  """Stores per-EE adhesion intent from the policy."""

  cfg: MicrospineAdhesionActionCfg

  def __init__(self, cfg: MicrospineAdhesionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._action_dim = len(cfg.ee_body_names)
    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
    self._probability = torch.sigmoid(self._raw_actions + cfg.activation_bias)
    self._request_on = self._probability >= cfg.activation_threshold

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def probability(self) -> torch.Tensor:
    """Latest per-EE request probability in [0, 1]."""
    return self._probability

  @property
  def request_on(self) -> torch.Tensor:
    """Latest per-EE binary adhesion request."""
    return self._request_on

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    self._probability = torch.sigmoid(self._raw_actions + self.cfg.activation_bias)
    self._request_on = self._probability >= self.cfg.activation_threshold

  def apply_actions(self) -> None:
    # Physical adhesion is applied by MicrospineController after contact gates.
    return

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._probability[env_ids] = torch.sigmoid(
      self._raw_actions[env_ids] + self.cfg.activation_bias
    )
    self._request_on[env_ids] = (
      self._probability[env_ids] >= self.cfg.activation_threshold
    )
