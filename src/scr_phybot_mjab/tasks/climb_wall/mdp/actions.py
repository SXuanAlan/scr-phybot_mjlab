"""Climb-specific action terms.

The stock MJLab joint-position action applies only the affine action mapping.
Climbing contacts can turn an out-of-range PD target into a useful joint-stop
exploit, so this task adds a final target-space safety bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


def clamp_to_joint_limits(
  target: torch.Tensor,
  limits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Clamp position targets and return a per-joint saturation mask."""
  if limits.shape != (*target.shape, 2):
    raise ValueError(
      f"Joint-limit shape {tuple(limits.shape)} does not match target shape "
      f"{tuple(target.shape)}."
    )
  lower = limits[..., 0]
  upper = limits[..., 1]
  clamped = torch.clamp(target, min=lower, max=upper)
  return clamped, clamped != target


@dataclass(kw_only=True)
class SoftJointPositionActionCfg(JointPositionActionCfg):
  """Joint-position action whose final PD target stays inside soft limits."""

  def build(self, env):
    return SoftJointPositionAction(self, env)


class SoftJointPositionAction(JointPositionAction):
  """Apply a joint target after clamping it to the entity's soft limits."""

  cfg: SoftJointPositionActionCfg

  def __init__(self, cfg: SoftJointPositionActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    shape = (self.num_envs, self.action_dim)
    self.applied_target = torch.zeros(shape, device=self.device)
    self.target_clamped = torch.zeros(shape, dtype=torch.bool, device=self.device)

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    requested_target = self._processed_actions - encoder_bias
    soft_limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    target, clamped = clamp_to_joint_limits(requested_target, soft_limits)
    self.applied_target.copy_(target)
    self.target_clamped.copy_(clamped)
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids=env_ids)
    self.applied_target[env_ids] = 0.0
    self.target_clamped[env_ids] = False
