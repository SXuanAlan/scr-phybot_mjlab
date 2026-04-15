"""Custom termination functions for the wall-climbing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def bad_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  max_displacement: float = 10.0,
  max_velocity: float = 50.0,
  max_ang_velocity: float = 200.0,
  max_joint_velocity: float = 200.0,
) -> torch.Tensor:
  """Terminate when simulation diverges (NaN/Inf or magnitude explodes).

  Covers every state tensor that feeds the actor observations so any
  divergence is caught before NaN propagates to the policy:
    - root pos (env-local), lin vel, ang vel, quat
    - joint pos, joint vel

  Uses env-local position to avoid false positives from env spacing.
  """
  asset: Entity = env.scene[asset_cfg.name]
  pos = asset.data.root_link_pos_w - env.scene.env_origins
  lin_vel = asset.data.root_link_lin_vel_w
  ang_vel = asset.data.root_link_ang_vel_w
  quat = asset.data.root_link_quat_w
  jpos = asset.data.joint_pos
  jvel = asset.data.joint_vel

  def _bad(t: torch.Tensor, limit: float | None = None) -> torch.Tensor:
    b = (~torch.isfinite(t)).any(dim=1)
    if limit is not None:
      b = b | (t.abs() > limit).any(dim=1)
    return b

  return (
    _bad(pos, max_displacement)
    | _bad(lin_vel, max_velocity)
    | _bad(ang_vel, max_ang_velocity)
    | _bad(quat)
    | _bad(jpos)
    | _bad(jvel, max_joint_velocity)
  )
