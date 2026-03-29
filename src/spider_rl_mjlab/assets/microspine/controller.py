"""Microspine controller — skeleton for future approach implementation.

The actual adhesion/damping logic will be implemented here when tuning begins.
Test-phase forces are handled separately by mdp.phase_force.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg


class MicrospineController:
  """Step-mode event for microspine adhesion physics.

  Currently a no-op skeleton. The three tuning approaches (Path A/B/C)
  will add adhesion force logic, directional damping, and solver
  parameter overrides here.

  Params (from EventTermCfg.params):
    asset_cfg: SceneEntityCfg for the robot.
    microspine_cfg: MicrospineCfg instance.
    wall_cfg: SceneEntityCfg for the wall.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._microspine_cfg = cfg.params["microspine_cfg"]
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._wall_cfg: SceneEntityCfg = cfg.params["wall_cfg"]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    """Per-step callback. No-op until an approach is implemented."""
    pass

  def reset(self, env_ids: torch.Tensor) -> None:
    """Reset internal state on episode reset."""
    pass
