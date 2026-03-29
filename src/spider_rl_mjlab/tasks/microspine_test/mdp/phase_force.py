"""Two-phase test force event for the microspine adhesion test.

Phase 1 [0, duration):     push force (e.g. 20N +x) to engage hand against wall.
Phase 2 [duration, inf):   shear force (e.g. 300N -z) to test adhesion holding.

This is the test harness only — no adhesion logic. The MicrospineController
(in assets/microspine/controller.py) will later add adhesion forces on top.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class PhaseForce:
  """Step-mode event that applies a two-phase external force to a body.

  Params (from EventTermCfg.params):
    asset_cfg: SceneEntityCfg for the entity containing the target body.
    body_name: Name of the body to apply forces to.
    phase1_force: (Fx, Fy, Fz) applied during phase 1 [N].
    phase1_duration: Duration of phase 1 [s].
    phase2_force: (Fx, Fy, Fz) applied during phase 2 [N].
  """

  def __init__(self, cfg, env):
    from mjlab.managers.event_manager import EventTermCfg

    assert isinstance(cfg, EventTermCfg)
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._body_name: str = cfg.params["body_name"]
    self._phase1_force = cfg.params["phase1_force"]
    self._phase1_duration: float = cfg.params["phase1_duration"]
    self._phase2_force = cfg.params["phase2_force"]

    asset: Entity = env.scene[self._asset_cfg.name]
    body_ids, _ = asset.find_bodies([self._body_name])
    self._body_ids = body_ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    """Apply phase-dependent force every step."""
    num_envs = env.num_envs
    device = env.device

    elapsed = env.episode_length_buf.float() * env.step_dt  # (num_envs,)
    in_phase1 = (elapsed < self._phase1_duration).view(num_envs, 1, 1)

    f1 = torch.tensor(self._phase1_force, device=device, dtype=torch.float32).expand(num_envs, 1, 3)
    f2 = torch.tensor(self._phase2_force, device=device, dtype=torch.float32).expand(num_envs, 1, 3)

    forces = torch.where(in_phase1, f1, f2)
    torques = torch.zeros(num_envs, 1, 3, device=device)

    asset: Entity = env.scene[self._asset_cfg.name]
    asset.write_external_wrench_to_sim(forces, torques, body_ids=self._body_ids)

  def reset(self, env_ids: torch.Tensor) -> None:
    """No persistent state to reset."""
    pass
