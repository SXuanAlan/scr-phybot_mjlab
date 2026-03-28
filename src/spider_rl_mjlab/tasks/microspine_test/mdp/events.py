"""Custom events for the microspine test task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def apply_timed_force(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  force: tuple[float, float, float] = (10.0, 0.0, 0.0),
  duration: float = 1.0,
  force_after: tuple[float, float, float] = (0.0, 0.0, 0.0),
  body_name: str | None = None,
) -> None:
  """Apply a two-phase external force to a body.

  Phase 1 [0, duration): apply ``force``.
  Phase 2 [duration, ∞): apply ``force_after``.

  Args:
    env: The RL environment.
    env_ids: Environment indices (unused — applied per-env based on timer).
    asset_cfg: Entity to apply force to.
    force: (Fx, Fy, Fz) applied during phase 1 [N].
    duration: Duration of phase 1 [s].
    force_after: (Fx, Fy, Fz) applied during phase 2 [N].
    body_name: Name of the body to apply force to. Defaults to root body.
  """
  asset: Entity = env.scene[asset_cfg.name]
  num_envs = env.num_envs

  if body_name is not None:
    body_ids, _ = asset.find_bodies([body_name])
  else:
    body_ids = [0]

  # episode_length_buf counts RL steps since last reset (incremented before events).
  # step_dt = decimation * physics_dt = 0.02 s.
  elapsed = env.episode_length_buf.float() * env.step_dt  # (num_envs,)
  in_phase1 = (elapsed < duration).view(num_envs, 1, 1)  # (num_envs, 1, 1)

  f1 = torch.tensor(force, device=env.device, dtype=torch.float32).expand(num_envs, 1, 3)
  f2 = torch.tensor(force_after, device=env.device, dtype=torch.float32).expand(num_envs, 1, 3)

  forces = torch.where(in_phase1, f1, f2)
  torques = torch.zeros(num_envs, 1, 3, device=env.device)
  asset.write_external_wrench_to_sim(forces, torques, body_ids=body_ids)
