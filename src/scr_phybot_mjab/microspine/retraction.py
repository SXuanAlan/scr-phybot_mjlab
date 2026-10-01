"""Retraction of latched pads' contact proxies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.contact import resolve_sphere_geom_ids

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class LatchedContactRetraction:
  """Retract latched pads' contact proxies; restore them on release.

  Contacts model seating only; while latched, the bilateral anchor pins the
  pad and the proxy contacts are a redundant second load path that fights
  it (penetration-correction transients, near-zero-normal contact flicker).
  A latched pad's proxy geoms are moved ``latched_contact_retract`` meters
  INTO the pad along the face axis so they cannot touch the wall; released
  pads get the compiled offsets back the same step. Implemented through
  ``geom_pos`` because the kinematics kernel reads it per world every step
  (runtime ``pair_*`` writes are not honored by the collision pipeline). The
  first sync expands ``geom_pos`` per world through mjlab, which re-captures
  the CUDA graphs, so it runs lazily on the live simulation.
  """

  def __init__(self, env: ManagerBasedRlEnv, mcfg: MicrospineCfg, face_local_axis: torch.Tensor):
    self._env = env
    self._mcfg = mcfg
    self._face_local_axis = face_local_axis
    self._num_ee = len(mcfg.ee_body_names)
    self._geom_ids: torch.Tensor | None = None
    self._pos_default: torch.Tensor | None = None
    self._pos_retracted: torch.Tensor | None = None

  def _resolve(self) -> None:
    geom_lists = resolve_sphere_geom_ids(self._env.sim.mj_model, self._mcfg)
    self._geom_ids = torch.tensor(geom_lists, dtype=torch.long, device=self._env.device).reshape(-1)  # (H*S,)
    pos = wp.to_torch(self._env.sim.wp_model.geom_pos)
    self._pos_default = pos[0, self._geom_ids].clone()
    # Proxy child bodies share the pad's orientation, so the pad-frame face
    # axis retracts them straight away from the wall face.
    self._pos_retracted = self._pos_default - self._face_local_axis.unsqueeze(0) * self._mcfg.latched_contact_retract
    if pos.shape[0] != self._env.num_envs:
      self._env.sim.expand_model_fields(("geom_pos",))

  def sync(self, engaged: torch.Tensor) -> None:
    if self._geom_ids is None:
      self._resolve()
    num_envs = self._env.num_envs
    view = wp.to_torch(self._env.sim.wp_model.geom_pos)  # (E, ngeom, 3)
    proxies = self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols
    latched = engaged.unsqueeze(-1).expand(-1, self._num_ee, proxies).reshape(num_envs, -1, 1)
    defaults = self._pos_default.unsqueeze(0).expand(num_envs, -1, -1)
    retracted = self._pos_retracted.unsqueeze(0).expand(num_envs, -1, -1)
    view[:, self._geom_ids] = torch.where(latched, retracted, defaults)
