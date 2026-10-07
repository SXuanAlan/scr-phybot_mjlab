"""Signed wall gaps of the pad contact proxies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.contact import resolve_sphere_geom_ids
from scr_phybot_mjab.microspine.wall import wall_inward_normal

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ProxyWallGaps:
  """Signed distance from each contact proxy's closest surface to the wall, (E, H*S).

  Positive is a gap, negative a penetration, zero touching; proxies are in
  EE-major order. A box proxy's support along the wall normal depends on its
  full orientation, so boxes use their oriented extent and spheres their
  radius. The wall surface sits ``wall_half`` off the wall body's origin
  along the outward normal, which is read per step so wall tilts apply.
  """

  def __init__(self, mj_model: mujoco.MjModel, mcfg: MicrospineCfg, device: torch.device | str):
    ids = [gid for ee_ids in resolve_sphere_geom_ids(mj_model, mcfg) for gid in ee_ids]
    if not all(t in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE) for t in mj_model.geom_type[ids]):
      raise ValueError("Pad distance supports only box and sphere contact proxies")
    self.geom_ids = torch.tensor(ids, dtype=torch.long, device=device)
    self.sizes = torch.tensor(mj_model.geom_size[ids], dtype=torch.float32, device=device)
    self.is_box = torch.tensor(mj_model.geom_type[ids] == mujoco.mjtGeom.mjGEOM_BOX, device=device)
    self.wall_half = float(mj_model.geom_size[mj_model.geom(mcfg.wall_geom_name).id, 0])
    self._wall_entity_name = mcfg.wall_entity_name

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    wall_pos = env.scene[self._wall_entity_name].data.root_link_pos_w  # (E, 3)
    normal = -wall_inward_normal(env, self._wall_entity_name)  # outward, toward the robot
    geom_pos = env.sim.data.geom_xpos[:, self.geom_ids, :]  # (E, P, 3)
    signed = ((geom_pos - wall_pos.unsqueeze(1)) * normal.unsqueeze(1)).sum(dim=-1) - self.wall_half
    rotations = env.sim.data.geom_xmat[:, self.geom_ids].reshape(-1, len(self.geom_ids), 3, 3)
    local_normal = torch.einsum("egij,ei->egj", rotations, normal)
    extent = (local_normal.abs() * self.sizes.unsqueeze(0)).sum(-1)
    extent = torch.where(self.is_box.unsqueeze(0), extent, self.sizes[:, 0].unsqueeze(0))
    return signed - extent
