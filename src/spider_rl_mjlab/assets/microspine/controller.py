"""Microspine controller — per-sphere adhesion via external wrench.

Each hand has 8 contact spheres (2x4 grid), each in its own sub-body.
The controller independently tracks whether each sphere is engaged
with the wall using hysteresis on **surface**-to-wall distance (geom
origin distance minus geom radius). For each engaged sphere, a
wall-normal adhesion force is applied to its sub-body.

Oscillation is suppressed via contact solver damping (solref[1]) rather
than explicit velocity damping forces.

All operations are batched across envs and spheres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

from spider_rl_mjlab.assets.microspine.config import MicrospineCfg

_GEOM_ROWS = 2
_GEOM_COLS = 4
_GEOMS_PER_HAND = _GEOM_ROWS * _GEOM_COLS


def _sphere_body_names(side: str) -> list[str]:
  return [
    f"{side}_hand_c_{i}_{j}"
    for i in range(_GEOM_ROWS)
    for j in range(_GEOM_COLS)
  ]


def add_adhesion_actuators(
  spec: mujoco.MjSpec,
  ee_body_names: tuple[str, ...],
  adhesion_force: float,
  robot_prefix: str = "robot",
) -> None:
  """No-op kept for import compatibility."""
  pass


class MicrospineController:
  """Per-sphere adhesion via external wrench (GPU-batched)."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._mcfg: MicrospineCfg = cfg.params["microspine_cfg"]
    self._robot = env.scene[cfg.params["asset_cfg"].name]
    self._wall = env.scene[cfg.params["wall_cfg"].name]

    num_envs = env.num_envs
    device = env.device
    mj_model = env.sim.mj_model
    robot_prefix = cfg.params["asset_cfg"].name

    # Wall geometry.
    wn = torch.tensor(self._mcfg.wall_normal, dtype=torch.float32, device=device)
    self._force_dir = -wn / wn.norm()
    wn_abs = wn.abs()
    self._wall_axis = int(wn_abs.argmax().item())
    self._wall_sign = -1.0 if wn[self._wall_axis] < 0 else 1.0

    wall_prefix = cfg.params["wall_cfg"].name
    wall_geom_id = mj_model.geom(f"{wall_prefix}/wall_geom").id
    self._wall_half = float(mj_model.geom_size[wall_geom_id, self._wall_axis])

    self._force_per_sphere = self._mcfg.adhesion_force / _GEOMS_PER_HAND

    # Collect geom IDs, body IDs, and radii.
    geom_ids: list[int] = []
    sphere_body_ids: list[int] = []
    radii: list[float] = []

    for bname in self._mcfg.ee_body_names:
      side = bname.split("_")[0]
      for sname in _sphere_body_names(side):
        full_gname = f"{robot_prefix}/{sname}"
        gid = mj_model.geom(full_gname).id
        geom_ids.append(gid)

        geom_type = int(mj_model.geom_type[gid])
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
          r = float(mj_model.geom_size[gid, 0])
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
          r = float(mj_model.geom_size[gid, self._wall_axis])
        else:
          r = float(mj_model.geom_size[gid, 0])
        radii.append(r)

        body_ids_found, _ = self._robot.find_bodies([sname])
        sphere_body_ids.append(body_ids_found[0])

    self._num_spheres = len(geom_ids)
    self._geom_ids = torch.tensor(geom_ids, dtype=torch.long, device=device)
    self._sphere_body_ids = sphere_body_ids
    self._radii = torch.tensor(radii, dtype=torch.float32, device=device)
    self._force_dir_expanded = self._force_dir.unsqueeze(0).unsqueeze(0)

    self._engaged = torch.zeros(num_envs, self._num_spheres, dtype=torch.bool, device=device)

  def _surface_distances(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Surface-to-wall distance. Positive=gap, negative=penetration."""
    ax = self._wall_axis
    wall_face = self._wall.data.root_link_pos_w[:, ax] + self._wall_sign * self._wall_half
    geom_x = env.sim.data.geom_xpos[:, self._geom_ids, ax]
    origin_dist = (wall_face.unsqueeze(1) - geom_x) * self._wall_sign
    return origin_dist - self._radii.unsqueeze(0)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    surf_dist = self._surface_distances(env)

    self._engaged = (
      (self._engaged | (surf_dist < self._mcfg.engage_threshold))
      & (surf_dist <= self._mcfg.disengage_threshold)
    )

    force_scale = (1.0 - surf_dist / self._mcfg.disengage_threshold).clamp(0.0, 1.0)
    adhesion_mag = self._force_per_sphere * force_scale * self._engaged.float()
    forces = adhesion_mag.unsqueeze(-1) * self._force_dir_expanded
    torques = torch.zeros_like(forces)

    self._robot.write_external_wrench_to_sim(
      forces, torques, body_ids=self._sphere_body_ids
    )

  @property
  def engaged(self) -> torch.Tensor:
    return self._engaged

  def reset(self, env_ids: torch.Tensor) -> None:
    self._engaged[env_ids] = False
