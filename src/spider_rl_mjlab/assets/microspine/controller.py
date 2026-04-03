"""Microspine controller — per-sphere adhesion via MuJoCo adhesion actuators.

Each hand has 8 contact spheres (2x4 grid), each in its own sub-body.
MuJoCo adhesion actuators are added to each sub-body via spec_fn.
The controller simply sets ctrl=1 for all adhesion actuators each step;
MuJoCo only applies adhesion force at actual contact points.

All operations are batched across envs — GPU friendly.
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
  """Add MuJoCo adhesion actuators for each contact sphere sub-body.

  Args:
    spec: MuJoCo spec to modify.
    ee_body_names: End-effector body names (e.g. ("left_hand", "right_hand")).
    adhesion_force: Total adhesion force [N] distributed across all spheres.
    robot_prefix: Robot entity name prefix in the spec.
  """
  num_spheres = len(ee_body_names) * _GEOMS_PER_HAND
  force_per_sphere = adhesion_force / num_spheres

  for bname in ee_body_names:
    side = bname.split("_")[0]
    for sname in _sphere_body_names(side):
      act = spec.add_actuator()
      act.name = f"{robot_prefix}/adh_{sname}"
      act.target = f"{robot_prefix}/{sname}"
      act.trntype = mujoco.mjtTrn.mjTRN_BODY
      act.set_to_adhesion(gain=force_per_sphere)
      act.ctrlrange[0] = 0.0
      act.ctrlrange[1] = 1.0
      act.ctrllimited = True
      act.forcelimited = True
      act.forcerange[0] = 0.0
      act.forcerange[1] = force_per_sphere


class MicrospineController:
  """Per-sphere adhesion via MuJoCo adhesion actuators (GPU-batched).

  Sets ctrl=1 for all adhesion actuators every step. MuJoCo's contact
  detection determines when adhesion force is actually applied — force
  only acts at real contact points, eliminating the "engage at distance"
  problem of the external wrench approach.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._mcfg: MicrospineCfg = cfg.params["microspine_cfg"]
    mj_model = env.sim.mj_model
    robot_prefix = cfg.params["asset_cfg"].name
    device = env.device

    # Collect adhesion actuator IDs.
    actuator_ids: list[int] = []
    for bname in self._mcfg.ee_body_names:
      side = bname.split("_")[0]
      for sname in _sphere_body_names(side):
        act_name = f"{robot_prefix}/adh_{sname}"
        actuator_ids.append(mj_model.actuator(act_name).id)

    self._actuator_ids = torch.tensor(actuator_ids, dtype=torch.long, device=device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    # MuJoCo only applies adhesion at actual contact points.
    env.sim.data.ctrl[:, self._actuator_ids] = self._mcfg.ctrl_value

  def reset(self, env_ids: torch.Tensor) -> None:
    pass
