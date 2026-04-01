"""Microspine controller — adhesion physics via normal pull-in force.

When an end-effector contacts the wall, the controller applies an adhesion
force in the wall-normal direction to keep the EE pressed against the surface.
This enables friction from the contact model (condim=6 + impratio=10) to
resist tangential shear loads.

Supports two modes:
  - External wrench (default): applies force via xfrc_applied.
  - Adhesion actuator (use_actuator=True): uses MuJoCo's native adhesion
    actuator which works through the constraint solver at contact points.

Supports multiple EE bodies (e.g. both hands for climbing).
Each EE body is tracked independently for engagement state.
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


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate vector v by quaternion q (wxyz convention).

  Args:
    q: (*, 4) quaternions in (w, x, y, z) order.
    v: (*, 3) or (3,) vectors.

  Returns:
    Rotated vectors (*, 3).
  """
  q_w = q[..., 0:1]
  q_vec = q[..., 1:4]
  t = 2.0 * torch.linalg.cross(q_vec, v)
  return v + q_w * t + torch.linalg.cross(q_vec, t)


def add_adhesion_actuators(
  spec: mujoco.MjSpec,
  ee_body_names: tuple[str, ...],
  adhesion_force: float,
  robot_prefix: str = "robot",
) -> None:
  """Add MuJoCo adhesion actuators for each EE body to the spec.

  Called from spec_fn before model compilation. Each actuator targets
  a body and creates constraint-level attractive forces at all active
  contact points of that body's geoms.

  Args:
    spec: The MjSpec to add actuators to.
    ee_body_names: Body names to add adhesion actuators for.
    adhesion_force: Gain (max force in N) for each adhesion actuator.
    robot_prefix: Namespace prefix for the robot in the spec.
  """
  for bname in ee_body_names:
    full_name = f"{robot_prefix}/{bname}"
    act = spec.add_actuator()
    act.name = f"{robot_prefix}/adhesion_{bname}"
    act.target = full_name
    act.trntype = mujoco.mjtTrn.mjTRN_BODY
    act.set_to_adhesion(gain=adhesion_force)
    act.ctrllimited = True
    act.ctrlrange[:] = [0, 1]


class MicrospineController:
  """Step-mode event for microspine adhesion physics.

  Applies a wall-normal adhesion force to each end-effector body independently
  when its spines are engaged. Uses hysteresis per body to prevent rapid cycling.

  Distance is computed by transforming the local contact offset into world frame
  using the body's orientation, then measuring the gap to the wall face.

  Params (from EventTermCfg.params):
    asset_cfg: SceneEntityCfg for the robot.
    microspine_cfg: MicrospineCfg instance.
    wall_cfg: SceneEntityCfg for the wall.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._mcfg: MicrospineCfg = cfg.params["microspine_cfg"]
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._wall_cfg: SceneEntityCfg = cfg.params["wall_cfg"]

    self._robot = env.scene[self._asset_cfg.name]
    self._wall = env.scene[self._wall_cfg.name]

    self._ee_body_ids = []
    for bname in self._mcfg.ee_body_names:
      ids, _ = self._robot.find_bodies([bname])
      self._ee_body_ids.extend(ids)
    self._num_ee = len(self._ee_body_ids)

    num_envs = env.num_envs
    device = env.device

    self._engaged = torch.zeros(num_envs, self._num_ee, dtype=torch.bool, device=device)

    wn = torch.tensor(self._mcfg.wall_normal, dtype=torch.float32, device=device)
    self._adhesion_dir = -wn / wn.norm()  # toward wall

    wn_abs = wn.abs()
    self._wall_axis = int(wn_abs.argmax().item())
    self._wall_sign = -1.0 if wn[self._wall_axis] < 0 else 1.0

    self._local_offset = torch.tensor(
      [self._mcfg.contact_offset, 0.0, 0.0], dtype=torch.float32, device=device
    )

    self._use_actuator = self._mcfg.use_actuator
    if self._use_actuator:
      # Find adhesion actuator ctrl indices in the compiled model.
      mj_model = env.sim.mj_model
      self._adhesion_ctrl_ids = []
      robot_prefix = self._asset_cfg.name
      for bname in self._mcfg.ee_body_names:
        act_name = f"{robot_prefix}/adhesion_{bname}"
        act_id = mj_model.actuator(act_name).id
        self._adhesion_ctrl_ids.append(act_id)
      self._adhesion_ctrl_ids_tensor = torch.tensor(
        self._adhesion_ctrl_ids, dtype=torch.long, device=device
      )

  def _get_ee_wall_distances(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Compute signed distance from each EE contact surface to the wall face.

    Returns (num_envs, num_ee). Positive = gap, negative = penetration.
    """
    wall_pos = self._wall.data.root_link_pos_w
    ax = self._wall_axis
    wall_face = wall_pos[:, ax] + self._wall_sign * 0.5

    distances = []
    for body_id in self._ee_body_ids:
      ee_pos = self._robot.data.body_link_pos_w[:, body_id, :]
      ee_quat = self._robot.data.body_link_quat_w[:, body_id, :]
      world_offset = _quat_rotate(ee_quat, self._local_offset.unsqueeze(0))
      contact_surface = ee_pos + world_offset
      dist = (wall_face - contact_surface[:, ax]) * self._wall_sign
      distances.append(dist)

    return torch.stack(distances, dim=1)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    """Per-step: apply adhesion force per EE body when engaged."""
    num_envs = env.num_envs
    device = env.device

    distances = self._get_ee_wall_distances(env)

    engage_mask = distances < self._mcfg.engage_threshold
    disengage_mask = distances > self._mcfg.disengage_threshold
    self._engaged = (self._engaged | engage_mask) & ~disengage_mask

    force_scale = (1.0 - distances / self._mcfg.disengage_threshold).clamp(0.0, 1.0)
    ctrl_signal = force_scale * self._engaged.float()  # 0..1

    if self._use_actuator:
      # Set adhesion actuator ctrl values directly.
      # ctrl shape: (num_envs, num_ee)
      env.sim.data.ctrl[:, self._adhesion_ctrl_ids_tensor] = ctrl_signal
    else:
      # Legacy: apply external wrench.
      force_mag = self._mcfg.adhesion_force * ctrl_signal
      adhesion_force = force_mag.unsqueeze(-1) * self._adhesion_dir.unsqueeze(0).unsqueeze(0)
      torques = torch.zeros(num_envs, self._num_ee, 3, device=device)
      self._robot.write_external_wrench_to_sim(
        adhesion_force, torques, body_ids=self._ee_body_ids
      )

  def reset(self, env_ids: torch.Tensor) -> None:
    """Reset engagement state on episode reset."""
    self._engaged[env_ids] = False
