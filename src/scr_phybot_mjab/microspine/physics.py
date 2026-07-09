"""Microspine physics: apply external adhesion forces from affordance state.

This module is deliberately only the physical executor. The contact-sensor
logic decides which end-effectors are engaged in ``MicrospineAffordance``; this
term turns those booleans into world-frame external forces on the sphere bodies.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import mujoco
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

from scr_phybot_mjab.microspine.config import MicrospineCfg


def _sphere_names(ee_body_name: str, rows: int, cols: int) -> list[str]:
  """Return sphere body/geom names for one end-effector."""
  return [
    f"{ee_body_name}_c_{i}_{j}"
    for i in range(rows)
    for j in range(cols)
  ]


def get_spheres_per_ee(cfg: MicrospineCfg) -> int:
  """Return number of contact spheres per end-effector."""
  return cfg.spheres_per_ee_rows * cfg.spheres_per_ee_cols


def get_sphere_geom_names(cfg: MicrospineCfg) -> tuple[str, ...]:
  """Return all microspine sphere geom names for a config."""
  return tuple(
    f"{ee}_c_{i}_{j}"
    for ee in cfg.ee_body_names
    for i in range(cfg.spheres_per_ee_rows)
    for j in range(cfg.spheres_per_ee_cols)
  )


def add_adhesion_actuators(
  spec: mujoco.MjSpec,
  ee_body_names: tuple[str, ...],
  adhesion_force: float,
  robot_prefix: str = "robot",
  rows: int = 2,
  cols: int = 4,
) -> None:
  """Deprecated compatibility shim.

  Adhesion is now applied as world-frame external force on sphere bodies by
  ``MicrospinePhysics``. This function intentionally adds no actuators.
  """
  del spec, ee_body_names, adhesion_force, robot_prefix, rows, cols


def add_contact_pairs_and_actuators(
  spec: mujoco.MjSpec,
  cfg: MicrospineCfg,
  robot_prefix: str = "robot",
) -> None:
  """Add wall contact pairs.

  Runtime adhesion is applied through external body forces rather than MuJoCo
  adhesion actuators, so this function does not create ``adh_*`` actuators.

  In ``attachment_mode == "constraint"`` the tangential lock and tension come
  from the per-EE anchor equalities, so the contacts revert to physical
  parameters: moderate friction, condim 3, critically damped solref. The
  legacy mode keeps the high-friction condim-6 pairs the external adhesion
  force relies on.
  """
  if cfg.attachment_mode == "constraint":
    friction = cfg.physical_pair_friction
    condim = cfg.physical_pair_condim
    solref = cfg.physical_pair_solref
  else:
    friction = cfg.wall_pair_friction
    condim = 6
    solref = cfg.wall_pair_solref
  for gname in get_sphere_geom_names(cfg):
    pair = spec.add_pair()
    pair.geomname1 = f"{robot_prefix}/{gname}"
    pair.geomname2 = cfg.wall_geom_name
    pair.condim = condim
    pair.friction = list(friction)
    pair.solref = list(solref)
    pair.solimp = list(cfg.wall_pair_solimp)
    pair.solreffriction = list(solref)
    pair.margin = cfg.wall_pair_margin
    pair.gap = cfg.wall_pair_gap


def make_physics_event_cfg(
  cfg: MicrospineCfg,
  affordance_name: str | None = None,
  entity_name: str = "robot",
) -> EventTermCfg:
  """Build the step event term that applies physical microspine adhesion."""
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  return EventTermCfg(
    func=MicrospinePhysics,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg(entity_name),
      "microspine_cfg": dataclasses.replace(
        cfg,
        affordance_name=affordance_name,
      ),
    },
  )


class MicrospinePhysics:
  """Apply per-EE microspine adhesion from an affordance command.

  The engage/disengage state machine intentionally lives in
  ``MicrospineAffordance``. This event term is only the physical executor:
  it mirrors the affordance's boolean state to external forces on the
  microspine sphere bodies.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._mcfg: MicrospineCfg = cfg.params["microspine_cfg"]
    self._affordance_name = self._mcfg.affordance_name
    if self._affordance_name is None:
      raise ValueError(
        "MicrospinePhysics requires MicrospineCfg.affordance_name. "
        "Physics-side engage/disengage thresholds have been removed."
      )

    self._env = env
    self._device = env.device
    self._num_envs = env.num_envs
    self._num_ee = len(self._mcfg.ee_body_names)
    self._debug = self._mcfg.debug

    robot_prefix = cfg.params["asset_cfg"].name
    self._asset = env.scene[robot_prefix]
    rows = self._mcfg.spheres_per_ee_rows
    cols = self._mcfg.spheres_per_ee_cols
    self._spheres_per_ee = rows * cols

    # The contact geometry and body names share the same suffix pattern. External
    # wrenches are written to bodies, not geoms, so keep an EE-major body list and
    # a reverse map from every sphere body back to its parent end-effector.
    sphere_body_names: list[str] = []
    sphere_to_ee: list[int] = []
    for ee_index, bname in enumerate(self._mcfg.ee_body_names):
      for sname in _sphere_names(bname, rows, cols):
        sphere_body_names.append(sname)
        sphere_to_ee.append(ee_index)

    sphere_body_ids, matched_names = self._asset.find_bodies(
      sphere_body_names, preserve_order=True
    )
    if len(sphere_body_ids) != len(sphere_body_names):
      raise ValueError(
        "MicrospinePhysics could not find all sphere bodies. "
        f"Expected {sphere_body_names}, found {matched_names}."
      )
    self._sphere_body_ids = sphere_body_ids
    self._sphere_to_ee = torch.tensor(
      sphere_to_ee, dtype=torch.long, device=self._device
    )
    # Adhesion is force-only. MuJoCo expects force and torque tensors together,
    # so keep a reusable zero-torque buffer with matching env/body dimensions.
    self._zero_sphere_torques = torch.zeros(
      self._num_envs,
      len(self._sphere_body_ids),
      3,
      dtype=torch.float32,
      device=self._device,
    )

    self.engaged = torch.zeros(
      self._num_envs, self._num_ee, dtype=torch.bool, device=self._device
    )
    self.disengaged_this_step = torch.zeros_like(self.engaged)
    self.current_disengage_time = torch.zeros(
      self._num_envs, self._num_ee, dtype=torch.float32, device=self._device
    )
    self.last_engage_command = torch.zeros_like(self.engaged)

  def _get_engage_command(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return latest per-EE affordance state as a bool tensor."""
    command = env.command_manager.get_command(self._affordance_name)
    # CommandTerm outputs are observation-friendly floats, but the physics layer
    # only needs a binary latch state. Treat >= 0.5 as engaged for robustness.
    engage_command = command if command.dtype == torch.bool else command >= 0.5
    engage_command = engage_command.to(self._device)

    if tuple(engage_command.shape) != tuple(self.engaged.shape):
      raise ValueError(
        f"Microspine affordance '{self._affordance_name}' shape "
        f"{tuple(engage_command.shape)} does not match engaged shape "
        f"{tuple(self.engaged.shape)}."
      )
    return engage_command

  def _wall_inward_normal(self) -> torch.Tensor:
    """Return wall inward normal in world frame, i.e. robot-to-wall direction."""
    wall = self._env.scene[self._mcfg.wall_entity_name]
    wquat = wall.data.root_link_quat_w
    # The wall asset convention uses local +x as the inward normal. Rotate that
    # axis into world frame so the adhesion force always points toward the wall,
    # even if the wall is randomized or tilted.
    ww, wx, wy, wz = wquat[:, 0], wquat[:, 1], wquat[:, 2], wquat[:, 3]
    inward_x = 1.0 - 2.0 * (wy * wy + wz * wz)
    inward_y = 2.0 * (wx * wy + ww * wz)
    inward_z = 2.0 * (wx * wz - ww * wy)
    return torch.stack([inward_x, inward_y, inward_z], dim=-1)

  def _external_adhesion_forces(self) -> torch.Tensor:
    """Broadcast per-EE engage state to per-sphere wall-normal forces."""
    # ``adhesion_force`` is configured per end-effector. Split it evenly across
    # that EE's sphere grid so changing rows/cols preserves total grip strength.
    force_per_sphere = self._mcfg.adhesion_force / self._spheres_per_ee
    # ``self._sphere_to_ee`` expands [env, ee] engaged flags to [env, sphere].
    # Disengaged EEs get zero force; engaged EEs get the configured force scale.
    sphere_scale = (
      self.engaged.float()[:, self._sphere_to_ee]
      * self._mcfg.ctrl_value
      * force_per_sphere
    )
    # Force direction is purely wall-normal. Tangential grip/friction still comes
    # from MuJoCo contacts; this term only pulls the sphere into the wall.
    wall_inward = self._wall_inward_normal()
    return sphere_scale.unsqueeze(-1) * wall_inward.unsqueeze(1)

  def _write_external_adhesion(
    self,
    env_ids: torch.Tensor | slice | None = None,
  ) -> torch.Tensor:
    """Write external adhesion force on each sphere body in world frame."""
    forces = self._external_adhesion_forces()
    torques = self._zero_sphere_torques
    # Reset calls pass a subset of env ids. Slice forces/torques the same way as
    # the write API so the body id list remains global while env rows are scoped.
    if env_ids is not None:
      forces = forces[env_ids]
      torques = torques[env_ids]
    self._asset.write_external_wrench_to_sim(
      forces,
      torques,
      env_ids=env_ids,
      body_ids=self._sphere_body_ids,
    )
    return forces

  def _update_disengage_time(
    self,
    env: ManagerBasedRlEnv,
    was_engaged: torch.Tensor,
  ) -> None:
    """Refresh one-step disengage flags and continuous-off timers."""
    self.disengaged_this_step = was_engaged & ~self.engaged
    self.current_disengage_time = torch.where(
      self.engaged,
      torch.zeros_like(self.current_disengage_time),
      self.current_disengage_time + env.step_dt,
    )

  def _log_transitions(
    self,
    env: ManagerBasedRlEnv,
    was_engaged: torch.Tensor,
  ) -> None:
    if not self._debug:
      return

    step = env.episode_length_buf[0].item()
    changed = self.engaged[0] != was_engaged[0]
    if not changed.any():
      return

    for ee_index, bname in enumerate(self._mcfg.ee_body_names):
      if not changed[ee_index].item():
        continue
      old = "ON" if was_engaged[0, ee_index].item() else "OFF"
      new = "ON" if self.engaged[0, ee_index].item() else "OFF"
      command = int(self.last_engage_command[0, ee_index].item())
      print(
        f"[Microspine] TRANSITION step={step} {bname} "
        f"{old}->{new} affordance={command}"
      )

  def _log_applied(self, env: ManagerBasedRlEnv, sphere_forces: torch.Tensor) -> None:
    if not self._debug:
      return

    step = env.episode_length_buf[0].item()
    if step % 50 != 0:
      return

    for ee_index, bname in enumerate(self._mcfg.ee_body_names):
      if not bool(self.engaged[0, ee_index].item()):
        continue
      start = ee_index * self._spheres_per_ee
      end = start + self._spheres_per_ee
      force = sphere_forces[0, start:end].norm(dim=-1).sum().item()
      print(
        f"[Microspine] APPLIED step={step} {bname} "
        f"engaged=1 external_adhesion={force:.1f}N"
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    was_engaged = self.engaged.clone()
    engage_command = self._get_engage_command(env)

    # All thresholding and hysteresis happen in the affordance command. This
    # executor mirrors that state directly, then writes the resulting forces.
    self.engaged[:] = engage_command
    self.last_engage_command = engage_command
    self._update_disengage_time(env, was_engaged)

    sphere_forces = self._write_external_adhesion()
    self._log_transitions(env, was_engaged)
    self._log_applied(env, sphere_forces)

  def reset(self, env_ids: torch.Tensor) -> None:
    engage_command = self._get_engage_command(self._env)
    self.engaged[env_ids] = engage_command[env_ids]
    self.disengaged_this_step[env_ids] = False
    self.current_disengage_time[env_ids] = 0.0
    self.last_engage_command[env_ids] = engage_command[env_ids]
    # Apply immediately after reset so a reset environment does not spend one
    # physics step with stale or missing adhesion forces.
    self._write_external_adhesion(env_ids)
