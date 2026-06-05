"""Microspine physics: apply adhesion actuators from affordance state."""

from __future__ import annotations

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


def add_adhesion_actuators(
  spec: mujoco.MjSpec,
  ee_body_names: tuple[str, ...],
  adhesion_force: float,
  robot_prefix: str = "robot",
  rows: int = 2,
  cols: int = 4,
) -> None:
  """Add one MuJoCo adhesion actuator for each microspine contact sphere."""
  spheres_per_ee = rows * cols
  force_per_sphere = adhesion_force / spheres_per_ee

  for bname in ee_body_names:
    for sname in _sphere_names(bname, rows, cols):
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


class MicrospinePhysics:
  """Apply per-EE microspine adhesion from an affordance command.

  The engage/disengage state machine intentionally lives in
  ``MicrospineAffordance``. This event term is only the physical executor:
  it mirrors the affordance's boolean state to MuJoCo adhesion actuators.
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

    mj_model = env.sim.mj_model
    robot_prefix = cfg.params["asset_cfg"].name
    rows = self._mcfg.spheres_per_ee_rows
    cols = self._mcfg.spheres_per_ee_cols
    self._spheres_per_ee = rows * cols

    actuator_ids: list[int] = []
    sphere_to_ee: list[int] = []
    for ee_index, bname in enumerate(self._mcfg.ee_body_names):
      for sname in _sphere_names(bname, rows, cols):
        actuator_ids.append(mj_model.actuator(f"{robot_prefix}/adh_{sname}").id)
        sphere_to_ee.append(ee_index)

    self._actuator_ids = torch.tensor(
      actuator_ids, dtype=torch.long, device=self._device
    )
    self._sphere_to_ee = torch.tensor(
      sphere_to_ee, dtype=torch.long, device=self._device
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
    engage_command = command if command.dtype == torch.bool else command >= 0.5
    engage_command = engage_command.to(self._device)

    if tuple(engage_command.shape) != tuple(self.engaged.shape):
      raise ValueError(
        f"Microspine affordance '{self._affordance_name}' shape "
        f"{tuple(engage_command.shape)} does not match engaged shape "
        f"{tuple(self.engaged.shape)}."
      )
    return engage_command

  def _write_actuator_ctrl(self) -> torch.Tensor:
    """Broadcast per-EE engage state to each EE's sphere actuators."""
    actuator_ctrl = (
      self.engaged.float()[:, self._sphere_to_ee] * self._mcfg.ctrl_value
    )
    self._env.sim.data.ctrl[:, self._actuator_ids] = actuator_ctrl
    return actuator_ctrl

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

  def _log_applied(self, env: ManagerBasedRlEnv, actuator_ctrl: torch.Tensor) -> None:
    if not self._debug:
      return

    step = env.episode_length_buf[0].item()
    if step % 50 != 0:
      return

    force_per_sphere = self._mcfg.adhesion_force / self._spheres_per_ee
    for ee_index, bname in enumerate(self._mcfg.ee_body_names):
      if not bool(self.engaged[0, ee_index].item()):
        continue
      start = ee_index * self._spheres_per_ee
      end = start + self._spheres_per_ee
      force = actuator_ctrl[0, start:end].sum().item() * force_per_sphere
      print(
        f"[Microspine] APPLIED step={step} {bname} "
        f"engaged=1 adhesion={force:.1f}N"
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    was_engaged = self.engaged.clone()
    engage_command = self._get_engage_command(env)

    self.engaged[:] = engage_command
    self.last_engage_command = engage_command
    self._update_disengage_time(env, was_engaged)

    actuator_ctrl = self._write_actuator_ctrl()
    self._log_transitions(env, was_engaged)
    self._log_applied(env, actuator_ctrl)

  def reset(self, env_ids: torch.Tensor) -> None:
    engage_command = self._get_engage_command(self._env)
    self.engaged[env_ids] = engage_command[env_ids]
    self.disengaged_this_step[env_ids] = False
    self.current_disengage_time[env_ids] = 0.0
    self.last_engage_command[env_ids] = engage_command[env_ids]
    self._env.sim.data.ctrl[
      env_ids[:, None], self._actuator_ids[None, :]
    ] = self.engaged[env_ids].float()[:, self._sphere_to_ee] * self._mcfg.ctrl_value
