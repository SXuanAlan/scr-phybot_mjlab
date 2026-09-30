"""Reference State Initialization (RSI) from the AMP motion dataset.

The stock ``amp-rsl-rl`` runner never calls ``AMPLoader.get_state_for_reset`` -- it
lets the env reset randomly. This command term wires that capability in: on every
episode reset it samples a full state from the AMP dataset and teleports the robot
there, so episodes start in valid reference poses (standing / mid-stride / turning)
instead of a scaled default pose.

AMP's style reward is distribution-based (a discriminator over state transitions),
so RSI here is a coverage/exploration aid rather than a requirement -- unlike a
DeepMimic-style per-frame tracking task, the reward does not collapse without it.

It is a ``CommandTerm`` only to hook the command manager's per-reset resample (the
same mechanism the tracking / squat tasks use for RSI); it produces no observed
command. ``get_state_for_reset`` returns orientation + joint pos/vel + base
velocities but no root position, so the base is placed at the env origin at a
fixed ``reset_height`` (exact for the dominant standing clip; walking samples drop
a few cm and settle).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from amp_rsl_rl.utils import AMPLoader, VelocityRepresentation
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class AmpReferenceResetCommand(CommandTerm):
  cfg: AmpReferenceResetCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: AmpReferenceResetCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.joint_ids = torch.tensor(
      self.robot.find_joints(cfg.joint_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.amp_data = AMPLoader(
      device=str(self.device),
      dataset_path_root=Path(cfg.amp_data_path),
      datasets=cfg.datasets,
      simulation_dt=self._env.step_dt,
      slow_down_factor=cfg.slow_down_factor,
      expected_joint_names=list(cfg.joint_names),
      velocity_representation=VelocityRepresentation(cfg.velocity_representation),
    )
    self._vel_is_body = (
      VelocityRepresentation(cfg.velocity_representation) == VelocityRepresentation.BODY_FIXED_REPRESENTATION
    )
    # Not observed; exists only to trigger _resample_command on reset.
    self._null_command = torch.zeros(self.num_envs, 1, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._null_command

  def _update_metrics(self) -> None:
    pass

  def _update_command(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    quat, joint_pos, joint_vel, base_lin_vel, base_ang_vel = self.amp_data.get_state_for_reset(len(env_ids))

    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=self.joint_ids)

    # get_state_for_reset omits root position: place the base at the env origin at
    # a fixed height (exact for standing, close for walking samples).
    root_pos = self._env.scene.env_origins[env_ids].clone()
    root_pos[:, 2] = self.cfg.reset_height
    if self._vel_is_body:
      base_lin_vel = quat_apply(quat, base_lin_vel)
      base_ang_vel = quat_apply(quat, base_ang_vel)
    root_state = torch.cat([root_pos, quat, base_lin_vel, base_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)


@dataclass(kw_only=True)
class AmpReferenceResetCommandCfg(CommandTermCfg):
  entity_name: str
  amp_data_path: str
  datasets: dict[str, float]
  joint_names: tuple[str, ...]
  slow_down_factor: int = 1
  velocity_representation: str = "body_fixed"
  reset_height: float = 0.726

  def build(self, env: ManagerBasedRlEnv) -> AmpReferenceResetCommand:
    return AmpReferenceResetCommand(self, env)
