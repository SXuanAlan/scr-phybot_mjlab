"""Command generators for the wall-climbing task."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _sample_uniform(
  count: int, value_range: tuple[float, float], device: torch.device
) -> torch.Tensor:
  lo, hi = value_range
  if lo == hi:
    return torch.full((count,), float(lo), device=device)
  return torch.empty(count, device=device).uniform_(lo, hi)


class ClimbVelocityCommand(CommandTerm):
  """World-frame linear velocity command for climbing."""

  cfg: ClimbVelocityCommandCfg

  def __init__(self, cfg: ClimbVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)

    self.metrics["error_lin_vel_w"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vz"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_w

  def _update_metrics(self) -> None:
    max_command_time = max(self.cfg.resampling_time_range[1], self._env.step_dt)
    max_command_step = max_command_time / self._env.step_dt
    lin_vel_w = self.robot.data.root_link_lin_vel_w
    err = self.vel_command_w - lin_vel_w
    self.metrics["error_lin_vel_w"] += torch.norm(err, dim=-1) / max_command_step
    self.metrics["error_vz"] += torch.abs(err[:, 2]) / max_command_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    ranges = self.cfg.ranges
    self.vel_command_w[env_ids, 0] = _sample_uniform(
      count, ranges.lin_vel_x, self.device
    )
    self.vel_command_w[env_ids, 1] = _sample_uniform(
      count, ranges.lin_vel_y, self.device
    )
    self.vel_command_w[env_ids, 2] = _sample_uniform(
      count, ranges.lin_vel_z, self.device
    )

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class ClimbVelocityCommandCfg(CommandTermCfg):
  """Config for :class:`ClimbVelocityCommand`."""

  entity_name: str = "robot"

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float] = (0.0, 0.0)
    lin_vel_y: tuple[float, float] = (0.0, 0.0)
    lin_vel_z: tuple[float, float] = (0.08, 0.08)

  ranges: Ranges = field(default_factory=Ranges)

  def build(self, env: ManagerBasedRlEnv) -> ClimbVelocityCommand:
    return ClimbVelocityCommand(self, env)


class ClimbGaitPhase(IntEnum):
  """Contact-driven phase for the currently commanded end-effector."""

  RELEASE = 0
  SWING = 1
  SEAT = 2
  TRANSFER = 3


def resolve_climb_gait_phase(
  phase: torch.Tensor,
  target_engaged: torch.Tensor,
  target_contact: torch.Tensor,
) -> torch.Tensor:
  """Resolve contact-driven phase transitions without advancing timers."""
  result = phase.clone()
  is_release = phase == int(ClimbGaitPhase.RELEASE)
  is_swing = phase == int(ClimbGaitPhase.SWING)
  is_seat = phase == int(ClimbGaitPhase.SEAT)
  is_transfer = phase == int(ClimbGaitPhase.TRANSFER)

  result[is_release & ~target_engaged] = int(ClimbGaitPhase.SWING)
  result[is_swing & target_contact & ~target_engaged] = int(ClimbGaitPhase.SEAT)
  result[(is_swing | is_seat) & target_engaged] = int(
    ClimbGaitPhase.TRANSFER
  )
  result[is_seat & ~target_contact & ~target_engaged] = int(
    ClimbGaitPhase.SWING
  )
  result[is_transfer & ~target_engaged] = int(ClimbGaitPhase.SWING)
  return result


class ClimbGaitCommand(CommandTerm):
  """Select one limb at a time and advance only on physical grip events.

  The command never toggles a microspine. The policy must unload an engaged
  target until the limit surface releases it, move it upward, compress it into
  the wall, and wait for the normal affordance debounce and hold confirmation.
  """

  cfg: ClimbGaitCommandCfg

  def __init__(self, cfg: ClimbGaitCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self._num_ee = len(cfg.ee_body_names)
    if sorted(cfg.sequence) != list(range(self._num_ee)):
      raise ValueError(
        "Climb gait sequence must contain every EE index exactly once; "
        f"got {cfg.sequence}."
      )

    body_ids = []
    for body_name in cfg.ee_body_names:
      ids, _ = self.robot.find_bodies([body_name])
      body_ids.append(ids[0])
    self._ee_body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)
    self._sequence = torch.tensor(cfg.sequence, dtype=torch.long, device=self.device)
    self._env_ids = torch.arange(self.num_envs, device=self.device)

    self._command = torch.zeros(
      self.num_envs, self._num_ee + len(ClimbGaitPhase) + 3, device=self.device
    )
    self.target_ee_index = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.phase = torch.full(
      (self.num_envs,),
      int(ClimbGaitPhase.RELEASE),
      dtype=torch.long,
      device=self.device,
    )
    self.sequence_cursor = torch.zeros_like(self.target_ee_index)
    self.phase_steps = torch.zeros_like(self.target_ee_index)
    self.steps_in_cycle = torch.zeros_like(self.target_ee_index)
    self.productive_steps_in_cycle = torch.zeros_like(self.target_ee_index)
    self.target_start_z = torch.zeros(self.num_envs, device=self.device)
    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.ever_engaged = torch.zeros(
      self.num_envs, self._num_ee, dtype=torch.bool, device=self.device
    )
    self.support_shortfall = torch.zeros(self.num_envs, device=self.device)
    self._needs_target_init = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self.regrip_this_step = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.productive_regrip_this_step = torch.zeros_like(self.regrip_this_step)
    self.cycle_complete_this_step = torch.zeros_like(self.regrip_this_step)
    self.productive_cycle_this_step = torch.zeros_like(self.regrip_this_step)
    self.phase_timeout_this_step = torch.zeros_like(self.regrip_this_step)
    self.target_release_this_step = torch.zeros_like(self.regrip_this_step)
    self.target_contact_this_step = torch.zeros_like(self.regrip_this_step)
    self.unscheduled_release_count = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self.anchor_advance_this_step = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self.episode_productive_regrips = torch.zeros_like(self.target_ee_index)
    self.episode_success = torch.zeros_like(self.regrip_this_step)
    self.required_productive_regrips = 1
    self.randomize_start_limb = bool(cfg.randomize_start_limb)

    self.metrics["productive_regrips"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["completed_cycles"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["phase_timeouts"] = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def set_training_stage(
    self,
    *,
    required_productive_regrips: int,
    randomize_start_limb: bool,
  ) -> None:
    """Set the per-episode success contract for subsequent resets."""
    if required_productive_regrips < 1:
      raise ValueError("required_productive_regrips must be at least one.")
    self.required_productive_regrips = int(required_productive_regrips)
    self.randomize_start_limb = bool(randomize_start_limb)

  def _affordance(self):
    return self._env.command_manager.get_term(self.cfg.affordance_name)

  def _physics(self):
    return self._env.event_manager.get_term_cfg(self.cfg.physics_event_name).func

  def _selected(self, values: torch.Tensor) -> torch.Tensor:
    return values[self._env_ids, self.target_ee_index]

  def _selected_body_pos(self) -> torch.Tensor:
    body_pos = self.robot.data.body_link_pos_w[:, self._ee_body_ids]
    return body_pos[self._env_ids, self.target_ee_index]

  def _set_target_from_current(self, mask: torch.Tensor) -> None:
    current = self._selected_body_pos()
    physics = self._physics()
    selected_anchor_z = self._selected(physics.anchor_pos_w[..., 2])
    selected_physics_engaged = self._selected(physics.engaged)
    start_z = torch.where(selected_physics_engaged, selected_anchor_z, current[:, 2])
    self.target_start_z[mask] = start_z[mask]
    self.target_pos_w[mask] = current[mask]
    self.target_pos_w[mask, 2] = start_z[mask] + self.cfg.step_height

  def _write_command(self) -> None:
    self._command.zero_()
    self._command[self._env_ids, self.target_ee_index] = 1.0
    phase_offset = self._num_ee
    self._command[self._env_ids, phase_offset + self.phase] = 1.0
    delta = (self.target_pos_w - self._selected_body_pos()) / self.cfg.step_height
    delta[self._needs_target_init] = 0.0
    self._command[:, phase_offset + len(ClimbGaitPhase) :] = delta.clamp(-2.0, 2.0)

  def _update_metrics(self) -> None:
    self.metrics["productive_regrips"] += self.productive_regrip_this_step.float()
    self.metrics["completed_cycles"] += self.productive_cycle_this_step.float()
    self.metrics["phase_timeouts"] += self.phase_timeout_this_step.float()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    if self.randomize_start_limb:
      self.sequence_cursor[env_ids] = torch.randint(
        0, self._num_ee, (count,), device=self.device
      )
    else:
      self.sequence_cursor[env_ids] = 0
    self.target_ee_index[env_ids] = self._sequence[self.sequence_cursor[env_ids]]
    engaged = self._affordance().engaged
    self.ever_engaged[env_ids] = engaged[env_ids]
    target_engaged = engaged[env_ids, self.target_ee_index[env_ids]]
    self.phase[env_ids] = torch.where(
      target_engaged,
      torch.full_like(self.phase[env_ids], int(ClimbGaitPhase.RELEASE)),
      torch.full_like(self.phase[env_ids], int(ClimbGaitPhase.SWING)),
    )
    self.phase_steps[env_ids] = 0
    self.steps_in_cycle[env_ids] = 0
    self.productive_steps_in_cycle[env_ids] = 0
    self.target_start_z[env_ids] = 0.0
    self.target_pos_w[env_ids] = 0.0
    self._needs_target_init[env_ids] = True
    self.regrip_this_step[env_ids] = False
    self.productive_regrip_this_step[env_ids] = False
    self.cycle_complete_this_step[env_ids] = False
    self.productive_cycle_this_step[env_ids] = False
    self.phase_timeout_this_step[env_ids] = False
    self.target_release_this_step[env_ids] = False
    self.target_contact_this_step[env_ids] = False
    self.unscheduled_release_count[env_ids] = 0.0
    self.anchor_advance_this_step[env_ids] = 0.0
    self.support_shortfall[env_ids] = 0.0
    self.episode_productive_regrips[env_ids] = 0
    self.episode_success[env_ids] = False
    self._write_command()

  def _update_command(self) -> None:
    self.regrip_this_step.zero_()
    self.productive_regrip_this_step.zero_()
    self.cycle_complete_this_step.zero_()
    self.productive_cycle_this_step.zero_()
    self.phase_timeout_this_step.zero_()
    self.target_release_this_step.zero_()
    self.target_contact_this_step.zero_()
    self.unscheduled_release_count.zero_()
    self.anchor_advance_this_step.zero_()

    affordance = self._affordance()
    physics = self._physics()
    self.ever_engaged |= affordance.engaged
    target_mask = torch.nn.functional.one_hot(
      self.target_ee_index, num_classes=self._num_ee
    ).bool()
    self.unscheduled_release_count = (
      affordance.forced_release_this_step & ~target_mask
    ).float().sum(dim=1)

    initialize = self._needs_target_init.clone()
    self._set_target_from_current(initialize)
    target_engaged = self._selected(affordance.engaged)
    self.phase[initialize] = torch.where(
      target_engaged[initialize],
      torch.full_like(self.phase[initialize], int(ClimbGaitPhase.RELEASE)),
      torch.full_like(self.phase[initialize], int(ClimbGaitPhase.SWING)),
    )
    self.phase_steps[initialize] = 0
    self._needs_target_init[initialize] = False

    active = ~initialize
    self.phase_steps[active] += 1
    old_phase = self.phase.clone()
    target_engaged = self._selected(affordance.engaged)
    ever_count = self.ever_engaged.sum(dim=1)
    required_without_target = (ever_count - 1).clamp(min=2, max=3)
    required_supports = torch.where(
      target_engaged,
      ever_count.clamp(max=self._num_ee),
      required_without_target,
    )
    self.support_shortfall = (
      required_supports - affordance.engaged.sum(dim=1)
    ).clamp_min(0).float()
    target_contact = self._selected(affordance.contact_count) >= self.cfg.contact_min_count
    target_confirmed = self._selected(physics.engaged) & (
      self._selected(physics.engage_age) >= self.cfg.confirmation_steps
    )
    self.target_release_this_step = (
      active
      & (self.phase == int(ClimbGaitPhase.RELEASE))
      & ~target_engaged
    )
    self.target_contact_this_step = (
      active
      & (self.phase == int(ClimbGaitPhase.SWING))
      & target_contact
      & ~target_engaged
    )

    next_phase = resolve_climb_gait_phase(
      self.phase, target_engaged, target_contact
    )
    self.phase[active] = next_phase[active]

    changed = self.phase != old_phase
    self.phase_steps[changed] = 0
    timeout_steps = max(1, round(self.cfg.phase_timeout_s / self._env.step_dt))
    self.phase_timeout_this_step = active & (self.phase_steps == timeout_steps)

    confirmed_regrip = (
      active
      & (self.phase == int(ClimbGaitPhase.TRANSFER))
      & target_confirmed
      & (self.phase_steps >= self.cfg.transfer_hold_steps)
    )
    anchor_z = self._selected(physics.anchor_pos_w[..., 2])
    advance = (anchor_z - self.target_start_z) / self.cfg.step_height
    self.anchor_advance_this_step[confirmed_regrip] = advance[confirmed_regrip]
    productive = confirmed_regrip & (
      advance >= self.cfg.minimum_advance_fraction
    )
    self.regrip_this_step = confirmed_regrip
    self.productive_regrip_this_step = productive
    self.episode_productive_regrips += productive.long()

    self.steps_in_cycle[confirmed_regrip] += 1
    self.productive_steps_in_cycle[productive] += 1
    cycle_complete = confirmed_regrip & (self.steps_in_cycle >= self._num_ee)
    productive_cycle = cycle_complete & (
      self.productive_steps_in_cycle >= self._num_ee
    )
    self.cycle_complete_this_step = cycle_complete
    self.productive_cycle_this_step = productive_cycle
    self.steps_in_cycle[cycle_complete] = 0
    self.productive_steps_in_cycle[cycle_complete] = 0

    self.sequence_cursor[confirmed_regrip] = (
      self.sequence_cursor[confirmed_regrip] + 1
    ) % self._num_ee
    self.target_ee_index[confirmed_regrip] = self._sequence[
      self.sequence_cursor[confirmed_regrip]
    ]
    self._set_target_from_current(confirmed_regrip)
    next_engaged = self._selected(affordance.engaged)
    self.phase[confirmed_regrip] = torch.where(
      next_engaged[confirmed_regrip],
      torch.full_like(
        self.phase[confirmed_regrip], int(ClimbGaitPhase.RELEASE)
      ),
      torch.full_like(self.phase[confirmed_regrip], int(ClimbGaitPhase.SWING)),
    )
    self.phase_steps[confirmed_regrip] = 0

    self.episode_success |= (
      self.episode_productive_regrips >= self.required_productive_regrips
    )
    self._write_command()


@dataclass(kw_only=True)
class ClimbGaitCommandCfg(CommandTermCfg):
  """Configuration for :class:`ClimbGaitCommand`."""

  entity_name: str = "robot"
  affordance_name: str = "microspine_affordance"
  physics_event_name: str = "microspine_grip"
  ee_body_names: tuple[str, ...]
  sequence: tuple[int, ...] = (2, 1, 3, 0)
  step_height: float = 0.06
  minimum_advance_fraction: float = 0.5
  contact_min_count: int = 2
  confirmation_steps: int = 16
  transfer_hold_steps: int = 5
  phase_timeout_s: float = 4.0
  randomize_start_limb: bool = True
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> ClimbGaitCommand:
    return ClimbGaitCommand(self, env)
