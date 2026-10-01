"""Microspine grip affordance: the per-pad latch state machine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm

from scr_phybot_mjab.microspine import latch_state
from scr_phybot_mjab.microspine.config import MicrospineAffordanceCfg
from scr_phybot_mjab.microspine.contact_features import PadContactReader
from scr_phybot_mjab.microspine.empirical_limit_surface import DEFAULT_TABLE
from scr_phybot_mjab.microspine.engage_gate import SpineField, engage_frame, face_wall_cos, to_engage_frame
from scr_phybot_mjab.microspine.hold_judge import HoldJudge, UnitVerdict
from scr_phybot_mjab.microspine.pad_geometry import matching_pad_geometry
from scr_phybot_mjab.microspine.release_reason import ReleaseReason
from scr_phybot_mjab.microspine.retraction import LatchedContactRetraction
from scr_phybot_mjab.microspine.surfaces import MicrospineSurfaces, rotate
from scr_phybot_mjab.microspine.wall import wall_inward_normal

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class MicrospineAffordance(CommandTerm):
  """Stateful world affordance for per-EE microspine grip.

  The affordance is intentionally outside the robot action space: it reads
  the world's contact features and owns the per-EE latch state machine, so
  policy outputs cannot command grip directly. The constraint executor
  (MicrospineConstraint) consumes the latch bits and toggles the per-EE
  equality anchors.

  Engagement requires unit preload and seating. Once latched, release is
  governed by the empirical hold judgement alone; losing contact does not
  silently drop the anchor. Each step ``_update_command`` reads the
  contacts, evaluates the engage gate, debounces it, judges the holds and
  commits the transitions.
  """

  cfg: MicrospineAffordanceCfg

  def __init__(self, cfg: MicrospineAffordanceCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._mcfg = cfg.microspine_cfg
    self._ls_cfg = self._mcfg.limit_surface
    self._num_ee = len(self._mcfg.ee_body_names)
    self._proxies_per_ee = self._mcfg.spheres_per_ee_rows * self._mcfg.spheres_per_ee_cols
    initial_mask = self._mcfg.resolve_initial_engaged_mask()
    if not self._mcfg.grip_enabled:
      initial_mask = (False,) * self._num_ee
    self._initial_engaged = torch.tensor(initial_mask, dtype=torch.bool, device=self.device)

    self._asset = env.scene[cfg.asset_name]
    self._surfaces = MicrospineSurfaces(env, self._mcfg) if self._mcfg.surfaces else None
    self._ee_body_ids = [self._asset.find_bodies([name])[0][0] for name in self._mcfg.ee_body_names]
    if tuple(self._ls_cfg.ee_rake_local_axis) != (0.0, 0.0, -1.0) or tuple(self._mcfg.face_local_axis) != (
      0.0,
      -1.0,
      0.0,
    ):
      raise ValueError("The empirical unit table assumes the pad rake axis -z and face axis -y")
    # The face axis points into the wall; the rake axis is the direction the
    # spines load along. Both are unit vectors (checked above).
    self._face_local_axis = torch.tensor(self._mcfg.face_local_axis, dtype=torch.float32, device=self.device)
    self._rake_local_axis = torch.tensor(self._ls_cfg.ee_rake_local_axis, dtype=torch.float32, device=self.device)
    geometry = matching_pad_geometry(
      env.sim.mj_model,
      self._mcfg.ee_body_names,
      cfg.asset_name,
      rake_local_axis=self._ls_cfg.ee_rake_local_axis,
      face_local_axis=self._mcfg.face_local_axis,
      unit_layout=self._ls_cfg.unit_layout,
    )
    if geometry.rows < 2 or geometry.cols < 2:
      raise ValueError(f"Moment billing needs a >=2x2 unit grid; got {geometry.rows}x{geometry.cols}.")
    if not np.isfinite(self._mcfg.engage_unit_preload_n) or self._mcfg.engage_unit_preload_n < 0:
      raise ValueError("engage_unit_preload_n must be finite and nonnegative")
    self.hold_judge = HoldJudge(geometry, self._ls_cfg.empirical_table_path or DEFAULT_TABLE, self.device)
    self.unit_site_names: tuple[str, ...] = geometry.unit_names
    self.spine_field = SpineField(
      env, self._asset, self._ee_body_ids, self._mcfg, cfg.asset_name, self._surfaces, self._live_wall_inward_normal
    )
    self._contacts = PadContactReader(env, cfg.contact_sensor_name, self._mcfg, self._proxy_inward_normal)
    self._retraction = LatchedContactRetraction(env, self._mcfg, self._face_local_axis)
    self._executor = None
    self._init_state((self.num_envs, self._num_ee), self.hold_judge.unit_count)
    self._sample_limit_params(torch.arange(self.num_envs, device=self.device))

  def _init_state(self, shape: tuple[int, int], units: int) -> None:
    """Allocate the latch state and every readout, (E, H) unless noted."""
    self._engaged = self._initial_engaged.expand(*shape).clone()
    # Debounce, re-engage cooldown, hold-violation and engage-age counters.
    self._engage_count = torch.zeros(shape, dtype=torch.long, device=self.device)
    self._cooldown_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
    self._violation_count = torch.zeros(shape, dtype=torch.long, device=self.device)
    self._steps_since_engage = torch.zeros(shape, dtype=torch.long, device=self.device)

    # Hold judgement readouts. Loads are in the engage frame (t̂, l̂, n̂_out).
    self.anchor_load_tnl = torch.zeros(*shape, 3, device=self.device)
    self.anchor_moment_tnl = torch.zeros_like(self.anchor_load_tnl)
    self.anchor_effective_load_tnl = torch.zeros_like(self.anchor_load_tnl)
    self.hold_margin = torch.ones(shape, device=self.device)
    self.hold_margin_raw = torch.ones(shape, device=self.device)
    """The hold margin with moments excluded; diagnostics only."""

    # One-step events. ``engaged_this_step`` is a debounced OFF -> ON
    # transition; unlike the constraint executor's rising edge it excludes
    # reset-time placement. ``release_reason`` holds the ReleaseReason on the
    # release step and is cleared the step after.
    self.forced_release_this_step = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self.engaged_this_step = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self.release_reason = torch.zeros(shape, dtype=torch.int8, device=self.device)
    # Seed provenance: ``_seeded`` is True while a reset-seeded anchor (see
    # ``seed_engaged``) is the one holding, and is cleared when it releases;
    # a latch the policy makes itself is never seeded. ``released_was_seeded``
    # flags, for one step, the release of a seeded anchor.
    self._seeded = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self.released_was_seeded = torch.zeros(shape, dtype=torch.bool, device=self.device)
    # Reseat hold: an external release (``force_release``) stays pending until
    # that EE re-latches, the hold times out or the env resets; the motion
    # command freezes its clock while any EE is pending (``reseat_hold``).
    self.dr_release_pending = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self._reseat_hold_steps = torch.zeros(shape[0], dtype=torch.long, device=self.device)

    # Contact readouts, live whether or not a pad is latched.
    self.face_cos = torch.zeros(shape, device=self.device)
    self.contact_count = torch.zeros(shape, device=self.device)
    self.pad_press_force = torch.zeros(shape, device=self.device)
    """Total proxy compression against the wall per EE (N)."""
    self.spine_fraction = torch.zeros(shape, device=self.device)
    """Engaged fraction of the virtual spine field: the seating judgement."""
    self.unit_preload_n = torch.zeros((*shape, units), device=self.device)
    self.pad_contact_wrench_tln = torch.zeros((*shape, 6), device=self.device)

    # Persistent per-unit draws from the bench distributions, and the unit
    # readouts of the hold judgement.
    self.unit_strength_quantile = torch.zeros((*shape, units), device=self.device)
    self.unit_loading_quantile = torch.zeros_like(self.unit_strength_quantile)
    self.unit_load_tln = torch.zeros((*shape, units, 3), device=self.device)
    self.unit_normal_capacity = torch.zeros_like(self.unit_strength_quantile)
    self.unit_empirical_supported = torch.ones((*shape, units), dtype=torch.bool, device=self.device)
    self.unit_empirical_early_failed = torch.zeros_like(self.unit_empirical_supported)
    self.unit_empirical_tail_capped = torch.zeros_like(self.unit_empirical_supported)

  # ------------------------------------------------------------------ API

  @property
  def command(self) -> torch.Tensor:
    return self._engaged.float()

  @property
  def engaged(self) -> torch.Tensor:
    return self._engaged

  @property
  def seeded(self) -> torch.Tensor:
    """(E, EE) bool: this EE's current hold is a reset-seeded anchor."""
    return self._seeded

  @property
  def cooldown_steps(self) -> torch.Tensor:
    """Remaining per-EE re-engagement cooldown in policy steps."""
    return self._cooldown_steps

  @property
  def reseat_hold(self) -> torch.Tensor:
    """(E,) bool: clock-freeze request while an external release awaits reseat."""
    return self.dr_release_pending.any(dim=-1)

  def set_initial_engaged_mask(self, mask: tuple[bool, ...]) -> None:
    """Set the latch mask used by subsequent episode resets."""
    if len(mask) != self._num_ee:
      raise ValueError(f"Initial mask has {len(mask)} entries; expected {self._num_ee}.")
    self._initial_engaged.copy_(torch.tensor(mask, dtype=torch.bool, device=self.device))

  def set_initial_engaged_ee_names(self, names: tuple[str, ...]) -> None:
    """Select reset latches by configured EE name."""
    selected = set(names)
    unknown = selected.difference(self._mcfg.ee_body_names)
    if unknown:
      raise ValueError("Unknown initial microspine EEs: " + ", ".join(sorted(unknown)))
    self.set_initial_engaged_mask(tuple(name in selected for name in self._mcfg.ee_body_names))

  def seed_engaged(self, env_ids: torch.Tensor, engaged: torch.Tensor) -> None:
    """Overwrite per-env latch state during a reset (motion-clip RSI).

    Must run AFTER this term's own ``_resample_command`` in the same reset
    pass (register the caller later in the command dict), or the seeded mask
    is clobbered by the global initial mask. The latch bookkeeping is zeroed
    here as well so correctness does not depend on that reset having run.
    """
    if not self._mcfg.grip_enabled:
      return
    if engaged.shape[-1] != self._num_ee:
      raise ValueError(f"Seed mask has {engaged.shape[-1]} EEs; expected {self._num_ee}.")
    self._engaged[env_ids] = engaged.to(device=self.device, dtype=torch.bool)
    self._seeded[env_ids] = self._engaged[env_ids]
    self.released_was_seeded[env_ids] = False
    self.dr_release_pending[env_ids] = False
    self._reseat_hold_steps[env_ids] = 0
    self._engage_count[env_ids] = 0
    self._cooldown_steps[env_ids] = 0
    self._violation_count[env_ids] = 0
    self._steps_since_engage[env_ids] = 0
    self._sync_retraction()

  def force_release(
    self,
    env_ids: torch.Tensor,
    ee_mask: torch.Tensor,
    reason: int = ReleaseReason.EXTERNAL,
    report: bool = True,
  ) -> None:
    """Release latched EEs outside the hold judgement.

    Serves the DR random-release event and scripted diagnostics. Callers run
    after this step's ``_update_command``, so ``forced_release_this_step`` is
    OR-ed (a same-step hold release must survive) while the latch bookkeeping
    mirrors a real release: the cooldown starts and the pad's contact
    proxies are restored so it can press the wall again. The equality
    constraint follows the cleared latch bit on the next step's constraint
    event, the same one-step lag as every latch transition.

    ``report=False`` skips the ``forced_release_this_step``/``release_reason``
    writes: a scheduled scripted release is not a fault and must not pollute
    forced-release statistics or the reason-billed penalty.
    """
    if not self._mcfg.grip_enabled:
      return
    if ee_mask.shape[-1] != self._num_ee:
      raise ValueError(f"Release mask has {ee_mask.shape[-1]} EEs; expected {self._num_ee}.")
    mask = ee_mask.to(device=self.device, dtype=torch.bool)
    mask = mask & self._engaged[env_ids]
    if not bool(mask.any()):
      return
    if report:
      # A scripted no-fault release never counts as seed demolition either.
      self.released_was_seeded[env_ids] |= mask & self._seeded[env_ids]
    self._seeded[env_ids] = self._seeded[env_ids] & ~mask
    self._engaged[env_ids] = self._engaged[env_ids] & ~mask
    if report and reason == ReleaseReason.EXTERNAL:
      # Arm the reseat hold; scripted no-fault releases never freeze the clock.
      self.dr_release_pending[env_ids] = self.dr_release_pending[env_ids] | mask
      self._reseat_hold_steps[env_ids] = torch.where(
        mask.any(dim=-1),
        torch.zeros_like(self._reseat_hold_steps[env_ids]),
        self._reseat_hold_steps[env_ids],
      )
    cooldown_set = max(0, self._mcfg.reengage_cooldown_steps)
    self._cooldown_steps[env_ids] = torch.where(
      mask,
      torch.full_like(self._cooldown_steps[env_ids], cooldown_set),
      self._cooldown_steps[env_ids],
    )
    self._engage_count[env_ids] = latch_state.clear_where(mask, self._engage_count[env_ids])
    self._violation_count[env_ids] = latch_state.clear_where(mask, self._violation_count[env_ids])
    self._steps_since_engage[env_ids] = latch_state.clear_where(mask, self._steps_since_engage[env_ids])
    if report:
      self.forced_release_this_step[env_ids] |= mask
      self.release_reason[env_ids] = torch.where(
        mask,
        torch.full_like(self.release_reason[env_ids], reason),
        self.release_reason[env_ids],
      )
    self._sync_retraction()

  # ------------------------------------------------------- command term

  def compute(self, dt: float) -> None:
    if self.cfg.skip_zero_time_update and dt <= 0.0:
      # A partial reset also calls compute for untouched environments. No
      # physics elapsed: preserve their release events, debounce and ages.
      self._update_metrics()
      return
    super().compute(dt)

  def _update_metrics(self) -> None:
    return

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Command terms are resampled on reset. Reset all latch bookkeeping alongside
    # the command so old contact history cannot leak into the new episode.
    self._engaged[env_ids] = self._initial_engaged if self._mcfg.grip_enabled else False
    self._engage_count[env_ids] = 0
    self._cooldown_steps[env_ids] = 0
    self._violation_count[env_ids] = 0
    self._steps_since_engage[env_ids] = 0
    self.anchor_load_tnl[env_ids] = 0.0
    self.anchor_moment_tnl[env_ids] = 0.0
    self.anchor_effective_load_tnl[env_ids] = 0.0
    self.hold_margin[env_ids] = 1.0
    self.hold_margin_raw[env_ids] = 1.0
    self.dr_release_pending[env_ids] = False
    self._reseat_hold_steps[env_ids] = 0
    self.face_cos[env_ids] = 0.0
    self.contact_count[env_ids] = 0.0
    self.pad_press_force[env_ids] = 0.0
    self.unit_preload_n[env_ids] = 0.0
    self.pad_contact_wrench_tln[env_ids] = 0.0
    self.spine_fraction[env_ids] = 0.0
    self.forced_release_this_step[env_ids] = False
    self.engaged_this_step[env_ids] = False
    self.release_reason[env_ids] = 0
    # Reset-mask latches are not seeds; seed_engaged (which runs after this in
    # the same reset pass) marks the actual seeded pattern.
    self._seeded[env_ids] = False
    self.released_was_seeded[env_ids] = False
    self._sample_limit_params(env_ids)
    self._sync_retraction()

  def _sample_limit_params(self, env_ids: torch.Tensor) -> None:
    """Draw each pad's persistent per-unit quantiles and reset its unit readouts."""
    shape = (len(env_ids), self._num_ee, self.hold_judge.unit_count)
    self.unit_strength_quantile[env_ids] = torch.rand(shape, device=self.device)
    self.unit_loading_quantile[env_ids] = torch.rand(shape, device=self.device)
    self.unit_load_tln[env_ids] = 0
    self.unit_normal_capacity[env_ids] = 0
    self.unit_empirical_supported[env_ids] = True
    self.unit_empirical_early_failed[env_ids] = False
    self.unit_empirical_tail_capped[env_ids] = False

  def _update_command(self) -> None:
    normal_forces = self._read_contacts()
    self.engaged_this_step.zero_()
    self.released_was_seeded.zero_()
    # Orientation and seating readouts are observation channels too, so they
    # stay live on every path, the grip-disabled ablation included.
    self.face_cos = self._face_wall_cos()
    self.spine_fraction = self._virtual_spine_fraction()
    if not self._mcfg.grip_enabled:
      self._clear_grip()
      return

    engage_condition = self._engage_condition()
    was_engaged = self._engaged.clone()
    candidate = engage_condition & ~self._engaged & ~(self._cooldown_steps > 0)
    engage_count = latch_state.count_consecutive(candidate, self._engage_count)
    engage = engage_count >= max(1, self._mcfg.engage_consecutive_steps)
    self.engaged_this_step.copy_(engage)
    # Holds are judged on the pre-transition latch state, so a pad engaged
    # this very step is never judged on stale loads.
    release = self._judge_holds()
    self.forced_release_this_step = release
    new_engaged = latch_state.next_latch_state(self._engaged, engage, release)
    self._debug_log_transitions(was_engaged, new_engaged, engage_condition, normal_forces, engage_count)
    self._commit(was_engaged, new_engaged, engage, release, engage_count)

  # ------------------------------------------------------ step pipeline

  def _read_contacts(self) -> torch.Tensor:
    """Update the contact readouts; returns the per-proxy normal forces (E, H*S)."""
    normal_forces, found_count = self._contacts.proxy_normal_forces()
    if self._mcfg.disable_contacts_while_latched:
      # Latched pads' proxies are retracted and cannot touch the wall; the
      # substep-history recovery must not replay stale pre-latch forces.
      latched = self._engaged.unsqueeze(-1).expand(-1, self._num_ee, self._proxies_per_ee).reshape(self.num_envs, -1)
      normal_forces = torch.where(latched, torch.zeros_like(normal_forces), normal_forces)
      found_count = torch.where(self._engaged, torch.zeros_like(found_count), found_count)
    self.contact_count = found_count
    self.pad_press_force = (
      normal_forces.clamp(min=0.0).view(self.num_envs, self._num_ee, self._proxies_per_ee).sum(dim=2)
    )
    return normal_forces

  def _clear_grip(self) -> None:
    """Grip-disabled ablation: no latches, loads or release events."""
    self._engaged.zero_()
    self._engage_count.zero_()
    self._cooldown_steps.zero_()
    self._violation_count.zero_()
    self._steps_since_engage.zero_()
    self.anchor_load_tnl.zero_()
    self.anchor_moment_tnl.zero_()
    self.anchor_effective_load_tnl.zero_()
    self.hold_margin.fill_(1.0)
    self.hold_margin_raw.fill_(1.0)
    self.forced_release_this_step.zero_()
    self.release_reason.zero_()
    self._seeded.zero_()
    self.unit_preload_n.zero_()
    self.pad_contact_wrench_tln.zero_()
    self._sync_retraction()

  def _engage_condition(self) -> torch.Tensor:
    """Pads eligible to latch this step, before debouncing (E, H) bool."""
    condition = torch.ones_like(self._engaged)
    if self._mcfg.engage_unit_preload_n > 0:
      condition &= self._unit_preload_condition()
    if self._mcfg.engage_parallel_cos_threshold > 0.0:
      condition &= self.face_cos >= self._mcfg.engage_parallel_cos_threshold
    if self._mcfg.engage_min_spine_fraction > 0.0:
      # Seating: enough of the virtual spine field must reach the wall.
      condition &= self.spine_fraction >= self._mcfg.engage_min_spine_fraction
    if self._mcfg.engage_max_antirake_vel > 0.0:
      # Spines hook only under along-rake loading (Wang 2017 §2): a pad sliding
      # up the wall against its rake cannot catch, so feet skidding upward in
      # a pull-up do not farm anchors. Stationary or down-sliding pads latch.
      condition &= self._rake_velocity() >= -self._mcfg.engage_max_antirake_vel
    return condition

  def _judge_holds(self) -> torch.Tensor:
    """Latched pads released this step (E, H) bool; sets ``release_reason``."""
    if self._ls_cfg.enable:
      release = self._limit_surface_release()
      if self._mcfg.debug and bool(release[0].any()):
        load = self.anchor_load_tnl[0].tolist()
        print(
          "[Microspine] LIMIT-SURFACE RELEASE env0 "
          f"released={release[0].tolist()} "
          f"load_tnl={[[round(v, 1) for v in ee] for ee in load]}"
        )
    else:
      release = torch.zeros_like(self._engaged)
    if self._surfaces is not None:
      outside = self._engaged & ~self.spine_field.surface_covered
      release |= outside
      self.release_reason = torch.where(
        outside, torch.full_like(self.release_reason, int(ReleaseReason.OFF_SURFACE)), self.release_reason
      )
    return release

  def _commit(
    self,
    was_engaged: torch.Tensor,
    new_engaged: torch.Tensor,
    engage: torch.Tensor,
    release: torch.Tensor,
    engage_count: torch.Tensor,
  ) -> None:
    """Apply this step's latch transitions to the state machine."""
    # Releasing a seeded anchor demolishes a gift; the policy's own latches
    # are never seeded.
    self.released_was_seeded |= was_engaged & ~new_engaged & self._seeded
    self._seeded &= new_engaged
    self._engaged[:] = new_engaged
    self._tick_reseat_hold()
    self._redraw_empirical_latches(engage)
    # Latched pads need no engage count.
    self._engage_count = latch_state.clear_where(self._engaged, engage_count)
    self._cooldown_steps = latch_state.tick_cooldown(
      self._cooldown_steps, release, max(0, self._mcfg.reengage_cooldown_steps)
    )
    # Latched pads hand their load path to the anchor; released pads get
    # their contacts back for seating and push-off.
    self._sync_retraction()
    self._steps_since_engage = latch_state.advance_engage_age(self._steps_since_engage, self._engaged, engage)
    self._violation_count = latch_state.clear_where(~self._engaged, self._violation_count)

  def _tick_reseat_hold(self) -> None:
    """A re-latch clears a pending reseat; a hold outliving its timeout is abandoned."""
    self.dr_release_pending &= ~self._engaged
    hold_active = self.dr_release_pending.any(dim=-1)
    self._reseat_hold_steps = torch.where(
      hold_active,
      self._reseat_hold_steps + 1,
      torch.zeros_like(self._reseat_hold_steps),
    )
    expired = self._reseat_hold_steps > self._mcfg.dr_reseat_hold_timeout_steps
    self.dr_release_pending = torch.where(
      expired.unsqueeze(-1),
      torch.zeros_like(self.dr_release_pending),
      self.dr_release_pending,
    )
    self._reseat_hold_steps = torch.where(expired, torch.zeros_like(self._reseat_hold_steps), self._reseat_hold_steps)

  def _redraw_empirical_latches(self, engage_transition: torch.Tensor) -> None:
    """Only newly engaged pads change; every retained unit surface is stable."""
    on = engage_transition.unsqueeze(-1)
    self.unit_strength_quantile = torch.where(
      on, torch.rand_like(self.unit_strength_quantile), self.unit_strength_quantile
    )
    self.unit_loading_quantile = torch.where(
      on, torch.rand_like(self.unit_loading_quantile), self.unit_loading_quantile
    )

  def _sync_retraction(self) -> None:
    """Retract latched pads' contact proxies and restore released ones."""
    if self._mcfg.disable_contacts_while_latched:
      self._retraction.sync(self._engaged)

  def _debug_log_transitions(
    self,
    was_engaged: torch.Tensor,
    new_engaged: torch.Tensor,
    engage_condition: torch.Tensor,
    normal_forces: torch.Tensor,
    engage_count: torch.Tensor,
  ) -> None:
    if not self._mcfg.debug:
      return
    changed = was_engaged[0] != new_engaged[0]
    if not changed.any():
      return
    step = self._env.episode_length_buf[0].item()
    proxies = self._proxies_per_ee
    for ee_index, bname in enumerate(self._mcfg.ee_body_names):
      if not changed[ee_index].item():
        continue
      ee_normal = normal_forces[0, ee_index * proxies : (ee_index + 1) * proxies]
      old = "ON" if was_engaged[0, ee_index].item() else "OFF"
      new = "ON" if new_engaged[0, ee_index].item() else "OFF"
      print(
        f"[Microspine] SENSOR step={step} {bname} {old}->{new} "
        f"condition={int(engage_condition[0, ee_index].item())} "
        f"found={int(self.contact_count[0, ee_index].item())}/{proxies} "
        f"normal_max={ee_normal.max().item():.3f}N normal_mean={ee_normal.mean().item():.3f}N "
        f"face_cos={self.face_cos[0, ee_index].item():.3f} "
        f"engage_count={int(engage_count[0, ee_index].item())} "
        f"cooldown={int(self._cooldown_steps[0, ee_index].item())}"
      )

  # ----------------------------------------------- gates and judgements

  def _wall_inward_normal(self) -> torch.Tensor:
    return wall_inward_normal(self._env, self._mcfg.wall_entity_name)

  def _live_wall_inward_normal(self) -> torch.Tensor:
    # Looked up per call so components see a replaced _wall_inward_normal.
    return self._wall_inward_normal()

  def _proxy_inward_normal(self) -> torch.Tensor:
    """Inward surface normal per contact proxy, broadcastable to (E, H*S, 3)."""
    if self._surfaces is not None:
      return -self._surfaces.frame()[1].repeat_interleave(self._proxies_per_ee, dim=1)
    return self._wall_inward_normal()[:, None, :]

  def _ee_quats(self) -> list[torch.Tensor]:
    quats = self._asset.data.body_link_quat_w
    return [quats[:, body_id, :] for body_id in self._ee_body_ids]

  def _engage_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """World-frame engage basis (t̂, l̂, n̂_out), each (E, H, 3); see ``engage_frame``."""
    if self._surfaces is not None:
      n_out = self._surfaces.frame()[1]
    else:
      n_out = -self._wall_inward_normal().unsqueeze(1)
    return engage_frame(self._ee_quats(), self._rake_local_axis, n_out)

  def _face_wall_cos(self) -> torch.Tensor:
    inward = -self._surfaces.frame()[1] if self._surfaces is not None else self._wall_inward_normal()
    return face_wall_cos(self._ee_quats(), self._face_local_axis, inward)

  def _virtual_spine_fraction(self) -> torch.Tensor:
    return self.spine_field.fraction()

  def _rake_velocity(self) -> torch.Tensor:
    """Each pad's sliding velocity along its rake direction t̂, (E, H)."""
    t_hat, _, _ = self._engage_frame()
    vel_w = self._asset.data.body_link_lin_vel_w[:, self._ee_body_ids, :]
    if self._surfaces is not None:
      # Relative to the moving surface, at the pad's contact centre.
      quat = self._asset.data.body_link_quat_w[:, self._ee_body_ids]
      offset = rotate(quat, self.spine_field.surface_centers)
      center = self._asset.data.body_link_pos_w[:, self._ee_body_ids] + offset
      omega = self._asset.data.body_link_ang_vel_w[:, self._ee_body_ids]
      vel_w = vel_w + torch.cross(omega, offset, dim=-1) - self._surfaces.velocity_at(center)
    return (vel_w * t_hat).sum(dim=-1)

  def _unit_preload_condition(self) -> torch.Tensor:
    """Require compression on every unit in the equal-stiffness contact model.

    The pad wrench about the contact-array centre, averaged over the policy
    step, is distributed onto the units exactly as the hold judge distributes
    an anchor load. An outward-normal unit load is tensile, so the preload
    is max(-f_n_unit, 0).
    """
    force, moment = self._contacts.pad_wrench(self._engaged)
    basis = self._engage_frame()
    components = (*to_engage_frame(force, basis), *to_engage_frame(moment, basis))
    self.pad_contact_wrench_tln = torch.stack(components, dim=-1)
    _, _, normal = self.hold_judge.distribute(*components)
    self.unit_preload_n = (-normal).clamp_min(0.0)
    return (
      torch.isfinite(self.pad_contact_wrench_tln).all(dim=-1)
      & torch.isfinite(self.unit_preload_n).all(dim=-1)
      & (self.unit_preload_n >= self._mcfg.engage_unit_preload_n).all(dim=-1)
    )

  def _get_executor(self):
    """Lazily fetch the physics executor (built after command terms)."""
    if self._executor is None:
      term_cfg = self._env.event_manager.get_term_cfg(self.cfg.physics_event_name)
      self._executor = term_cfg.func
      if not hasattr(self._executor, "compute_anchor_loads"):
        raise ValueError(
          "Limit surface requires the constraint executor "
          "(MicrospineConstraint) registered as event term "
          f"'{self.cfg.physics_event_name}'."
        )
    return self._executor

  def _limit_surface_release(self) -> torch.Tensor:
    """Judge the latched pads' anchor loads, (E, H) bool releases.

    The anchor load comes from the constraint efc rows, a smooth function of
    the pose error unlike contact friction impulses, which is what makes the
    release condition detectable at all. Every violation goes through the
    debounce counter.
    """
    executor = self._get_executor()
    loads = executor.compute_anchor_loads()  # (E, H, 3) world
    basis = self._engage_frame()
    f_t, f_l, f_n = to_engage_frame(loads, basis)
    self.anchor_load_tnl = torch.stack([f_t, f_l, f_n], dim=-1)
    if self._ls_cfg.moment_capacity_enable and self._mcfg.anchor_lock_rotation:
      moments = to_engage_frame(executor.anchor_moment_w, basis)
      self.anchor_moment_tnl = torch.stack(moments, dim=-1)
    else:
      self.anchor_moment_tnl.zero_()
      zeros = torch.zeros_like(f_t)
      moments = (zeros, zeros, zeros)

    verdict = self._judge_units(f_t, f_l, f_n, *moments)
    self.anchor_effective_load_tnl = verdict.worst_load_pad_scale
    margin_raw = self.hold_judge.force_only_margin(
      f_t, f_l, f_n, self.unit_strength_quantile, self.unit_loading_quantile
    )
    self.hold_margin = torch.where(self._engaged, verdict.margin, torch.ones_like(verdict.margin))
    self.hold_margin_raw = torch.where(self._engaged, margin_raw, torch.ones_like(margin_raw))

    guard = latch_state.in_hold_guard(
      self._steps_since_engage,
      self._seeded,
      self._ls_cfg.hold_judge_guard_steps,
      self._ls_cfg.seed_hold_judge_guard_steps,
    )
    active = self._engaged & ~guard
    self._violation_count = latch_state.count_consecutive(active & verdict.violation, self._violation_count)
    release = self._violation_count >= max(1, self._ls_cfg.violation_consecutive_steps)
    self.release_reason = torch.where(release, verdict.reason, torch.zeros_like(verdict.reason))
    return release

  def _judge_units(self, f_t, f_l, f_n, m_t, m_l, m_n) -> UnitVerdict:
    """Judge a pad wrench against the persistent unit draws; updates the unit readouts."""
    verdict = self.hold_judge.judge(
      f_t, f_l, f_n, m_t, m_l, m_n, self.unit_strength_quantile, self.unit_loading_quantile
    )
    self.unit_load_tln = verdict.unit_load_tln
    self.unit_normal_capacity = verdict.unit_normal_capacity
    self.unit_empirical_supported = verdict.unit_supported
    self.unit_empirical_early_failed = verdict.unit_early_failed
    self.unit_empirical_tail_capped = verdict.unit_tail_capped
    return verdict
