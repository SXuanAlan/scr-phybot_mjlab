"""Microspine constraint physics: runtime-toggled weld anchors.

Each end-effector owns one pre-declared *inactive* weld equality (a "virtual
anchor"). On an engage transition the executor captures the pad's contact
centre position and orientation into ``eq_data`` and flips ``eq_active`` on;
on release it flips ``eq_active`` off. The solver then holds the pad
bilaterally at every physics substep, so tangential lock and tension
capacity do not depend on contact friction. Orientation is locked too: a
real spine array grips asperities across the whole pad face, so the wall,
not the wrist servo, holds an engaged pad's orientation, and releasing and
re-engaging is the only way to reorient it.

The weld eq_data layout is the connect layout swapped: data[0:3] pairs with
body2 (the world anchor) and data[3:6] with body1 (the pad-local offset).
data[6:10] holds the relpose ``conj(q_anchor)``, since the mjwarp weld
rotational residual is ``vec(inv(q_body2) * q_body1 * relpose)``.

When ``cfg.surfaces`` is supplied, body2 is the declared moving target.
Position and orientation anchors are then captured in that body's local
frame; both bodies receive solver reactions and can move together freely.

Feasibility facts this relies on (mujoco_warp):
  * ``Data.eq_active`` is ``(nworld, neq)`` and read by the equality kernels
    every step, so per-env runtime toggling works on GPU without touching the
    model topology.
  * ``Model.eq_data`` is declared ``("*", neq, vec11)`` and the kernels index
    it with ``worldid % shape[0]``; after mjlab expands the field per env
    (via the ``model_fields`` hook below), per-env anchors are honored.
  * The MJCF compiler recomputes the anchor and relpose slots at qpos0, so
    anchors written in the spec do not survive compilation. The executor
    writes every runtime slot before activating a constraint.

The microspine behavior itself (hold envelope, release rules) lives in
``MicrospineAffordance``; this term is only the physical executor, mirroring
the affordance's latch bits into constraints.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import torch

from scr_phybot_mjab.microspine.anchor_loads import WeldRows, weld_rotational_force_to_world_moment
from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.contact import pad_contact_center, sphere_names
from scr_phybot_mjab.microspine.quat import conjugate, multiply, rotate
from scr_phybot_mjab.microspine.spec import microspine_equality_name
from scr_phybot_mjab.microspine.surfaces import MicrospineSurfaces
from scr_phybot_mjab.microspine.wall import wall_inward_normal

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


def make_constraint_event_cfg(
  cfg: MicrospineCfg,
  affordance_name: str | None = None,
  entity_name: str = "robot",
) -> EventTermCfg:
  """Build the step event term that drives the per-EE anchor constraints."""
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  return EventTermCfg(
    func=MicrospineConstraint,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg(entity_name),
      "microspine_cfg": dataclasses.replace(
        cfg,
        affordance_name=affordance_name,
      ),
    },
  )


class MicrospineConstraint:
  """Mirror the affordance's per-EE engage state into weld equalities.

  Exposes engage state and timing for MDP terms, plus anchor diagnostics for
  the limit-surface state machine (``anchor_pos_w``,
  ``compute_anchor_loads``, ``anchor_error_w``).
  """

  # mjlab expands these model fields per env before the first step (the
  # EventManager reads this attribute right after term construction). Without
  # the expansion, per-env eq_data writes would silently alias one shared row.
  model_fields = ("eq_data", "eq_solref")

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._mcfg: MicrospineCfg = cfg.params["microspine_cfg"]
    self._affordance_name = self._mcfg.affordance_name
    if self._affordance_name is None:
      raise ValueError("MicrospineConstraint requires MicrospineCfg.affordance_name.")

    self._env = env
    self._device = env.device
    self._num_envs = env.num_envs
    self._num_ee = len(self._mcfg.ee_body_names)

    robot_prefix = cfg.params["asset_cfg"].name
    self._asset = env.scene[robot_prefix]
    self._surfaces = MicrospineSurfaces(env, self._mcfg) if self._mcfg.surfaces else None

    # Equality ids from the compiled model (the scene spec_fn added them).
    mj_model = env.sim.mj_model
    eq_ids = [mj_model.equality(microspine_equality_name(ee)).id for ee in self._mcfg.ee_body_names]
    self._eq_ids = torch.tensor(eq_ids, dtype=torch.long, device=self._device)

    # Entity-local EE body indices for body_link_pos_w / body_link_quat_w.
    ee_body_ids: list[int] = []
    for bname in self._mcfg.ee_body_names:
      ids, _ = self._asset.find_bodies([bname])
      ee_body_ids.append(ids[0])
    self._ee_body_ids = torch.tensor(ee_body_ids, dtype=torch.long, device=self._device)

    # Pad-centre offsets in the spine-body frame. The compiler recomputes the
    # weld's anchor slots at qpos0, so the offset is rebuilt from the
    # contact-face geometry (the proxy bodies are direct children of the EE
    # body, so their positions are EE-local) and written back into the model
    # in _check_expansion.
    pad_local_rows = []
    for ee in self._mcfg.ee_body_names:
      ee_id = mj_model.body(f"{robot_prefix}/{ee}").id
      for name in sphere_names(ee, self._mcfg.spheres_per_ee_rows, self._mcfg.spheres_per_ee_cols):
        if int(mj_model.body_parentid[mj_model.body(f"{robot_prefix}/{name}").id]) != ee_id:
          raise ValueError(
            f"Proxy body '{robot_prefix}/{name}' is not a direct child of '{robot_prefix}/{ee}'; "
            "the pad-local anchor assumes EE-local proxy positions."
          )
      pad_local_rows.append(pad_contact_center(mj_model, self._mcfg, ee, robot_prefix))
    self._pad_local = torch.tensor(np.stack(pad_local_rows), dtype=torch.float32, device=self._device)

    # State buffers consumed by observations and rewards.
    self.engaged = torch.zeros(self._num_envs, self._num_ee, dtype=torch.bool, device=self._device)
    self.disengaged_this_step = torch.zeros_like(self.engaged)
    self.engaged_this_step = torch.zeros_like(self.engaged)
    self.current_disengage_time = torch.zeros(self._num_envs, self._num_ee, dtype=torch.float32, device=self._device)
    self.last_engage_command = torch.zeros_like(self.engaged)
    # Cached world anchors; only rows with engaged=True are meaningful.
    self.anchor_pos_w = torch.zeros(self._num_envs, self._num_ee, 3, dtype=torch.float32, device=self._device)
    self.anchor_load_w = torch.zeros_like(self.anchor_pos_w)
    # Physical couple ABOUT C, world axes. The translational rows already
    # apply force at C; adding the B-to-C lever again is wrong.
    self.anchor_moment_w = torch.zeros_like(self.anchor_pos_w)
    # Engage-time pad world orientation (wxyz); the weld locks the pad to it.
    self.anchor_quat_w = torch.zeros(self._num_envs, self._num_ee, 4, dtype=torch.float32, device=self._device)
    self.anchor_quat_w[..., 0] = 1.0
    self.anchor_pos_local = torch.zeros_like(self.anchor_pos_w)
    self.anchor_quat_local = self.anchor_quat_w.clone()
    # Policy steps since the engage transition, maintained here (not in the
    # affordance) because the two engage paths reach the affordance counter
    # with different offsets: a mid-episode transition starts its counter at
    # 0, the reset/initial_on path at 1. The executor sees both as a rising
    # edge, so its own age is path-independent. Re-seating stops at
    # grace - 1 so the first limit-surface check reads post-re-grip physics
    # on both paths.
    self._engage_age = torch.zeros(self._num_envs, self._num_ee, dtype=torch.long, device=self._device)
    self._reseat_max_age = max(1, self._mcfg.limit_surface.hold_grace_steps - 1)
    # Envs whose constraints must be (re)asserted on the next call: after any
    # reset the sim re-tiles eq_active to eq_active0 (all False) and the
    # kinematics seen here are stale, so anchor placement is deferred.
    self._needs_init = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
    self._weld_rows = WeldRows(self._eq_ids, self._num_envs, self._device)
    self._checked_expansion = False

  @property
  def engage_age(self) -> torch.Tensor:
    """Policy-step age of each currently engaged equality.

    This controller-owned timer is safe for reward/observation terms to read;
    exposing it avoids coupling those terms to the executor's private storage.
    """
    return self._engage_age

  # ---------------------------------------------------------------------
  # Affordance input.
  # ---------------------------------------------------------------------

  def _get_engage_command(self, env: ManagerBasedRlEnv) -> torch.Tensor:
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

  # ---------------------------------------------------------------------
  # Kinematics.
  # ---------------------------------------------------------------------

  def pad_center_world(self) -> torch.Tensor:
    """Current pad-center world positions, (E, H, 3)."""
    pos = self._asset.data.body_link_pos_w[:, self._ee_body_ids]
    quat = self._asset.data.body_link_quat_w[:, self._ee_body_ids]
    return pos + rotate(quat, self._pad_local)

  def _down_wall_dir(self) -> torch.Tensor:
    """Down-the-wall rake direction, (E, 1, 3): gravity projected in-plane.

    Mirrors the affordance's gravity-mode engage frame (t-hat), with the same
    world -z fallback when gravity is wall-normal.
    """
    import warp as wp

    if getattr(self, "_surfaces", None) is not None:
      _, normal, _, _ = self._surfaces.frame()
      quat = self._asset.data.body_link_quat_w[:, self._ee_body_ids]
      local_axis = torch.tensor(self._mcfg.limit_surface.ee_rake_local_axis, device=self._device)
      rake = rotate(quat, local_axis)
      rake = rake - (rake * normal).sum(-1, keepdim=True) * normal
      return rake / torch.linalg.vector_norm(rake, dim=-1, keepdim=True).clamp_min(1e-9)

    n_in = wall_inward_normal(self._env, self._mcfg.wall_entity_name)
    g = wp.to_torch(self._env.sim.wp_model.opt.gravity)
    if g.shape[0] != n_in.shape[0]:
      g = g.expand(n_in.shape[0], -1)
    t = g - (g * n_in).sum(dim=-1, keepdim=True) * n_in
    norm = torch.linalg.norm(t, dim=-1, keepdim=True)
    fb = torch.tensor([0.0, 0.0, -1.0], device=n_in.device).expand_as(t)
    fb = fb - (fb * n_in).sum(dim=-1, keepdim=True) * n_in
    fb = fb / torch.linalg.norm(fb, dim=-1, keepdim=True).clamp_min(1.0e-6)
    t = torch.where(norm > 1.0e-6, t / norm.clamp_min(1.0e-9), fb)
    return t.unsqueeze(1)

  # ---------------------------------------------------------------------
  # Anchor load readout (consumed by the affordance's limit surface).
  # ---------------------------------------------------------------------

  def compute_anchor_loads(self) -> torch.Tensor:
    """Per-EE anchor loads from the weld efc rows, (E, H, 3) world.

    The force rows' ``efc.force`` is the world-frame force the constraint
    applies ON the EE (a hanging robot has force_z > 0). The limit surface
    constrains the reaction, the load the spine puts on the asperity, hence
    ``F_load = -efc_force``. The rotational multipliers are converted through
    J_rot.T and cached as the physical world moment ABOUT C in
    ``anchor_moment_w``. Do not shift it from B again: only an externally
    measured B-referenced wrench needs M_C = M_B - (C-B) cross F before unit
    allocation.
    """
    self.refresh_target_anchors()
    found, force_rows, rotational_rows = self._weld_rows.read(self._env.sim.data)
    loads = self.anchor_load_w
    torch.where(found.unsqueeze(-1), force_rows, torch.zeros_like(force_rows), out=loads)
    loads.neg_()  # F_load = -efc_force (EE-on-anchor reaction).
    body_quat_w = self._asset.data.body_link_quat_w[:, self._ee_body_ids]
    moment_on_ee = weld_rotational_force_to_world_moment(
      rotational_rows,
      body_quat_w,
      self.anchor_quat_w,
      self._mcfg.anchor_torquescale,
      self._surfaces.pose()[1] if getattr(self, "_surfaces", None) is not None else None,
    )
    torch.where(
      found.unsqueeze(-1),
      moment_on_ee,
      torch.zeros_like(moment_on_ee),
      out=self.anchor_moment_w,
    )
    self.anchor_moment_w.neg_()
    torch.nan_to_num_(self.anchor_moment_w, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    # Rare solver transients under extreme poses can emit non-finite efc
    # forces; unguarded, NaN loads poison the hold margins and the critic
    # observations. NaN reads as "no reading" (no release that step); inf
    # clamps to a huge finite load the envelope instantly releases.
    torch.nan_to_num_(loads, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    return loads

  def anchor_error_w(self) -> torch.Tensor:
    """Pad-center minus anchor (world), zero where disengaged, (E, H, 3).

    Debug/validation aid: the constraint force is (impedance-scaled)
    proportional to this error, so a linear fit against
    ``compute_anchor_loads`` cross-checks the efc gather without needing the
    per-config invweight scaling analytically.
    """
    self.refresh_target_anchors()
    err = self.pad_center_world() - self.anchor_pos_w
    return torch.where(self.engaged.unsqueeze(-1), err, torch.zeros_like(err))

  # ---------------------------------------------------------------------
  # Constraint writes.
  # ---------------------------------------------------------------------

  def refresh_target_anchors(self) -> None:
    """Update diagnostic world frames, without moving the physical local anchors."""
    if getattr(self, "_surfaces", None) is not None:
      pos, quat = self._surfaces.pose()
      self.anchor_pos_w = pos + rotate(quat, self.anchor_pos_local)
      self.anchor_quat_w = multiply(quat, self.anchor_quat_local)

  def _check_expansion(self) -> None:
    if self._checked_expansion:
      return
    if self._num_envs > 1 and "eq_data" not in self._env.sim.expanded_fields:
      # An unexpanded "*" model field surfaces as a stride-0 view: per-env
      # writes would silently alias a single shared row.
      raise RuntimeError(
        "Model field 'eq_data' was not expanded per env. MicrospineConstraint "
        "must be registered as an mjlab event term so its model_fields hook "
        "runs before the first step."
      )
    # Static weld slots, written once post-expansion: the compiler owns these
    # in the compiled model (recomputed at qpos0), so re-assert the runtime
    # truth. data[3:6] pairs with body1, the pad-local anchor.
    eq_data = self._env.sim.model.eq_data
    eq_data[:, self._eq_ids, 3:6] = self._pad_local
    eq_data[:, self._eq_ids, 10] = self._mcfg.anchor_torquescale
    self._checked_expansion = True

  def _apply_state(self, engage_command: torch.Tensor, env_mask=None) -> None:
    """Branchless per-step constraint update (no host syncs).

    Rising edges capture the anchor at the current pad centre and
    orientation. With ``reanchor_after_grace``, every following step inside
    the grace window re-seats it (re-grip: the spine array seats into the
    asperities under its initial load): the position follows only the pad's
    down-the-wall rake motion, a capped step at a time, and the orientation
    conforms to the pad. The cached anchors of all EEs are then written
    every step (idempotent: the kernels ignore eq_data of inactive
    equalities) and eq_active mirrors the command. Both writes land before
    the next physics step, the "anchor before activate" ordering the
    compiled eq_data overwrite demands.
    """
    self._check_expansion()
    self.refresh_target_anchors()
    pad_raw = self.pad_center_world()
    rising = engage_command & ~self.engaged
    continuing = engage_command & self.engaged
    if env_mask is not None:
      rising &= env_mask[:, None]
      continuing &= env_mask[:, None]
    age = torch.where(continuing, self._engage_age + 1, torch.zeros_like(self._engage_age))
    self._engage_age = age if env_mask is None else torch.where(env_mask[:, None], age, self._engage_age)
    self.anchor_pos_w = torch.where(rising.unsqueeze(-1), pad_raw, self.anchor_pos_w)
    capture = rising
    if self._mcfg.reanchor_after_grace:
      reseat = continuing & (self._engage_age <= self._reseat_max_age)
      # Spines seat by dragging DOWN the wall into asperities (Wang 2017
      # engagement); hooks never relocate outward, upward or sideways. Latched
      # pads' proxies are retracted, so the press preload simply lands as
      # anchor compression, which the limit surface treats as free.
      t_hat = self._down_wall_dir()
      follow = (
        ((pad_raw - self.anchor_pos_w) * t_hat)
        .sum(dim=-1, keepdim=True)
        .clamp(min=0.0, max=self._mcfg.anchor_reseat_step_cap)
      )
      self.anchor_pos_w = torch.where(
        reseat.unsqueeze(-1),
        self.anchor_pos_w + follow * t_hat,
        self.anchor_pos_w,
      )
      # Orientation conforms during the same window: the chassis floats on the
      # spine suspensions and rotates into its loaded equilibrium while the
      # tips grip, so the weld locks the settled orientation rather than spawn
      # sag or a fresh-latch load flip. After grace it freezes.
      capture = capture | reseat
    eq_data = self._env.sim.model.eq_data
    pad_quat = self._asset.data.body_link_quat_w[:, self._ee_body_ids]
    self.anchor_quat_w = torch.where(capture.unsqueeze(-1), pad_quat, self.anchor_quat_w)
    position, orientation = self.anchor_pos_w, self.anchor_quat_w
    if getattr(self, "_surfaces", None) is not None:
      target_pos, target_quat = self._surfaces.pose()
      self.anchor_pos_local = torch.where(
        capture.unsqueeze(-1),
        rotate(conjugate(target_quat), position - target_pos),
        self.anchor_pos_local,
      )
      self.anchor_quat_local = torch.where(
        capture.unsqueeze(-1),
        multiply(conjugate(target_quat), orientation),
        self.anchor_quat_local,
      )
      position, orientation = self.anchor_pos_local, self.anchor_quat_local
    # data[0:3] = body2-local anchor, data[6:10] = relpose. The rotational
    # residual is vec(q_pad * relpose) against body2, so locking the
    # engage-time orientation means relpose = conj(anchor_quat).
    eq_data[:, self._eq_ids, 0:3] = position
    eq_data[:, self._eq_ids, 6:7] = orientation[..., 0:1]
    eq_data[:, self._eq_ids, 7:10] = -orientation[..., 1:4]
    self._env.sim.data.eq_active[:, self._eq_ids] = engage_command

  def _update_disengage_time(self, env: ManagerBasedRlEnv, was_engaged: torch.Tensor) -> None:
    self.engaged_this_step = ~was_engaged & self.engaged
    self.disengaged_this_step = was_engaged & ~self.engaged
    self.current_disengage_time = torch.where(
      self.engaged,
      torch.zeros_like(self.current_disengage_time),
      self.current_disengage_time + env.step_dt,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    **kwargs,
  ) -> None:
    was_engaged = self.engaged.clone()
    engage_command = self._get_engage_command(env)

    # Freshly reset envs: eq_active was re-tiled to False by sim.reset and
    # the anchor must be placed from current (now fresh) kinematics, so treat
    # them as fully disengaged to route initial_on through the rising edge.
    needs_init = self._needs_init.unsqueeze(1)
    self.engaged = self.engaged & ~needs_init
    was_engaged = was_engaged & ~needs_init
    self._needs_init.fill_(False)

    self._apply_state(engage_command)
    self.engaged = engage_command.clone()
    self.last_engage_command = engage_command
    self._update_disengage_time(env, was_engaged)

  def assert_now(self, env: ManagerBasedRlEnv) -> None:
    """Apply the current affordance state immediately.

    Validation scripts call this right after ``env.reset()`` to close the
    one-decimation gap before the first step event would place the anchors.
    """
    self(env, None)

  def initialize_seeded_anchors(self, env_ids: torch.Tensor) -> None:
    """Install mature RSI anchors after FK refresh, before any physics step.

    A partial reset must not advance another environment's seating clock or
    move its anchors. Birth is an initial holding state, not an engage event.
    Fresh physical engages still use the configured down-rake seating window.
    """
    mask = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
    mask[env_ids] = True
    self.engaged[env_ids] = False
    command = torch.where(mask[:, None], self._get_engage_command(self._env), self.engaged)
    self._apply_state(command, env_mask=mask)
    self.engaged = command.clone()
    self.last_engage_command[env_ids] = command[env_ids]
    self._engage_age[env_ids] = self._reseat_max_age + 1
    self._needs_init[env_ids] = False
    self.engaged_this_step[env_ids] = False
    self.disengaged_this_step[env_ids] = False
    self.current_disengage_time[env_ids] = 0.0

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.engaged[ids] = False
    self.engaged_this_step[ids] = False
    self.disengaged_this_step[ids] = False
    self.current_disengage_time[ids] = 0.0
    self.last_engage_command[ids] = False
    self.anchor_pos_w[ids] = 0.0
    self.anchor_load_w[ids] = 0.0
    self.anchor_moment_w[ids] = 0.0
    self.anchor_quat_w[ids] = 0.0
    self.anchor_quat_w[ids, :, 0] = 1.0
    if hasattr(self, "anchor_pos_local"):
      self.anchor_pos_local[ids] = 0.0
      self.anchor_quat_local[ids] = 0.0
      self.anchor_quat_local[ids, :, 0] = 1.0
    self._engage_age[ids] = 0
    # eq_active for these envs was already re-tiled to False by sim.reset
    # (which runs before event-term resets); kinematics here are stale, so
    # anchor placement is deferred to the next __call__ in the same step.
    self._needs_init[ids] = True
