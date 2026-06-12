"""Microspine constraint physics: runtime-toggled equality anchors.

This is the ``attachment_mode == "constraint"`` executor. Instead of pulling
the spine spheres into the wall with a large external force (legacy
``MicrospinePhysics``), each end-effector owns one pre-declared *inactive*
connect equality ("virtual anchor"). On an engage transition the executor
writes the current pad-center world position into ``eq_data[..., 3:6]`` and
flips ``eq_active`` on; on release it flips ``eq_active`` off. The solver then
holds the pad bilaterally at every physics substep — tangential lock and
tension capacity no longer depend on contact friction at all.

With ``cfg.anchor_lock_rotation`` the equality is a *weld* instead of a
connect: 6 efc rows (3 force + 3 torque) and the executor additionally
captures the pad world orientation at engage time into ``eq_data[..., 6:10]``
(as ``conj(q_pad)`` — the mjwarp weld rotational residual is
``vec(inv(q_body2) * q_body1 * relpose)`` and body2 is the world), so an
engaged pad is locked in position AND orientation, like a real spine array
whose orientation is held by the wall rather than the wrist servo. NOTE the
weld eq_data layout is the connect layout *swapped*: data[0:3] pairs with
body2 (world anchor) and data[3:6] with body1 (pad-local offset).

Feasibility facts this relies on (mujoco_warp 3.6.0):
  * ``Data.eq_active`` is ``(nworld, neq)`` and read by the equality kernels
    every step, so per-env runtime toggling works on GPU without touching the
    model topology.
  * ``Model.eq_data`` is declared ``("*", neq, vec11)`` and the kernels index
    it with ``worldid % shape[0]`` — after mjlab expands the field per env
    (via the ``model_fields`` hook below), per-env anchors are honored.
  * The MJCF compiler overwrites ``eq_data[3:6]`` with the body1 anchor's
    world position at qpos0, so anchors written in the spec do NOT survive
    compilation. The executor therefore always writes the runtime anchor
    before activating a constraint.

The anisotropic/asymmetric microspine behavior itself (hold envelope, release
rules) intentionally lives in ``MicrospineAffordance``; this term is only the
physical executor, mirroring the affordance's boolean state into constraints.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

# ConstraintType is not re-exported at package level; the efc rows we filter
# are written by mujoco_warp kernels, so its enum is the source of truth.
from mujoco_warp._src.types import ConstraintType

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

from scr_phybot_mjab.microspine.config import MicrospineCfg
from scr_phybot_mjab.microspine.physics import _sphere_names

_EFC_EQUALITY = int(ConstraintType.EQUALITY)


def microspine_equality_name(ee_body_name: str) -> str:
  """Return the scene-level equality name for one end-effector."""
  return f"microspine_{ee_body_name}"


def _quat_rotate_batched(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """Rotate vec by quat (wxyz). quat (..., 4), vec broadcastable to (..., 3)."""
  w = quat[..., 0:1]
  q = quat[..., 1:4]
  v = vec.expand(q.shape)
  t = 2.0 * torch.cross(q, v, dim=-1)
  return v + w * t + torch.cross(q, t, dim=-1)


def _wall_inward_normal_w(
  env: ManagerBasedRlEnv, wall_entity_name: str
) -> torch.Tensor:
  """Wall inward normal (robot-to-wall, wall local +x) in world frame, (E, 3)."""
  wall = env.scene[wall_entity_name]
  wquat = wall.data.root_link_quat_w
  ww, wx, wy, wz = wquat[:, 0], wquat[:, 1], wquat[:, 2], wquat[:, 3]
  inward_x = 1.0 - 2.0 * (wy * wy + wz * wz)
  inward_y = 2.0 * (wx * wy + ww * wz)
  inward_z = 2.0 * (wx * wz - ww * wy)
  return torch.stack([inward_x, inward_y, inward_z], dim=-1)


def pad_local_anchor(
  cfg: MicrospineCfg,
  sphere_pos: np.ndarray,
  sphere_radius: float,
) -> np.ndarray:
  """Pad-center anchor in the spine-body frame: the centroid of the EE's
  contact spheres pushed one sphere radius along ``face_local_axis`` (toward
  the wall)."""
  face_axis = np.asarray(cfg.face_local_axis, dtype=float)
  face_axis = face_axis / max(float(np.linalg.norm(face_axis)), 1.0e-9)
  return np.mean(sphere_pos, axis=0) + sphere_radius * face_axis


def add_attachment_equalities(
  spec: mujoco.MjSpec,
  cfg: MicrospineCfg,
  robot_prefix: str = "robot",
) -> None:
  """Add one inactive equality ("virtual anchor") per end-effector.

  connect (default): ``data[0:3]`` is the pad-local anchor; ``data[3:6]`` is
  recomputed by the compiler and rewritten at runtime on every engage, so its
  spec value is irrelevant.

  weld (``cfg.anchor_lock_rotation``): every slot the compiler does not own
  is written at runtime (the compiler recomputes the anchor/relpose slots at
  qpos0, so nothing authored here survives); only torquescale is authored for
  spec readability and re-asserted by the executor anyway.
  """
  for ee in cfg.ee_body_names:
    sphere_bodies = [
      f"{robot_prefix}/{name}"
      for name in _sphere_names(
        ee, cfg.spheres_per_ee_rows, cfg.spheres_per_ee_cols
      )
    ]
    anchor1 = pad_local_anchor(
      cfg,
      np.stack([np.asarray(spec.body(name).pos) for name in sphere_bodies]),
      float(spec.geom(sphere_bodies[0]).size[0]),
    )
    data = np.zeros(11)
    if cfg.anchor_lock_rotation:
      data[10] = cfg.anchor_torquescale
    else:
      data[0:3] = anchor1
    spec.add_equality(
      name=microspine_equality_name(ee),
      type=(
        mujoco.mjtEq.mjEQ_WELD
        if cfg.anchor_lock_rotation
        else mujoco.mjtEq.mjEQ_CONNECT
      ),
      objtype=mujoco.mjtObj.mjOBJ_BODY,
      name1=f"{robot_prefix}/{ee}",
      name2="",  # World.
      active=False,
      solref=list(cfg.anchor_solref),
      solimp=list(cfg.anchor_solimp),
      data=data.tolist(),
    )


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
  """Mirror the affordance's per-EE engage state into connect equalities.

  Exposes the same duck-typed surface as ``MicrospinePhysics`` (``engaged``,
  ``disengaged_this_step``, ``current_disengage_time``) so the existing MDP
  terms keep working, plus anchor diagnostics for the limit-surface state
  machine (``anchor_pos_w``, ``compute_anchor_loads``, ``anchor_error_w``).
  """

  # mjlab expands these model fields per env before the first step (the
  # EventManager reads this attribute right after term construction). Without
  # the expansion, per-env eq_data writes would silently alias one shared row.
  model_fields = ("eq_data", "eq_solref")

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._mcfg: MicrospineCfg = cfg.params["microspine_cfg"]
    self._affordance_name = self._mcfg.affordance_name
    if self._affordance_name is None:
      raise ValueError(
        "MicrospineConstraint requires MicrospineCfg.affordance_name."
      )
    if self._mcfg.limit_surface.release_mode != "binary":
      raise NotImplementedError(
        "Only release_mode='binary' is implemented; 'slip_drag' is reserved "
        "for a later milestone."
      )

    self._env = env
    self._device = env.device
    self._num_envs = env.num_envs
    self._num_ee = len(self._mcfg.ee_body_names)
    self._debug = self._mcfg.debug
    self._lock_rotation = self._mcfg.anchor_lock_rotation

    robot_prefix = cfg.params["asset_cfg"].name
    self._asset = env.scene[robot_prefix]

    # Equality ids from the compiled model (the scene spec_fn added them).
    mj_model = env.sim.mj_model
    eq_ids = [
      mj_model.equality(microspine_equality_name(ee)).id
      for ee in self._mcfg.ee_body_names
    ]
    self._eq_ids = torch.tensor(eq_ids, dtype=torch.long, device=self._device)

    # Entity-local EE body indices for body_link_pos_w / body_link_quat_w.
    ee_body_ids: list[int] = []
    for bname in self._mcfg.ee_body_names:
      ids, _ = self._asset.find_bodies([bname])
      ee_body_ids.append(ids[0])
    self._ee_body_ids = torch.tensor(
      ee_body_ids, dtype=torch.long, device=self._device
    )

    # Pad-center offsets in the spine-body frame. For connect the single
    # source of truth is the compiled eq_data[:, 0:3] written by
    # add_attachment_equalities; for weld the compiler recomputes those slots
    # at qpos0, so the offset is rebuilt from the sphere-grid geometry (the
    # sphere bodies are direct children of the EE body, so body pos is
    # EE-local) and written back into the model in _check_expansion.
    if self._lock_rotation:
      pad_local_rows = []
      for ee in self._mcfg.ee_body_names:
        sphere_names = [
          f"{robot_prefix}/{name}"
          for name in _sphere_names(
            ee,
            self._mcfg.spheres_per_ee_rows,
            self._mcfg.spheres_per_ee_cols,
          )
        ]
        ee_id = mj_model.body(f"{robot_prefix}/{ee}").id
        for name in sphere_names:
          if int(mj_model.body_parentid[mj_model.body(name).id]) != ee_id:
            raise ValueError(
              f"Sphere body '{name}' is not a direct child of "
              f"'{robot_prefix}/{ee}'; pad-local anchor recomputation "
              "assumes EE-local sphere positions."
            )
        pad_local_rows.append(
          pad_local_anchor(
            self._mcfg,
            np.stack(
              [np.asarray(mj_model.body(name).pos) for name in sphere_names]
            ),
            float(mj_model.geom(sphere_names[0]).size[0]),
          )
        )
      pad_local = np.stack(pad_local_rows)
    else:
      pad_local = np.stack([mj_model.eq_data[eq, 0:3] for eq in eq_ids])
    self._pad_local = torch.tensor(
      pad_local, dtype=torch.float32, device=self._device
    )

    if self._mcfg.anchor_project_to_wall:
      wall_geom = mj_model.geom(self._mcfg.wall_geom_name)
      self._wall_half_thickness = float(wall_geom.size[0])
    else:
      self._wall_half_thickness = 0.0

    # State buffers (same duck-typed surface as MicrospinePhysics).
    self.engaged = torch.zeros(
      self._num_envs, self._num_ee, dtype=torch.bool, device=self._device
    )
    self.disengaged_this_step = torch.zeros_like(self.engaged)
    self.current_disengage_time = torch.zeros(
      self._num_envs, self._num_ee, dtype=torch.float32, device=self._device
    )
    self.last_engage_command = torch.zeros_like(self.engaged)
    # Cached world anchors; only rows with engaged=True are meaningful.
    self.anchor_pos_w = torch.zeros(
      self._num_envs, self._num_ee, 3, dtype=torch.float32, device=self._device
    )
    self.anchor_load_w = torch.zeros_like(self.anchor_pos_w)
    # Engage-time pad world orientation (wxyz); weld mode locks the pad to it.
    self.anchor_quat_w = torch.zeros(
      self._num_envs, self._num_ee, 4, dtype=torch.float32, device=self._device
    )
    self.anchor_quat_w[..., 0] = 1.0
    # Policy steps since the engage transition, maintained here (not in the
    # affordance) because the two engage paths reach the affordance counter
    # with different offsets: a mid-episode transition starts its counter at
    # 0, the reset/initial_on path at 1. The executor sees both as a rising
    # edge, so its own age is path-independent. Re-seating stops at
    # grace - 1 so the first limit-surface check reads post-re-grip physics
    # on both paths.
    self._engage_age = torch.zeros(
      self._num_envs, self._num_ee, dtype=torch.long, device=self._device
    )
    self._reseat_max_age = max(1, self._mcfg.limit_surface.hold_grace_steps - 1)
    # Envs whose constraints must be (re)asserted on the next call: after any
    # reset the sim re-tiles eq_active to eq_active0 (all False) and the
    # kinematics seen here are stale, so anchor placement is deferred.
    self._needs_init = torch.ones(
      self._num_envs, dtype=torch.bool, device=self._device
    )
    self._row_arange = None  # Lazily sized to njmax at first load readout.
    self._checked_expansion = False

  # ---------------------------------------------------------------------
  # Affordance input (same contract as MicrospinePhysics).
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
    return pos + _quat_rotate_batched(quat, self._pad_local)

  def _project_to_wall(self, points: torch.Tensor) -> torch.Tensor:
    """Project points onto the robot-side wall face plane, (E, H, 3)."""
    wall = self._env.scene[self._mcfg.wall_entity_name]
    inward = _wall_inward_normal_w(self._env, self._mcfg.wall_entity_name)
    face_point = (
      wall.data.root_link_pos_w - self._wall_half_thickness * inward
    )
    offset = (
      (points - face_point.unsqueeze(1)) * inward.unsqueeze(1)
    ).sum(dim=-1, keepdim=True)
    return points - offset * inward.unsqueeze(1)

  # ---------------------------------------------------------------------
  # Anchor load readout (consumed by the affordance's limit surface).
  # ---------------------------------------------------------------------

  def compute_anchor_loads(self) -> torch.Tensor:
    """Gather per-EE anchor loads from the equality efc rows, (E, H, 3).

    A connect contributes 3 consecutive efc rows whose Jacobian is the world
    x/y/z positional Jacobian of the pad point, so ``efc.force`` is the
    world-frame force the constraint applies ON the EE (a hanging robot has
    force_z > 0). The limit surface constrains the reaction — the load the
    spine puts on the asperity — hence ``F_load = -efc_force``. A weld
    (anchor_lock_rotation) contributes 6 rows with the same 3 force rows
    first, so the first-match + [0,1,2] gather is mode-independent; the
    trailing 3 rotational rows (residual units: quat axis * torquescale) are
    intentionally not read here.

    Only constraint counters are zeroed each step; rows past ``ne`` are stale
    garbage and EQUALITY == 0 would happily match them, so the ``row < ne``
    mask is mandatory. ``ne`` can also exceed the written rows on njmax
    overflow, hence the clamp.
    """
    data = self._env.sim.data
    njmax = int(data.njmax)
    if self._row_arange is None:
      self._row_arange = torch.arange(njmax, device=self._device)
      self._comp_arange = torch.arange(3, device=self._device)

    ne = data.ne[:].to(torch.long).clamp(max=njmax)
    valid = self._row_arange.unsqueeze(0) < ne.unsqueeze(1)
    is_eq = valid & (data.efc.type[:, :njmax] == _EFC_EQUALITY)
    efc_id = data.efc.id[:, :njmax]
    efc_force = data.efc.force[:, :njmax]

    # (E, njmax, H) row-vs-equality match; each engaged connect owns exactly 3
    # consecutive rows, so the first matching row + [0, 1, 2] addresses them.
    match = is_eq.unsqueeze(-1) & (
      efc_id.unsqueeze(-1) == self._eq_ids.view(1, 1, -1)
    )
    found = match.any(dim=1)  # (E, H)
    first = match.long().argmax(dim=1)  # (E, H); 0 where not found.
    rows = (first.unsqueeze(-1) + self._comp_arange).clamp(max=njmax - 1)
    gathered = efc_force.gather(
      1, rows.reshape(self._num_envs, -1)
    ).reshape(self._num_envs, self._num_ee, 3)
    loads = self.anchor_load_w
    torch.where(
      found.unsqueeze(-1), gathered, torch.zeros_like(gathered), out=loads
    )
    loads.neg_()  # F_load = -efc_force (EE-on-anchor reaction).
    # Rare solver transients under extreme poses can emit non-finite efc
    # forces; unguarded they reach the limit-surface margins as NaN (inf
    # loads make cap - load = inf - inf) and the critic observations kill
    # the whole training run (observed once at 2048 envs x ~94 iters). NaN
    # reads as "no reading" (no release that step); inf clamps to a huge
    # finite load the envelope instantly, correctly, releases.
    torch.nan_to_num_(loads, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    return loads

  def anchor_error_w(self) -> torch.Tensor:
    """Pad-center minus anchor (world), zero where disengaged, (E, H, 3).

    Debug/validation aid: the constraint force is (impedance-scaled)
    proportional to this error, so a linear fit against
    ``compute_anchor_loads`` cross-checks the efc gather without needing the
    per-config invweight scaling analytically.
    """
    err = self.pad_center_world() - self.anchor_pos_w
    return torch.where(self.engaged.unsqueeze(-1), err, torch.zeros_like(err))

  # ---------------------------------------------------------------------
  # Constraint writes.
  # ---------------------------------------------------------------------

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
    if self._lock_rotation:
      # Static weld slots, written once post-expansion: the compiler owns
      # these in the compiled model (recomputed at qpos0), so re-assert the
      # runtime truth. data[3:6] pairs with body1 (pad-local anchor) in the
      # weld kernel — the connect layout swapped.
      eq_data = self._env.sim.model.eq_data
      eq_data[:, self._eq_ids, 3:6] = self._pad_local
      eq_data[:, self._eq_ids, 10] = self._mcfg.anchor_torquescale
    self._checked_expansion = True

  def _apply_state(self, engage_command: torch.Tensor) -> None:
    """Branchless per-step constraint update (no host syncs).

    Rising edges place the anchor at the current (wall-projected) pad center.
    With ``reanchor_after_grace``, every following step inside the grace
    window re-seats the anchor at the *unprojected* settled pad position
    (re-grip: the spine array seats into the asperities under its initial
    load). Re-seating at the raw settled point matters: the anchor follows
    the pad as contact compression from any pad tilt / pose preload pushes it
    to its structural equilibrium, instead of fighting the contacts in an
    anchor-vs-contact standoff that reads as phantom pull-off tension. In
    weld mode the pad world orientation is captured/re-seated alongside the
    position, so the rotational standoff walks out during grace the same way.
    The cached anchors of all EEs are then written every step (idempotent —
    the kernels ignore eq_data of inactive equalities) and eq_active mirrors
    the command. Both writes land before the next physics step, satisfying
    the "anchor before activate" ordering the compiled eq_data overwrite
    demands.
    """
    self._check_expansion()
    pad_raw = self.pad_center_world()
    pad_engage = (
      self._project_to_wall(pad_raw)
      if self._mcfg.anchor_project_to_wall
      else pad_raw
    )
    rising = engage_command & ~self.engaged
    continuing = engage_command & self.engaged
    self._engage_age = torch.where(
      continuing, self._engage_age + 1, torch.zeros_like(self._engage_age)
    )
    self.anchor_pos_w = torch.where(
      rising.unsqueeze(-1), pad_engage, self.anchor_pos_w
    )
    capture = rising
    if self._mcfg.reanchor_after_grace:
      reseat = continuing & (self._engage_age <= self._reseat_max_age)
      self.anchor_pos_w = torch.where(
        reseat.unsqueeze(-1), pad_raw, self.anchor_pos_w
      )
      capture = capture | reseat
    eq_data = self._env.sim.model.eq_data
    if self._lock_rotation:
      pad_quat = self._asset.data.body_link_quat_w[:, self._ee_body_ids]
      self.anchor_quat_w = torch.where(
        capture.unsqueeze(-1), pad_quat, self.anchor_quat_w
      )
      # Weld layout: data[0:3] = world anchor (body2 side), data[6:10] =
      # relpose. The rotational residual is vec(q_pad * relpose) against the
      # world, so locking the engage-time orientation means relpose =
      # conj(anchor_quat).
      eq_data[:, self._eq_ids, 0:3] = self.anchor_pos_w
      eq_data[:, self._eq_ids, 6:7] = self.anchor_quat_w[..., 0:1]
      eq_data[:, self._eq_ids, 7:10] = -self.anchor_quat_w[..., 1:4]
    else:
      eq_data[:, self._eq_ids, 3:6] = self.anchor_pos_w
    self._env.sim.data.eq_active[:, self._eq_ids] = engage_command

  def _update_disengage_time(
    self, env: ManagerBasedRlEnv, was_engaged: torch.Tensor
  ) -> None:
    self.disengaged_this_step = was_engaged & ~self.engaged
    self.current_disengage_time = torch.where(
      self.engaged,
      torch.zeros_like(self.current_disengage_time),
      self.current_disengage_time + env.step_dt,
    )

  def _log_transitions(
    self, env: ManagerBasedRlEnv, was_engaged: torch.Tensor
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
      anchor = self.anchor_pos_w[0, ee_index].tolist()
      print(
        f"[MicrospineConstraint] TRANSITION step={step} {bname} "
        f"{old}->{new} anchor=({anchor[0]:.3f}, {anchor[1]:.3f}, "
        f"{anchor[2]:.3f})"
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
    self._log_transitions(env, was_engaged)

  def assert_now(self, env: ManagerBasedRlEnv) -> None:
    """Apply the current affordance state immediately.

    Validation scripts call this right after ``env.reset()`` to close the
    one-decimation gap before the first step event would place the anchors.
    """
    self(env, None)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.engaged[ids] = False
    self.disengaged_this_step[ids] = False
    self.current_disengage_time[ids] = 0.0
    self.last_engage_command[ids] = False
    self.anchor_pos_w[ids] = 0.0
    self.anchor_load_w[ids] = 0.0
    self.anchor_quat_w[ids] = 0.0
    self.anchor_quat_w[ids, :, 0] = 1.0
    self._engage_age[ids] = 0
    # eq_active for these envs was already re-tiled to False by sim.reset
    # (which runs before event-term resets); kinematics here are stale, so
    # anchor placement is deferred to the next __call__ in the same step.
    self._needs_init[ids] = True
