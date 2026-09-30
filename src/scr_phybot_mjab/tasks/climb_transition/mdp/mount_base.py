"""Scripted WBC+QP mount base with force-controlled engage/release.

The B-line offline teacher: executes the staggered mount prologue
(stand → raise → LH weld → braced RH weld → feet onto the wall → handoff)
closed-loop in one play env, per control step:

  event-gated acyclic clock → mixed-contact QP (λ*, c*) → kinematic WBC → q_d

Ported from the climb scripted base (climb-osc-wbc-base branch,
``tasks/climb_wall/mdp/scripted_base.py``). Mount-specific structure:

  * the clock runs the clip ONCE (no wrap), clamping at the last frame;
  * the support set mixes microspine anchors with FLOOR contacts — stance
    feet contribute heel/toe friction-cone units to the QP and 6
    zero-velocity rows to the WBC; a foot the clip slides (the toe-pivot
    stretch of right_foot_mount) keeps only its toe unit and its WBC rows
    become clip tracking;
  * foot lift-off is pre-unloaded through the QP share caps exactly like a
    pad release (the CoM walks onto the remaining support first);
  * while the clock is frozen on a pending seat, the CoM command's
    wall-normal component follows the QP (which feels the press pushback
    and leans the body IN to buy balance) while the lateral/vertical
    components station-hold — the standing press is the one place the
    parent's pure station-keeping is wrong (tipping budget lives in the
    ankle strategy);
  * the admittance press target depends on support: a free-standing first
    weld presses to the gentle weldable band, a braced limb (any anchor
    already holding) presses firmly.

All the parent's hard-won mechanics survive: "该挂未挂 freezes, 该松未松
does not" (deadlock), admittance integrates only once the pad feels the
wall, the press channel bypasses the WBC through the limb's own jacobian
row once the pad reaches the wall (press_lock), release is the pre-unloaded
up-wall reverse-shear ladder, and the position command integrates under a
windup cap and soft joint limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from scr_phybot_mjab.tasks.climb_transition.mdp.mount_qp import (
  ContactUnit,
  MountQpCfg,
  MountQpSolution,
  solve_mount_qp,
)
from scr_phybot_mjab.tasks.climb_transition.mdp.mount_wbc import (
  MountKinematicWbc,
  MountWbcGains,
  SwingTarget,
  flat_on_wall_rotation,
  quat_to_mat,
)

_FOOT_EE_INDICES = (2, 3)  # (left_foot_spine, right_foot_spine) in EE order
_FOOT_BODY_NAMES = ("left_ankle_roll", "right_ankle_roll")


class MountClip:
  """Read-only view of the raw mount-prologue npy (world-frame, acyclic).

  Deliberately bypasses ``ClimbMotionClip``: the command term's view is
  root-relative and drops the floor/anchor tables the controller needs
  (foot_on_ground, pad_positions, ee_anchor_positions, segment names).
  """

  def __init__(self, motion_file: str | Path) -> None:
    d = np.load(Path(motion_file), allow_pickle=True).item()
    self.fps = float(d["fps"])
    self.joint_names = tuple(d["joints_list"])
    self.ee_names = tuple(d["ee_list"])
    self.joint_pos = np.asarray(d["joint_positions"], dtype=np.float64)
    self.root_pos = np.asarray(d["root_position"], dtype=np.float64)
    self.root_quat = np.asarray(d["root_quaternion"], dtype=np.float64)
    self.ee_pos = np.asarray(d["ee_positions"], dtype=np.float64)
    self.pad_pos = np.asarray(d["pad_positions"], dtype=np.float64)
    self.anchor_pos = np.asarray(d["ee_anchor_positions"], dtype=np.float64)
    self.ee_engaged = np.asarray(d["ee_engaged"], dtype=bool)
    self.foot_on_ground = np.asarray(d["foot_on_ground"], dtype=bool)
    self.segment_names = tuple(d["segment_names"])
    self.phase_segment = np.asarray(d["phase_segment"], dtype=int)
    self.wall_surface_x = float(d["wall_surface_x"])
    self.wall_angle = float(d["wall_angle"])
    self.frames = int(self.joint_pos.shape[0])

    # Per-EE next engage-rise / engage-fall times (seconds from each frame;
    # +inf when no further event). Acyclic: no wrap, no phantom edges.
    def _next_event_s(col: np.ndarray, rising: bool) -> np.ndarray:
      diff = np.diff(col.astype(int))
      edges = np.where(diff == (1 if rising else -1))[0] + 1
      out = np.full(len(col), np.inf)
      nxt = np.inf
      for f in range(len(col) - 1, -1, -1):
        out[f] = (nxt - f) / self.fps if np.isfinite(nxt) else np.inf
        if f in set(edges.tolist()):
          nxt = f
          out[f] = 0.0
      return out

    num_ee = self.ee_engaged.shape[1]
    self.t_to_rise = np.stack(
      [_next_event_s(self.ee_engaged[:, h], True) for h in range(num_ee)],
      axis=1,
    )
    self.t_to_foot_lift = np.stack(
      [
        _next_event_s(self.foot_on_ground[:, f], False)
        for f in range(self.foot_on_ground.shape[1])
      ],
      axis=1,
    )

    # Reference foot speed (m/s) — distinguishes a parked stance foot from
    # the toe-pivot slide (foot_on_ground still true, foot moving).
    foot_ref = self.ee_pos[:, list(_FOOT_EE_INDICES), :]
    vel = np.diff(foot_ref, axis=0) * self.fps
    speed = np.linalg.norm(vel, axis=-1)
    self.foot_ref_speed = np.vstack([speed, speed[-1:]])  # (F, 2)

  def segment_of(self, frame: int) -> str:
    return self.segment_names[int(self.phase_segment[frame])]


@dataclass
class MountBaseCfg:
  # -- force-controlled seat (admittance) ------------------------------------
  press_force_free: float = 12.0
  """Normal-force target (N) for a free-standing first weld: the engine
  needs one sphere ≥ 3 N over 5 consecutive steps; a near-flat pad splits
  the total over ≤4 spheres, so 12 N clears the gate with margin while
  staying inside the standing tipping budget (17–35 N)."""

  press_force_braced: float = 22.0
  """Normal-force target (N) once any anchor already holds — the braced
  economy the training side moved to (v55): real preload, cheap balance."""

  press_force_foot: float = 12.0
  """Foot (shin-pad) press target (N) — gentle: the toe still carries the
  leg's share and the stiff leg chain overshoots hard."""

  toe_ground_gain: float = 3.0
  """rad/s of ankle-pitch bias per meter of toe clearance: the toe-planted
  servo. Playing the reference leg joints leaves the toe wherever the
  BODY-pose error puts it — a 1 cm pitch drift lifts the toe, the leg's
  support silently vanishes and 220 N of body weight walks onto the hand
  anchors (step autopsy: both hands ripped in the same step). The toe's
  actual height is the ground truth the joints must serve."""

  toe_ground_max: float = 0.3
  """Cap on the accumulated toe-planted ankle bias (rad)."""

  seat_delta_max_foot: float = 0.04
  """Foot press depth cap (m) — the arm's long leash assumed the ~125 N/m
  arm chain; the leg is many times stiffer."""

  seat_admittance_gain: float = 0.004
  seat_delta_max: float = 0.10
  """Anti-windup cap on the press depth (m). The parent's 4 cm assumed the
  climb stance's stiff bracing; the standing arm chain measures ~125 N/m
  along the wall normal (40 mm bought 5 N), so a 12 N seat needs a longer
  leash."""

  seat_lead_time_s: float = 0.4
  seat_timeout_s: float = 3.5
  seat_retry_lift: float = 0.02
  seat_retry_lift_s: float = 0.35
  seat_max_retries: int = 3

  # -- release (hand-walk-up regrips) ----------------------------------------
  release_push_start: float = 0.02
  release_push_step: float = 0.01
  release_push_period_s: float = 0.5
  release_push_max: float = 0.06
  preunload_lead_s: float = 0.5
  preunload_share: float = 0.05
  seat_share_start: float = 0.25
  seat_share_ramp_s: float = 0.5
  swing_clearance: float = 0.02

  # -- floor support ---------------------------------------------------------
  foot_lift_lead_s: float = 1.5
  """Pre-unload window before a scheduled foot lift-off (share caps AND
  the lateral CoM walk). The weight must finish walking onto the other
  supports before the foot leaves; 0.5 s left the QP infeasible when the
  schedule took the foot away while the body still stood on it, and the
  lateral walk needs ~1.3 s at the CoM rate limit."""

  foot_move_speed: float = 0.02
  """Reference foot speed (m/s) above which a grounded foot counts as the
  toe-pivot slide: toe contact only in the QP, clip tracking in the WBC."""

  # -- lift-off probing gate (continuous support handover) -------------------
  probe_lift_m: float = 0.025
  """Probe target: the lifting foot is asked to rise this much IN PLACE.
  The seat FSM verifies engagement before the clock rolls; this is the
  symmetric verification for DISENGAGEMENT — the schedule's lift instant
  used to drop the stance rows in one step while the body still stood on
  the foot, and the 0.3 s weight dump sheared the hand anchors (the
  final wall of the parameter campaign). A loaded foot cannot rise (the
  leg PD's probe force is far below the body weight), so actual foot
  height IS a force probe, no contact sensor needed."""

  probe_clear_m: float = 0.012
  """Sole clearance (m) above the floor that counts as unloaded."""

  probe_ok_steps: int = 5
  """Consecutive clear steps before the foot is declared free."""

  # -- λ feedback ------------------------------------------------------------
  lambda_trim_enable: bool = False
  """The parent's quasi-static anchor-load trim assumed the climb stance
  (all support IS anchors, load split must be honest). The standing mount
  carries weight on the feet and holds anchors with a deliberate hang
  bias, so "measured anchor load ≠ QP target" is by design — integrating
  it walks the body around and shakes the welds."""
  lambda_trim_gain: float = 0.5
  lambda_trim_max: float = 0.04
  com_offset_cap: float = 0.12

  # -- command integration ---------------------------------------------------
  windup_limit: float = 0.30

  # -- misc ------------------------------------------------------------------
  press_approach_vel: float = 0.015
  press_weight: float = 4.0e2
  press_rot_weight: float = 40.0
  press_com_assist: float = 0.025
  freeze_body_dof_reg: float = 3.0

  # -- ground mode (fewer than wall_mode_min_anchors latched) ----------------
  wall_mode_min_anchors: int = 2
  """Anchors required before the whole-body WBC takes over. Below this the
  base is NOT constraint-locked and the kinematic WBC's 'base motion
  emerges through contact' assumption is false — the least-squares solve
  happily budgets base pitch the physics executes as falling over
  (measured: posture residual 0.5→4.6 while the com residual stayed 0.05).
  On the floor the clip posture IS the statically-balanced answer, so
  ground mode passes it through and stabilizes with an ankle strategy."""

  ankle_balance_gain: float = 3.0
  """rad of ankle-pitch bias per meter of CoM wall-normal error (QP c* −
  measured). Positive error (need to move toward the wall) pitches the
  body forward."""

  ankle_balance_dgain: float = 2.0
  """Damping on the measured CoM x-velocity (rad·s/m). 0.6 left a ±4 cm
  limit cycle at ~0.5 Hz that occasionally escaped; 2.0 sits near critical
  for the ~1 s inverted-pendulum constant."""

  ankle_balance_max: float = 0.2
  """Cap on the ankle-pitch balance bias (rad)."""

  ankle_bias_const: float = 0.0
  """Diagnostic: constant ankle-pitch bias added on top of the balance
  law (system-identification probes only)."""

  single_stance_crouch_m: float = 0.09
  """CoM height drop through the single-stance window."""

  single_stance_wall_lean_m: float = 0.10
  """CoM wall-normal lean accompanying the crouch (see comment at use)."""

  single_stance_com_mix: float = 0.4
  """How far the CoM anchor's lateral component walks toward the support
  foot during single stance (0 = trust the reference, 1 = directly over
  the foot). Small on purpose: the choreography carries single stance as
  a TRIPOD (support foot + both hand anchors) and the arms cannot follow
  a 12 cm body walk — at 0.7 the far hand's lateral load climbed until
  the cone ripped while the body never moved (the anchor is a leash)."""

  com_cmd_rate: float = 0.08
  """Rate limit (m/s) on the CoM command handed to the WBC. Reference-CoM
  jumps (press station → seated-on-anchors station) otherwise reach the
  solver at full k_com speed the instant the mode flips, and the body
  pitch that executes them shears fresh anchors off in reverse."""

  wall_mode_ramp_s: float = 1.0
  """Ramp on the WBC's velocity command after entering wall mode — the
  ground→wall handover must not step the whole-body solution."""

  hang_force_target: float = 6.0
  """Rake-load target (N) the welded-hand admittance servos the measured
  anchor f_t to. Positive keeps the anchor clear of the reverse-shear
  release line (debounce 1, physical constant — the engage↔unhook churn
  of a hovering "light touch"); closed-loop beats the earlier fixed 3 cm
  down-bias, whose open-loop load drifted with body geometry to 65 N and
  primed the foot-mount rips."""

  hang_force_target_single: float = 0.0
  """Hang target (N) while a foot is lifting or lifted: the body drops a
  couple of cm the instant a leg leaves and a lightly-held hand gets
  pushed up-wall through the reverse-shear line (measured −104 N deep
  reverse). Pre-tensioning the anchors buys the margin."""

  hang_release_cap: float = 55.0
  """Per-anchor rake-load ceiling (N) of the single-stance banded ratchet:
  inside [target_single, cap] the transfer holds, above it the arm sheds
  load."""

  hang_admittance_gain: float = 0.004
  """δ̇ per Newton of hang-force error (m/s/N), the seat admittance's
  constant."""

  hang_delta_max: float = 0.08
  """Cap on the hang admittance travel (m), symmetric."""

  lateral_relax_deadband: float = 5.0
  """|f_l| (N) a welded hand tolerates before the lateral admittance lets
  the arm follow. The anchor's lateral direction is otherwise RIGID — a
  leash: the CoM's lateral walk (single-stance weight transfer) never
  executed because the arms pinned the body midline, and the lift-off
  roll then drove the anchors out of the cone (|f_l| 15-16 vs the
  0.3·f_t budget). Inside the deadband the anchor still contributes its
  small lateral share."""

  lateral_relax_gain: float = 0.004
  lateral_relax_max: float = 0.06

  tension_relax_deadband: float = 2.0
  """Pull-off tension (N) a welded hand tolerates before the normal-axis
  admittance walks the arm toward the wall. Third softened cone face:
  after the lateral axis, the lift-off rips moved to the tension line
  (f_n +20-24 vs the f_t/4 budget). Compression is never relaxed — it
  rides the anchor for free."""

  tension_relax_gain: float = 0.004
  tension_relax_max: float = 0.06

  torque_relax_gain: float = 0.12
  """rad/s per N·m of measured anchor moment: the welded arm slowly turns
  back down the twist it is putting on the pad. The per-unit moment
  billing rips a locked pad at single-digit FORCES when the arm wrings a
  few N·m through it (unit capacity is F0/12), so the moment must be
  actively bled, not just avoided."""

  torque_relax_max: float = 0.25
  """Cap on the accumulated torque-relax joint offset (rad, per joint)."""

  slow_segments: tuple[str, ...] = (
    "left_foot_mount",
    "right_foot_mount",
    "hand_up_left",
    "hand_up_right",
  )
  """Segments whose clock runs at ``slow_factor``: the dynamic transfers
  (foot lifts, hand regrips) are choreographed at 1.6 s and the
  quasi-static controller cannot move the support set that fast. The
  teacher rollout has no env-clock bookkeeping to stay in sync with —
  slowing the schedule is the cheap lever."""

  slow_factor: float = 0.5

  limb_hold_blend_tau_s: float = 0.5
  """A welded arm's hold pose blends toward the clip reference with this
  time constant. The engage-instant snapshot protects the fresh weld from
  the press-bias drop, but it must dissolve as fast as the CoM command
  walks to the reference station: a 2 s blend against the 0.08 m/s CoM
  rate loaded the anchors ~17 N per cm of body motion (measured 50 N
  within 6 steps). The snapshot differs from the reference mostly by the
  press-depth bias, so a fast blend is a gentle press release."""

  posture_integral_gain: float = 1.5
  """1/s integrator on (posture target − measured q) in ground mode: the
  model-free gravity-sag compensator. The floating-base qfrc_bias is NOT
  usable for this — its joint components carry only the DISTAL subtree's
  gravity (the foot, for an ankle), while a standing joint holds the
  proximal side; the closed-form feedforward built from it kicked the
  robot over forward (sysid probe). The integrator finds each joint's
  true holding bias without a support model."""

  posture_integral_max: float = 0.35
  """Cap on the per-joint integral bias (rad). At knee kp=100 this funds
  35 N·m of holding torque — the hand_place end pose (arms overhead at the
  wall) buckled the knees through the earlier 0.2 rad budget (root_z
  0.71 → 0.32 vertical collapse at frame ~460)."""
  qp_cfg: MountQpCfg = field(default_factory=MountQpCfg)


@dataclass
class EventRecord:
  t: float
  frame: int
  ee: str
  kind: str
  detail: str


class MountScriptedBase:
  """Drives one env of the climb-transition task through the mount."""

  def __init__(
    self,
    env,
    cfg: MountBaseCfg | None = None,
    gains: MountWbcGains | None = None,
  ) -> None:
    self.cfg = cfg or MountBaseCfg()
    self.env = env
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    self.u = u
    self.dt = float(u.step_dt)
    self.motion = u.command_manager.get_term("climb_motion")
    self.aff = u.command_manager.get_term("microspine_affordance")
    self.robot = u.scene["robot"]
    self.clip = MountClip(self.motion.cfg.motion_file)
    self.num_ee = len(self.clip.ee_names)
    self.ee_names = self.clip.ee_names

    mcfg = self.aff._mcfg
    if tuple(mcfg.ee_body_names) != self.clip.ee_names:
      raise ValueError(
        f"EE order mismatch: affordance {mcfg.ee_body_names} vs clip "
        f"{self.clip.ee_names}."
      )
    from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
      PHYBOT_C2_ACTION_SCALE,
      PHYBOT_C2_POLICY_JOINT_NAMES,
    )

    self.policy_joint_names = PHYBOT_C2_POLICY_JOINT_NAMES
    self.action_scale = PHYBOT_C2_ACTION_SCALE
    self.wbc = MountKinematicWbc(
      u.sim.mj_model,
      mcfg,
      self.policy_joint_names,
      foot_body_names=_FOOT_BODY_NAMES,
      gains=gains,
    )

    ids, names = self.robot.find_joints(
      list(self.policy_joint_names), preserve_order=True
    )
    if tuple(names) != tuple(self.policy_joint_names):
      raise ValueError("Policy joints failed to map onto the robot in order.")
    self._policy_ids = torch.tensor(ids, dtype=torch.long, device=u.device)

    # Wall basis: the mount teacher is 90°-specific.
    inward = self.aff._wall_inward_normal()[0].cpu().numpy()
    if not np.allclose(inward, [1.0, 0.0, 0.0], atol=1.0e-4):
      raise ValueError(f"Mount base expects a vertical wall; inward={inward}.")
    self.wall_inward = np.array([1.0, 0.0, 0.0])
    self.n_out = -self.wall_inward
    self.wall_x = self.clip.wall_surface_x

    self._face_local = np.asarray(mcfg.face_local_axis, dtype=float)
    self._rake_local = np.asarray(
      mcfg.limit_surface.ee_rake_local_axis, dtype=float
    )
    self._clip_to_policy = np.array(
      [self.clip.joint_names.index(n) for n in self.policy_joint_names],
      dtype=int,
    )

    # Reference CoM per clip frame (shadow-FK batch at init). This is the
    # QP's CoM regularizer anchor: on floor support the moment equality
    # leaves c free over the whole support polygon, so anchoring at the
    # MEASURED CoM is zero-stiffness self-tracking — the first probe fell
    # backwards at t=2.8 s while every solve stayed "feasible".
    self.ref_com = np.zeros((self.clip.frames, 3))
    for f in range(self.clip.frames):
      self.ref_com[f] = self.wbc.reference_com(
        self.clip.root_pos[f],
        self.clip.root_quat[f],
        self.clip.joint_pos[f][self._clip_to_policy],
      )

    self._ankle_pitch_ids = np.array(
      [
        self.policy_joint_names.index(n)
        for n in ("left_ankle_pitch", "right_ankle_pitch")
      ],
      dtype=int,
    )
    self._ankle_roll_ids = np.array(
      [
        self.policy_joint_names.index(n)
        for n in ("left_ankle_roll", "right_ankle_roll")
      ],
      dtype=int,
    )

    # Per-joint PD stiffness (the sim-PD table in phybot_c2_constants):
    # gravity sag feedforward divides the shadow's bias force by these.
    def _kp_for(name: str) -> float:
      if "hip" in name or "knee" in name:
        return 100.0
      if "waist" in name:
        return 20.0
      return 50.0  # ankle / shoulder / elbow

    self._joint_kp = np.array(
      [_kp_for(n) for n in self.policy_joint_names]
    )

    # Per-EE own-limb joint masks (policy-joint order) for the press channel.
    self._limb_masks: list[np.ndarray] = []
    for name in self.ee_names:
      side = "left" if name.startswith("left") else "right"
      if "hand" in name:
        keys = (f"{side}_shoulder", f"{side}_elbow")
      else:
        keys = (f"{side}_hip", f"{side}_knee", f"{side}_ankle")
      self._limb_masks.append(
        np.array(
          [any(j.startswith(k) for k in keys) for j in self.policy_joint_names]
        )
      )

    self.reset()

  # -- lifecycle -------------------------------------------------------------

  def reset(self) -> None:
    """Reset the CONTROL state for a fresh episode; the wall-clock log and
    event list span the whole rollout (a fall must not erase evidence)."""
    if not hasattr(self, "t"):
      self.t = 0.0
      self.total_steps = 0
      self.freeze_steps = 0
      self.events: list[EventRecord] = []
      self.log: dict[str, list] = {
        key: []
        for key in (
          "t",
          "frame",
          "segment",
          "freeze",
          "engaged",
          "sched",
          "fog_sched",
          "stance",
          "lambda_star_anchor",
          "lambda_meas",
          "ground_force_z",
          "margin",
          "qp_margin",
          "qp_feasible",
          "com",
          "com_target",
          "seat_delta",
          "press_force",
          "spine_fraction",
          "residuals",
          "root_z",
          "mode",
          "ankle_bias",
          "knee_err",
          "pad_pos",
          "eff_load",
          "moment",
          "deltas",
        )
      }
    self.frame_f = 0.0
    self.q_d: np.ndarray | None = None
    self.freeze = False
    self.qp_sol: MountQpSolution | None = None
    self.com_trim = np.zeros(3)
    self.seat_state = ["idle"] * self.num_ee
    self.seat_delta = np.zeros(self.num_ee)
    self.seat_timer = np.zeros(self.num_ee)
    self.seat_retries = np.zeros(self.num_ee, dtype=int)
    self.release_timer = np.zeros(self.num_ee)
    self.share_ramp_t = np.full(self.num_ee, -1.0)
    self._last_segment = ""
    self._prev_com_xy: np.ndarray | None = None
    self._ankle_bias = 0.0
    self._ankle_roll_bias = 0.0
    self._posture_int = np.zeros(len(self.policy_joint_names))
    self._press_ema = np.zeros(self.num_ee)
    self._limb_hold: dict[int, np.ndarray] = {}
    self._com_cmd_lp: np.ndarray | None = None
    self._wall_mode_since: float | None = None
    self._hang_delta = np.zeros(self.num_ee)
    self._lat_delta = np.zeros(self.num_ee)
    self._tens_delta = np.zeros(self.num_ee)
    self._foot_free = [False, False]
    self._probe_ok = [0, 0]
    self._toe_bias = np.zeros(2)
    self._hold_com_snap: np.ndarray | None = None
    self._swing_com_snap: np.ndarray | None = None
    self._torque_relax = np.zeros(
      (self.num_ee, len(self.policy_joint_names))
    )

  def _event(self, ee: int | None, kind: str, detail: str = "") -> None:
    name = self.ee_names[ee] if ee is not None else "-"
    self.events.append(
      EventRecord(self.t, int(self.frame_f), name, kind, detail)
    )

  # -- helpers ---------------------------------------------------------------

  def _engage_frame_cols(self, pad_rot: np.ndarray) -> np.ndarray:
    """World columns [t̂, l̂, n̂_out] mirroring the affordance's default
    gravity-rake frame (t̂ = gravity projected onto the wall plane; at 90°
    with a near-flat pad both conventions coincide)."""
    raw = np.array([0.0, 0.0, -1.0])
    t = raw - (raw @ self.n_out) * self.n_out
    t_hat = t / np.linalg.norm(t)
    l_hat = np.cross(self.n_out, t_hat)
    return np.stack([t_hat, l_hat, self.n_out], axis=1)

  def _flat_rot(self) -> np.ndarray:
    return flat_on_wall_rotation(
      self._face_local,
      self._rake_local,
      self.wall_inward,
      np.array([0.0, 0.0, -1.0]),
    )

  def _press_force_target(self, engaged: np.ndarray) -> float:
    if engaged.any():
      return self.cfg.press_force_braced
    return self.cfg.press_force_free

  # -- main step -------------------------------------------------------------

  def step_action(self) -> torch.Tensor:
    """Compute the (1, J) action for the current env state."""
    cfg = self.cfg
    clip = self.clip

    # ---- live state into the shadow ----
    joint_pos_t = self.robot.data.joint_pos[0]
    joint_names = list(self.robot.joint_names)
    joint_pos = {n: float(joint_pos_t[i]) for i, n in enumerate(joint_names)}
    root_pos = self.robot.data.root_link_pos_w[0].cpu().numpy().astype(float)
    root_quat = self.robot.data.root_link_quat_w[0].cpu().numpy().astype(float)
    self.wbc.sync(root_pos, root_quat, joint_pos)

    engaged = self.aff.engaged[0].cpu().numpy().astype(bool)
    press_force = self.aff.pad_press_force[0].cpu().numpy()
    fraction = self.aff.spine_fraction[0].cpu().numpy()
    anchor_tnl = self.aff.anchor_load_tnl[0].cpu().numpy()
    moment_tnl = self.aff.anchor_moment_tnl[0].cpu().numpy()
    eff_tnl = self.aff.anchor_effective_load_tnl[0].cpu().numpy()
    margin = self.aff.hold_margin[0].cpu().numpy()
    forced = self.aff.forced_release_this_step[0].cpu().numpy().astype(bool)
    reasons = self.aff.release_reason[0].cpu().numpy()
    just_engaged = self.aff.engaged_this_step[0].cpu().numpy().astype(bool)
    seat_scale = self.aff._seat_scale[0].cpu().numpy()
    gravity = np.array([0.0, 0.0, -9.81])

    # ---- schedule at the gated clock ----
    frame = int(min(self.frame_f, clip.frames - 1))
    sched = clip.ee_engaged[frame]
    fog_sched = clip.foot_on_ground[frame]
    t_rise = clip.t_to_rise[frame]
    t_lift = clip.t_to_foot_lift[frame]
    segment = clip.segment_of(frame)
    if segment != self._last_segment:
      self._event(None, "segment", segment)
      self._last_segment = segment

    # ---- transition events from the affordance ----
    for h in range(self.num_ee):
      if just_engaged[h]:
        self._event(
          h,
          "engage",
          f"fraction={fraction[h]:.2f} press={press_force[h]:.1f}N "
          f"retries={self.seat_retries[h]}",
        )
        self.seat_state[h] = "idle"
        self.seat_delta[h] = 0.0
        self.seat_timer[h] = 0.0
        self.seat_retries[h] = 0
        self.share_ramp_t[h] = self.t
        # Freeze the limb's command at the weld pose: dropping the press
        # depth bias / direct-play posture on the same step yanks the limb
        # back through the weld and rips the fresh anchor within steps.
        # (The old "never freeze a leg" rule belonged to the swing-lift
        # semantic; under toe-drag the weld pose — shin on wall, toe
        # planted — is exactly the support configuration to hold.)
        self._limb_hold[h] = self.wbc.joint_pos().copy()
      if forced[h]:
        kind = "release_free" if reasons[h] == 1 else "release_forced"
        self._event(
          h,
          kind,
          f"reason={int(reasons[h])} load={anchor_tnl[h].round(1).tolist()}",
        )
        self.release_timer[h] = 0.0
        self._limb_hold.pop(h, None)

    # ---- floor support classification ----
    # stance foot: scheduled on the floor AND the clip holds it still.
    # pivot foot: scheduled on the floor but sliding (toe pivot) — toe
    # contact only in the QP, clip tracking in the WBC.
    # probing foot: scheduled OFF the floor but not yet VERIFIED unloaded —
    # the lift-off gate. The schedule used to drop the stance rows in one
    # step while the body still stood on the foot; now the foot must
    # actually rise off the floor (a force probe: a loaded foot cannot)
    # before the support handover completes and the clock rolls on.
    stance = [False, False]
    pivot = [False, False]
    probing = [False, False]
    for f in range(2):
      if fog_sched[f]:
        self._foot_free[f] = False
        self._probe_ok[f] = 0
        if clip.foot_ref_speed[frame, f] <= cfg.foot_move_speed:
          stance[f] = True
        else:
          pivot[f] = True
      elif not engaged[_FOOT_EE_INDICES[f]]:
        # TOE-DRAG (the choreography's actual foot-mount semantic): the
        # foot's spine pad is on the SHIN, not the sole — the reference
        # pad only travels 15 cm forward / 8 cm up from stand to anchor.
        # The physical move is heel-up, toe on the floor, knee forward
        # pressing the shin pad to the wall; the toe keeps carrying load
        # the whole way, so the support handover is naturally continuous
        # and the earlier lift-probe gate asked for a move (a full sole
        # lift) the choreography never contains. Treat it exactly like
        # the right foot's scripted toe pivot.
        pivot[f] = True

    # ---- pending transitions & clock gate ----
    seat_pending = sched & ~engaged
    release_pending = ~sched & engaged
    qp_ok = self.qp_sol.feasible if self.qp_sol is not None else True
    was_frozen = self.freeze
    # "该挂未挂" freezes; "该松未松" does not (waiting for a foot the body
    # still stands on deadlocks — parent's rule). A PROBING foot freezes:
    # the weight transfer is verifiably unfinished and the choreography
    # ahead assumes it is. The acyclic clock clamps at the last frame.
    self.freeze = bool(seat_pending.any() or not qp_ok or any(probing))
    if self.freeze and not was_frozen:
      self._event(
        None,
        "freeze",
        f"seat={list(np.flatnonzero(seat_pending))} "
        f"rel={list(np.flatnonzero(release_pending))} "
        f"probe={probing} qp_ok={qp_ok}",
      )
    if not self.freeze:
      rate = (
        cfg.slow_factor if segment in cfg.slow_segments else 1.0
      )
      self.frame_f = min(
        self.frame_f + self.dt * clip.fps * rate, clip.frames - 1
      )
    else:
      self.freeze_steps += 1

    # ---- seat FSM (admittance press) ----
    swing_targets: dict[int, SwingTarget] = {}
    press_lock = [False] * self.num_ee
    flat_rot_default = self._flat_rot()
    for h in range(self.num_ee):
      foot_slot = (
        _FOOT_EE_INDICES.index(h) if h in _FOOT_EE_INDICES else None
      )
      on_floor = foot_slot is not None and (
        stance[foot_slot] or pivot[foot_slot]
      )
      if engaged[h]:
        continue
      press_armed = bool(seat_pending[h]) or t_rise[h] <= cfg.seat_lead_time_s
      if (
        on_floor
        and foot_slot is not None
        and pivot[foot_slot]
        and not press_armed
      ):
        # Toe-pivot / toe-drag slide: track the clip's pad path (the
        # physics keeps the toe on the floor, still carrying load). Once
        # the seat window arms, the press FSM takes the limb — with the
        # toe still planted, which IS the choreography's brace posture.
        swing_targets[h] = SwingTarget(
          pos_w=clip.ee_pos[frame, h].copy(), rot_w=None, point="origin"
        )
        continue
      if on_floor and not press_armed:
        continue  # parked stance foot: zero-velocity rows own it
      if not press_armed:
        # Free swing: world-frame clip reference (the mount clip stores
        # world EE paths and play spawns on the reference station). The
        # clip's ee_positions track the EE body ORIGIN, not the pad center
        # (~22 cm apart on the hand). A scheduled-OFF pad also must stay
        # off the wall (clearance clamp) or it re-latches on cooldown.
        p_ref = clip.ee_pos[frame, h].copy()
        origin_standoff = float((flat_rot_default @ self.wbc.pad_local[h])[0])
        p_ref[0] = min(
          p_ref[0], self.wall_x - cfg.swing_clearance - origin_standoff
        )
        swing_targets[h] = SwingTarget(
          pos_w=p_ref, rot_w=flat_rot_default, point="origin"
        )
        if self.seat_state[h] != "idle":
          self.seat_state[h] = "idle"
          self.seat_delta[h] = 0.0
          self.seat_timer[h] = 0.0
        continue

      # Armed: force-controlled seating toward the scheduled anchor.
      if self.seat_state[h] == "idle":
        self.seat_state[h] = "press"
        self.seat_timer[h] = 0.0
        self.seat_delta[h] = 0.0
        self._event(h, "seat_start", f"t_rise={t_rise[h]:.2f}s seg={segment}")
      self.seat_timer[h] += self.dt
      if (
        self.seat_state[h] == "press"
        and self.wbc.pad_pose(h)[0][0] > self.wall_x - 0.003
        and h not in _FOOT_EE_INDICES
      ):
        # Hands only: the foot press plays the reference posture directly
        # (below) — freezing the whole leg with the toe still planted let
        # body momentum ram the shin into the wall (40–288 N).
        press_lock[h] = True

      # Seat target: the scheduled anchor position (clip ee_anchor_positions
      # at the engage edge), addressed at the PAD CENTER in wall coords.
      # Past the limb's last scheduled rise (re-seat of a ripped anchor,
      # or a foot after its single rise) t_rise is +inf: aim at the
      # current frame's anchor.
      if np.isfinite(t_rise[h]):
        rise_frame = frame + int(round(t_rise[h] * clip.fps))
      else:
        rise_frame = frame
      rise_frame = int(min(max(rise_frame, 0), clip.frames - 1))
      anchor = clip.anchor_pos[rise_frame, h]
      if not np.isfinite(anchor).all() or np.linalg.norm(anchor) < 1.0e-6:
        anchor = clip.pad_pos[rise_frame, h]
      if self.seat_state[h] == "retry_lift":
        if self.seat_timer[h] >= cfg.seat_retry_lift_s:
          self.seat_state[h] = "press"
          self.seat_timer[h] = 0.0
          self.seat_delta[h] = 0.0
        target_x = self.wall_x - cfg.seat_retry_lift
      else:  # press
        pad_x = self.wbc.pad_pose(h)[0][0]
        near = press_force[h] > 0.5 or pad_x > self.wall_x - 0.005
        if near:
          if h in _FOOT_EE_INDICES:
            # The shin/leg chain is far stiffer than the wristless arm:
            # the arm's 0.10 m leash overshot the foot press to 60 N and
            # the wall pushback ripped both hand anchors (158/81 N)
            # before the foot even welded. Lower target, shorter leash.
            f_star = cfg.press_force_foot
            delta_cap = cfg.seat_delta_max_foot
          else:
            f_star = self._press_force_target(engaged)
            delta_cap = cfg.seat_delta_max
          err = f_star - press_force[h]
          self.seat_delta[h] = float(
            np.clip(
              self.seat_delta[h] + cfg.seat_admittance_gain * err * self.dt,
              0.0,
              delta_cap,
            )
          )
        else:
          self.seat_delta[h] *= 0.9
        target_x = self.wall_x
        if (
          self.seat_state[h] == "press"
          and self.seat_timer[h] >= cfg.seat_timeout_s
        ):
          self.seat_retries[h] += 1
          if self.seat_retries[h] >= cfg.seat_max_retries:
            self._event(
              h,
              "seat_failed",
              f"retries={self.seat_retries[h]} press={press_force[h]:.1f}N "
              f"fraction={fraction[h]:.2f}",
            )
            self.seat_state[h] = "press"
            self.seat_timer[h] = 0.0
            self.seat_retries[h] = 0
          else:
            self.seat_state[h] = "retry_lift"
            self.seat_timer[h] = 0.0
            self._event(h, "seat_retry", f"n={self.seat_retries[h]}")
      if h in _FOOT_EE_INDICES:
        # Foot seat: no cartesian pull — the leg plays the reference
        # posture directly (the clip frames ARE the statically-balanced
        # shin-press choreography, 2 mm rake stroke included) and the δ
        # channel adds the last few millimeters of contact force.
        continue
      target = np.array([target_x, anchor[1], anchor[2]])
      ff = None
      if self.seat_state[h] == "press" and press_force[h] < 0.5:
        ff = self.wall_inward * cfg.press_approach_vel
      swing_targets[h] = SwingTarget(
        pos_w=target,
        rot_w=flat_rot_default,
        vel_ff_w=ff,
        weight=cfg.press_weight,
        rot_weight=cfg.press_rot_weight,
      )

    # ---- release FSM (hand-walk-up regrips) + share caps ----
    anchor_share = np.full(self.num_ee, np.nan)
    wbc_latched = engaged.copy()
    n_support = (
      int(engaged.sum())
      + stance.count(True)
      + pivot.count(True)
      + probing.count(True)
    )
    for h in range(self.num_ee):
      if (
        not self.freeze
        and engaged[h]
        and sched[h]
        and self._t_to_fall(frame, h) <= cfg.preunload_lead_s
        and n_support - 1 >= 2
      ):
        anchor_share[h] = cfg.preunload_share
      if release_pending[h]:
        anchor_share[h] = cfg.preunload_share
        if self.release_timer[h] == 0.0:
          self._event(h, "unhook_start", f"seg={segment}")
        self.release_timer[h] += self.dt
        rung = int(self.release_timer[h] / cfg.release_push_period_s)
        push = min(
          cfg.release_push_start + rung * cfg.release_push_step,
          cfg.release_push_max,
        )
        pos, rot = self.wbc.pad_pose(h)
        t_hat = self._engage_frame_cols(rot)[:, 0]
        target = pos - push * t_hat  # up-wall = against the rake
        wbc_latched[h] = False
        swing_targets[h] = SwingTarget(pos_w=target, rot_w=None)
      elif not release_pending[h] and self.release_timer[h] > 0.0:
        self.release_timer[h] = 0.0
      if engaged[h] and self.share_ramp_t[h] >= 0.0:
        age = self.t - self.share_ramp_t[h]
        if age < cfg.seat_share_ramp_s:
          frac_r = age / cfg.seat_share_ramp_s
          anchor_share[h] = np.nanmin(
            [
              anchor_share[h],
              cfg.seat_share_start + frac_r * (1.0 - cfg.seat_share_start),
            ]
          )
        else:
          self.share_ramp_t[h] = -1.0

    # ---- contact QP (mixed anchors + floor) ----
    units: list[ContactUnit] = []
    for h in range(self.num_ee):
      if not engaged[h]:
        continue
      pos, rot = self.wbc.pad_pose(h)
      units.append(
        ContactUnit(
          name=f"wang_{self.ee_names[h]}",
          kind="wang",
          pos_w=pos,
          frame_w=self._engage_frame_cols(rot),
          cap_scale=float(seat_scale[h]),
          share_cap=float(anchor_share[h]),
        )
      )
    for f in range(2):
      if not (stance[f] or pivot[f] or probing[f]):
        continue
      # Pre-unload the whole foot ahead of its scheduled lift-off; a
      # probing foot is still physically on the floor and keeps its
      # (capped) contact units until the probe verifies it free.
      foot_share = (
        cfg.preunload_share
        if probing[f] or t_lift[f] <= cfg.foot_lift_lead_s
        else float("nan")
      )
      points = self.wbc.foot_sole_points_w(f)
      chosen = [points[1]] if pivot[f] else points  # toe only while pivoting
      tags = ["toe"] if pivot[f] else ["heel", "toe"]
      for tag, p in zip(tags, chosen):
        units.append(
          ContactUnit(
            name=f"gnd_{_FOOT_BODY_NAMES[f]}_{tag}",
            kind="ground",
            pos_w=np.asarray(p),
            frame_w=np.eye(3),
            share_cap=foot_share,
          )
        )
    com_meas = self.wbc.com()
    # CoM regularizer anchor. Base case: the clip's reference CoM (frozen
    # frames give station keeping WITH stiffness for free). While BOTH
    # hands hold anchors, the anchor freezes at the both-welded snapshot
    # instead: the wristless arm renders any body walk as pad torque on
    # the locked welds, and the per-unit moment billing rips them at
    # single-digit forces (measured reason-4 releases at 4–10 N). The
    # hold choreography stands still anyway; the lateral pre-lift walk
    # below still applies on top.
    both_hands = bool(engaged[0] and engaged[1])
    if both_hands:
      if self._hold_com_snap is None:
        self._hold_com_snap = self.ref_com[frame].copy()
      com_anchor = self._hold_com_snap.copy()
    else:
      self._hold_com_snap = None
      com_anchor = self.ref_com[frame].copy()
    if seat_pending.any():
      com_anchor += cfg.press_com_assist * self.wall_inward
    # Single-stance: pull the anchor's lateral component over the support
    # foot. The reference (and the raw QP) lean on the hand anchors to
    # carry the roll moment, but anchor force is REACTIVE — it only shows
    # up once the body already tips, and the tip rips the cone first
    # (probe: both hands ultimate at f_l 10–16 N, body rolling +y).
    # Under the toe-drag semantic both feet keep ground contact through
    # their mounts (the toe carries load until the shin pad welds), so a
    # true single-stance window never occurs and the transfer surgery
    # (crouch / wall lean / hang ratchet) stays dormant.
    n_stance = stance.count(True) + pivot.count(True)
    f_sup = None
    if engaged.any() and n_stance == 1:
      f_sup = 0 if (stance[0] or pivot[0]) else 1
    if f_sup is not None:
      lift_f = 1 - f_sup
      if (
        self._foot_free[lift_f]
        and not engaged[_FOOT_EE_INDICES[lift_f]]
      ):
        # Swing phase: WHOLE-BODY QUIET. The probe verified the handover
        # by settling most of the weight onto the anchors; a moving body
        # under that load wrings pad torque faster than the relax loop
        # can bleed it (per-unit rips at 104 N in-cone). Freeze the CoM
        # command at the free-instant snapshot — the only moving chain is
        # the swing leg, and the leg does not twist the arms.
        if self._swing_com_snap is None:
          self._swing_com_snap = com_anchor.copy()
          self._swing_com_snap[1] = (
            (1.0 - cfg.single_stance_com_mix) * com_anchor[1]
            + cfg.single_stance_com_mix
            * float(self.wbc.foot_sole_points_w(f_sup)[0][1])
          )
          self._swing_com_snap[2] -= cfg.single_stance_crouch_m
          self._swing_com_snap[0] += 0.4 * cfg.single_stance_wall_lean_m
        com_anchor = self._swing_com_snap.copy()
      else:
        self._swing_com_snap = None
        foot_y = float(self.wbc.foot_sole_points_w(f_sup)[0][1])
        com_anchor[1] = (
          (1.0 - cfg.single_stance_com_mix) * com_anchor[1]
          + cfg.single_stance_com_mix * foot_y
        )
        # Crouch through the transfer: a lower CoM shortens the roll lever,
        # turns the arm load direction toward pure hang (the strong region
        # of the Wang surface), and softens the leg's weight-catch step.
        # The crouch must LEAN INTO the wall as it sinks — a straight-down
        # squat under high anchors straightens the arms and turns the
        # transferred weight into pull-off tension (measured 47 N).
        com_anchor[2] -= cfg.single_stance_crouch_m
        com_anchor[0] += cfg.single_stance_wall_lean_m
    else:
      self._swing_com_snap = None
    # Low-pass the press pushback before it reaches the balance (τ = 0.3 s):
    # the raw contact force chatters at ~5 Hz against the compliant arm and
    # feeding it straight into the QP made the CoM target — and the body —
    # chatter with it, which kept knocking the pad off its own contact.
    alpha = self.dt / 0.3
    self._press_ema += alpha * (press_force - self._press_ema)
    # No hard zero on engage: the latched pad's press reading drops to 0 on
    # its own (contacts disable while welded) and the EMA decays smoothly.
    # Zeroing it on the engage step yanked the QP's balance input, the body
    # sprang back upright and the fresh anchor sheared off in reverse
    # within 3 steps (measured f_t −48 N).
    extra_f = np.zeros(3)
    extra_m = np.zeros(3)
    for h in np.flatnonzero(~engaged):
      if self._press_ema[h] > 0.1:
        pos, _ = self.wbc.pad_pose(h)
        f_c = self._press_ema[h] * self.n_out
        extra_f += f_c
        extra_m += np.cross(pos, f_c)
    self.qp_sol = solve_mount_qp(
      units,
      mass=self.wbc.robot_mass,
      gravity_w=gravity,
      com_meas=com_meas,
      cfg=cfg.qp_cfg,
      warm_start=self.qp_sol,
      extra_force_w=extra_f,
      extra_moment_w=extra_m,
      com_anchor=com_anchor,
    )

    # ---- λ-error → CoM integral trim (quiet anchored stance only) ----
    w_vec = self.wbc.robot_mass * gravity
    w_norm2 = float(w_vec @ w_vec)
    share_active = np.isfinite(anchor_share).any() or any(
      np.isfinite(u.share_cap) for u in units
    )
    stance_quiet = (
      cfg.lambda_trim_enable
      and self.qp_sol.feasible
      and not self.freeze
      and not share_active
      and engaged.sum() >= 2
      and w_norm2 > 1.0
    )
    if stance_quiet:
      m_err = np.zeros(3)
      k = 0
      for u in self.qp_sol.unit_names:
        if u.startswith("wang_"):
          h = self.ee_names.index(u[len("wang_") :])
          pos, rot = self.wbc.pad_pose(h)
          frame_cols = self._engage_frame_cols(rot)
          f_meas = frame_cols @ anchor_tnl[h]
          f_star = frame_cols @ self.qp_sol.lambda_local[k]
          m_err += np.cross(pos, f_meas - f_star)
        k += 1
      implied_c_err = np.cross(w_vec, m_err) / w_norm2
      self.com_trim += -cfg.lambda_trim_gain * self.dt * implied_c_err
      norm = float(np.linalg.norm(self.com_trim))
      if norm > cfg.lambda_trim_max:
        self.com_trim *= cfg.lambda_trim_max / norm

    # One CoM path for frozen and rolling alike: the QP's c* is anchored at
    # the reference CoM (station keeping with stiffness) and already leans
    # with any press pushback through the balance equality.
    com_cmd = self.qp_sol.com_target + self.com_trim
    com_offset = com_cmd - com_meas
    offset_norm = float(np.linalg.norm(com_offset))
    if offset_norm > cfg.com_offset_cap:
      com_cmd = com_meas + com_offset * (cfg.com_offset_cap / offset_norm)
    # Rate-limit the command so reference-station jumps arrive as a walk,
    # not a yank.
    if self._com_cmd_lp is None:
      self._com_cmd_lp = com_meas.copy()
    step_cap = cfg.com_cmd_rate * self.dt
    delta = com_cmd - self._com_cmd_lp
    dn = float(np.linalg.norm(delta))
    if dn > step_cap:
      delta *= step_cap / dn
    self._com_cmd_lp = self._com_cmd_lp + delta
    com_cmd = self._com_cmd_lp.copy()

    # Ankle balance law (both modes — any stance foot keeps stabilizing the
    # CoM; the whole-body solve has no other fast roll/pitch authority).
    # Error against the RAW c* (the offset-capped com_cmd chases the
    # measurement and hides a runaway). Signs calibrated by sysid probes.
    com_vel = (
      np.zeros(2)
      if self._prev_com_xy is None
      else (com_meas[:2] - self._prev_com_xy) / self.dt
    )
    self._prev_com_xy = com_meas[:2].copy()
    balance_tgt = self.qp_sol.com_target[:2] + self.com_trim[:2]
    err_x = float(balance_tgt[0] - com_meas[0])
    err_y = float(balance_tgt[1] - com_meas[1])
    self._ankle_bias = float(
      np.clip(
        -cfg.ankle_balance_gain * err_x
        + cfg.ankle_balance_dgain * float(com_vel[0])
        + cfg.ankle_bias_const,
        -cfg.ankle_balance_max,
        cfg.ankle_balance_max,
      )
    )
    self._ankle_roll_bias = float(
      np.clip(
        cfg.ankle_balance_gain * err_y
        - cfg.ankle_balance_dgain * float(com_vel[1]),
        -cfg.ankle_balance_max,
        cfg.ankle_balance_max,
      )
    )

    # ---- command synthesis: ground mode vs wall mode ----
    # Below wall_mode_min_anchors the base is not constraint-locked and the
    # kinematic WBC's least-squares budget for base motion is physically a
    # fall (measured). On the floor the clip posture is the statically
    # balanced answer: pass it through and stabilize with an ankle strategy
    # driven by the QP's CoM target (which feels press pushback).
    mode = (
      "wall"
      if int(engaged.sum()) >= cfg.wall_mode_min_anchors
      else "ground"
    )
    posture_ref = clip.joint_pos[frame][self._clip_to_policy]
    q_now = self.wbc.joint_pos()
    jrange = self.wbc.joint_range()
    span = jrange[:, 1] - jrange[:, 0]
    soft_lo = jrange[:, 0] + 0.011 * span
    soft_hi = jrange[:, 1] - 0.011 * span

    if mode == "ground":
      self._wall_mode_since = None
      target = posture_ref.copy()
      target[self._ankle_pitch_ids] += self._ankle_bias
      target[self._ankle_roll_ids] += self._ankle_roll_bias
      # Model-free gravity-sag compensation: integrate the tracking error
      # toward the target (which already carries the balance bias, so the
      # integrator and the ankle strategy do not fight).
      self._posture_int = np.clip(
        self._posture_int
        + cfg.posture_integral_gain * (target - q_now) * self.dt,
        -cfg.posture_integral_max,
        cfg.posture_integral_max,
      )
      self.q_d = np.clip(target + self._posture_int, soft_lo, soft_hi)
      # Welded limbs hold their engage-time pose — the weld owns the
      # geometry, the arm must not fight it.
      for h, hold_q in self._limb_hold.items():
        if engaged[h]:
          mask = self._limb_masks[h]
          self.q_d[mask] = hold_q[mask]
      residuals: dict[str, float] = {"mode": 0.0}
    else:
      base_rot_target = quat_to_mat(clip.root_quat[frame])
      base_xy_target = clip.root_pos[frame, :2]
      dof_reg = None
      if self.freeze:
        if not hasattr(self, "_body_dof_mask"):
          arm_key = ("shoulder", "elbow")
          joint_is_arm = np.array(
            [any(k in n for k in arm_key) for n in self.policy_joint_names]
          )
          self._body_dof_mask = np.concatenate(
            [np.ones(6, dtype=bool), ~joint_is_arm]
          )
        dof_reg = np.where(self._body_dof_mask, cfg.freeze_body_dof_reg, 0.0)
      if any(press_lock):
        if dof_reg is None:
          dof_reg = np.zeros(6 + len(self.policy_joint_names))
        for h in range(self.num_ee):
          if press_lock[h]:
            dof_reg[6:] = np.where(self._limb_masks[h], 1.0e3, dof_reg[6:])
            swing_targets.pop(h, None)
      if f_sup is not None:
        lift_f = 1 - f_sup
        lift_h = _FOOT_EE_INDICES[lift_f]
        if self._foot_free[lift_f] and not engaged[lift_h]:
          # Direct-play swing leg: drop its cartesian target so the solve
          # neither drags the leg nor recruits the body chasing the foot.
          swing_targets.pop(lift_h, None)
          if dof_reg is None:
            dof_reg = np.zeros(6 + len(self.policy_joint_names))
          dof_reg[6:] = np.where(
            self._limb_masks[lift_h], 1.0e3, dof_reg[6:]
          )
      sol = self.wbc.solve(
        latched=wbc_latched,
        stance_feet=(stance[0], stance[1]),
        swing_targets=swing_targets,
        com_target=com_cmd,
        base_rot_target=base_rot_target,
        base_xy_target=base_xy_target,
        posture_ref=posture_ref,
        dof_reg=dof_reg,
      )
      self._last_sol = sol
      residuals = sol.task_residuals
      if self.q_d is None:
        self.q_d = q_now.copy()
      if self._wall_mode_since is None:
        self._wall_mode_since = self.t
      ramp = min(
        1.0, (self.t - self._wall_mode_since) / max(cfg.wall_mode_ramp_s, self.dt)
      )
      self.q_d = self.q_d + sol.qdot_j * self.dt * ramp
      self.q_d = np.clip(
        self.q_d, q_now - cfg.windup_limit, q_now + cfg.windup_limit
      )
      self.q_d = np.clip(self.q_d, soft_lo, soft_hi)
      # Welded limbs hold their engage-time pose in wall mode too — the
      # WBC's latch rows keep the whole-body solve consistent with the
      # anchors, but the arm command itself must not chase anything.
      for h, hold_q in self._limb_hold.items():
        if engaged[h] and not release_pending[h]:
          mask = self._limb_masks[h]
          self.q_d[mask] = hold_q[mask]
      # A foot mid-seat plays the clip's joint trajectory DIRECTLY: the
      # reference frames are the statically-balanced shin-press pose and
      # the WBC's cartesian pull / whole-leg press_lock both rammed the
      # shin (40–288 N).
      for f in range(2):
        h_f = _FOOT_EE_INDICES[f]
        if self.seat_state[h_f] == "press" and not engaged[h_f]:
          mask = self._limb_masks[h_f]
          self.q_d[mask] = posture_ref[mask]

    # ---- dedicated press channel (stateless bias from δ each step) ----
    q_cmd = self.q_d.copy()
    if mode == "wall":
      # Stance feet keep their ankle balance authority in wall mode — as a
      # stateless OUTPUT bias. Adding it to the integrated q_d state turned
      # a 0.1 rad bias into a 5 rad/s ramp (measured ±200 N anchor-load
      # oscillation during the two-hand hold).
      for f in range(2):
        if stance[f]:
          q_cmd[self._ankle_pitch_ids[f]] += self._ankle_bias
          q_cmd[self._ankle_roll_ids[f]] += self._ankle_roll_bias
      # Toe-planted servo: through the toe-drag and shin-press the toe's
      # ACTUAL clearance drives an ankle plantarflex bias (positive pitch
      # = heel up / toe down, calibrated from the clip: stand −0.295 →
      # deep toe-drag +1.26). See toe_ground_gain.
      for f in range(2):
        h_f = _FOOT_EE_INDICES[f]
        toe_phase = not fog_sched[f] and not engaged[h_f] and (
          pivot[f] or self.seat_state[h_f] == "press"
        )
        if toe_phase:
          clearance = max(self.wbc.toe_tip_z(f), 0.0)
          self._toe_bias[f] = float(
            np.clip(
              self._toe_bias[f] + cfg.toe_ground_gain * clearance * self.dt,
              0.0,
              cfg.toe_ground_max,
            )
          )
          q_cmd[self._ankle_pitch_ids[f]] += self._toe_bias[f]
        else:
          self._toe_bias[f] = 0.0
      # Grounded legs keep their gravity-sag integrator in wall mode too
      # (stateless output add of the shared integral state): with only the
      # WBC's weak posture row on them, the legs sag, the body sinks and
      # hangs its weight on the hand anchors (toe-drag phase measured
      # 113–146 N rips as the body settled onto the welds).
      # Only during the toe-drag phase: with both feet flat the ankle
      # strategy suffices, and this integrator running through the quiet
      # hold slowly cranked the left anchor 26 -> 122 N while the body
      # never moved a millimeter.
      leg_mask = np.zeros(len(self.policy_joint_names), dtype=bool)
      if any(pivot):
        for f in range(2):
          if stance[f] or pivot[f]:
            leg_mask |= self._limb_masks[_FOOT_EE_INDICES[f]]
      if leg_mask.any():
        # Leg/anchor load coordination: the legs push the body up ONLY
        # while the welded hands still carry honest rake load. Body
        # height is weld-locked — an unconditional integrator turns into
        # a leg-vs-anchor internal-force loop that shoves the hands
        # through the reverse-shear line (measured: alternating −30/−50 N
        # free unhooks at 3 Hz, 129 engages per run).
        hand_ft = [
          float(anchor_tnl[h, 0])
          for h in range(2)
          if engaged[h]
        ]
        if hand_ft and min(hand_ft) < 5.0:
          # Hands near the unhook line: back the legs off.
          self._posture_int[leg_mask] -= (
            cfg.posture_integral_gain * 2.0 * self.dt
          ) * np.sign(self._posture_int[leg_mask])
        else:
          self._posture_int[leg_mask] = np.clip(
            self._posture_int[leg_mask]
            + cfg.posture_integral_gain
            * (posture_ref[leg_mask] - q_now[leg_mask])
            * self.dt,
            -cfg.posture_integral_max,
            cfg.posture_integral_max,
          )
        q_cmd[leg_mask] += self._posture_int[leg_mask]
    for h in range(self.num_ee):
      if self.seat_state[h] == "press" and self.seat_delta[h] > 0.0:
        row = self.wbc.pad_axis_joint_row(h, self.wall_inward)
        row = row * self._limb_masks[h]
        denom = float(row @ row)
        if denom > 1.0e-9:
          q_cmd = q_cmd + row * (self.seat_delta[h] / denom)
      elif engaged[h] and h not in _FOOT_EE_INDICES:
        # Welded hand: servo the measured rake load to the gentle hang
        # target (see hang_force_target). Positive δ pulls down-wall.
        hang_tgt = (
          cfg.hang_force_target_single
          if f_sup is not None
          else cfg.hang_force_target
        )
        f_t_meas = float(anchor_tnl[h, 0])
        # Overload shed is ALWAYS on duty. It was once gated on the
        # single-stance window, which the toe-drag semantic never enters —
        # so the hands were crushed past 100 N during the foot press with
        # the shed asleep. Judged on the COMBINED load (tension included).
        combined = float(
          np.sqrt(
            anchor_tnl[h, 0] ** 2
            + anchor_tnl[h, 1] ** 2
            + max(anchor_tnl[h, 2], 0.0) ** 2
          )
        )
        if combined > cfg.hang_release_cap:
          err = 5.0 * (cfg.hang_release_cap - combined)
        elif f_t_meas < hang_tgt:
          err = hang_tgt - f_t_meas
        else:
          err = 0.0
        self._hang_delta[h] = float(
          np.clip(
            self._hang_delta[h] + cfg.hang_admittance_gain * err * self.dt,
            0.0,  # shedding stops at zero pull — a negative delta is an
            # active up-wall push and unhooks the anchor in reverse shear
            # (measured: both hands popped at -15/-30 N mid-hold)
            cfg.hang_delta_max,
          )
        )
        row = self.wbc.pad_axis_joint_row(h, np.array([0.0, 0.0, -1.0]))
        row = row * self._limb_masks[h]
        denom = float(row @ row)
        if denom > 1.0e-9:
          q_cmd = q_cmd + row * (self._hang_delta[h] / denom)
        # Lateral admittance: outside the deadband the arm follows, so the
        # body's lateral walk is not fighting a rigid leash.
        f_l = float(anchor_tnl[h, 1])
        excess = np.sign(f_l) * max(abs(f_l) - cfg.lateral_relax_deadband, 0.0)
        self._lat_delta[h] = float(
          np.clip(
            self._lat_delta[h]
            - cfg.lateral_relax_gain * excess * self.dt,
            -cfg.lateral_relax_max,
            cfg.lateral_relax_max,
          )
        )
        row_l = self.wbc.pad_axis_joint_row(h, np.array([0.0, -1.0, 0.0]))
        row_l = row_l * self._limb_masks[h]
        denom_l = float(row_l @ row_l)
        if denom_l > 1.0e-9:
          q_cmd = q_cmd + row_l * (self._lat_delta[h] / denom_l)
        # Tension admittance: pull-off beyond the deadband walks the arm
        # toward the wall (compression is free and never relaxed).
        tension = max(float(anchor_tnl[h, 2]), 0.0)
        exc_n = max(tension - cfg.tension_relax_deadband, 0.0)
        self._tens_delta[h] = float(
          np.clip(
            self._tens_delta[h] + cfg.tension_relax_gain * exc_n * self.dt,
            0.0,
            cfg.tension_relax_max,
          )
        )
        row_n = self.wbc.pad_axis_joint_row(h, self.wall_inward)
        row_n = row_n * self._limb_masks[h]
        denom_n = float(row_n @ row_n)
        if denom_n > 1.0e-9:
          q_cmd = q_cmd + row_n * (self._tens_delta[h] / denom_n)
        # Torque relax: turn the arm back down the twist it puts on the
        # locked pad (see torque_relax_gain).
        frame_cols = self._engage_frame_cols(None)
        m_w = frame_cols @ moment_tnl[h]
        jr = self.wbc.pad_rot_rows(h) * self._limb_masks[h][None, :]
        dq = -cfg.torque_relax_gain * (jr.T @ m_w) * self.dt
        self._torque_relax[h] = np.clip(
          self._torque_relax[h] + dq,
          -cfg.torque_relax_max,
          cfg.torque_relax_max,
        )
        q_cmd = q_cmd + self._torque_relax[h]
      elif not engaged[h]:
        self._hang_delta[h] = 0.0
        self._lat_delta[h] = 0.0
        self._tens_delta[h] = 0.0
        self._torque_relax[h] = 0.0
    q_cmd = np.clip(
      q_cmd, q_now - 2.0 * cfg.windup_limit, q_now + 2.0 * cfg.windup_limit
    )
    q_cmd = np.clip(q_cmd, soft_lo, soft_hi)

    default = (
      self.robot.data.default_joint_pos[0, self._policy_ids].cpu().numpy()
    )
    bias = (
      self.robot.data.encoder_bias[0, self._policy_ids].cpu().numpy()
      if hasattr(self.robot.data, "encoder_bias")
      else 0.0
    )
    action = (q_cmd + bias - default) / self.action_scale

    # ---- logging ----
    lam_star_anchor = np.zeros((self.num_ee, 3))
    gnd_z = 0.0
    for k, uname in enumerate(self.qp_sol.unit_names):
      if uname.startswith("wang_"):
        h = self.ee_names.index(uname[len("wang_") :])
        lam_star_anchor[h] = self.qp_sol.lambda_local[k]
      else:
        gnd_z += float(self.qp_sol.force_on_robot_w[k, 2])
    lg = self.log
    lg["t"].append(self.t)
    lg["frame"].append(frame)
    lg["segment"].append(segment)
    lg["freeze"].append(self.freeze)
    lg["engaged"].append(engaged.copy())
    lg["sched"].append(sched.copy())
    lg["fog_sched"].append(fog_sched.copy())
    lg["stance"].append((stance[0], stance[1], pivot[0], pivot[1]))
    lg["lambda_star_anchor"].append(lam_star_anchor)
    lg["lambda_meas"].append(anchor_tnl.copy())
    lg["ground_force_z"].append(gnd_z)
    lg["margin"].append(margin.copy())
    lg["qp_margin"].append(self.qp_sol.margin)
    lg["qp_feasible"].append(self.qp_sol.feasible)
    lg["com"].append(com_meas.copy())
    lg["com_target"].append(com_cmd.copy())
    lg["seat_delta"].append(self.seat_delta.copy())
    lg["press_force"].append(press_force.copy())
    lg["spine_fraction"].append(fraction.copy())
    lg["residuals"].append(residuals)
    lg["root_z"].append(float(root_pos[2]))
    lg["mode"].append(mode)
    lg["ankle_bias"].append(self._ankle_bias)
    if not hasattr(self, "_knee_ids"):
      self._knee_ids = np.array(
        [
          self.policy_joint_names.index(n)
          for n in ("left_knee", "right_knee")
        ],
        dtype=int,
      )
    lg["knee_err"].append(
      (q_now[self._knee_ids] - posture_ref[self._knee_ids]).copy()
    )
    lg["pad_pos"].append(
      np.stack([self.wbc.pad_pose(h)[0] for h in range(self.num_ee)])
    )
    lg["eff_load"].append(eff_tnl.copy())
    lg["moment"].append(moment_tnl.copy())
    lg["deltas"].append(
      np.array(
        [
          self._hang_delta[0],
          self._hang_delta[1],
          self._lat_delta[0],
          self._lat_delta[1],
          self._tens_delta[0],
          self._tens_delta[1],
          float(np.abs(self._torque_relax[0]).max()),
          float(np.abs(self._torque_relax[1]).max()),
          float(np.abs(self._posture_int).max()),
        ]
      )
    )

    self.t += self.dt
    self.total_steps += 1
    device = self.robot.data.joint_pos.device
    return torch.as_tensor(
      action, dtype=torch.float32, device=device
    ).unsqueeze(0)

  def _t_to_fall(self, frame: int, h: int) -> float:
    """Seconds until this EE's next scheduled engage-fall (inf if none)."""
    col = self.clip.ee_engaged[:, h]
    for f in range(frame + 1, self.clip.frames):
      if not col[f] and col[f - 1]:
        return (f - frame) / self.clip.fps
    return float("inf")
