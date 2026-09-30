"""Reward functions for the wall-climbing task.

Empirical tasks select 16 terms: BeyondMimic body tracking, climbing progress
and landing, contact schedule/support/release shaping, regularization and a
failure penalty. No explicit survival reward is used. Legacy joint/EE and
auxiliary terms remain available to the Wang transition configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .motion import ClimbMotionCommand


class _SphereGeomCache:
  """Contact-proxy geometry cache (legacy class name), for boxes and spheres.

  Wall pose is queried per-step from the wall entity's mocap state so that
  curriculum-driven tilts are reflected in the derived surface/normal maths.
  """
  _instance: _SphereGeomCache | None = None
  geom_ids: torch.Tensor
  sizes: torch.Tensor
  is_box: torch.Tensor
  wall_half: float
  initialized: bool = False

  @classmethod
  def get(cls, env: ManagerBasedRlEnv) -> _SphereGeomCache:
    if cls._instance is None or getattr(cls._instance, "model", None) is not env.sim.mj_model:
      cls._instance = cls()
      mj = env.sim.mj_model
      ids = []
      for ee_name in (
        "left_hand_spine",
        "right_hand_spine",
        "left_foot_spine",
        "right_foot_spine",
      ):
        for i in range(2):
          for j in range(2):
            gid = mj.geom(f"robot/{ee_name}_c_{i}_{j}").id
            ids.append(gid)
      cls._instance.geom_ids = torch.tensor(ids, dtype=torch.long, device=env.device)
      cls._instance.model = mj
      cls._instance.sizes = torch.tensor(mj.geom_size[ids], dtype=torch.float32, device=env.device)
      cls._instance.is_box = torch.tensor(mj.geom_type[ids] == mujoco.mjtGeom.mjGEOM_BOX, device=env.device)
      if not all(t in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE) for t in mj.geom_type[ids]):
        raise ValueError("Pad distance supports only box and sphere contact proxies")
      wall_geom_id = mj.geom("wall/wall_geom").id
      cls._instance.wall_half = float(mj.geom_size[wall_geom_id, 0])
      cls._instance.initialized = True
    return cls._instance

  @classmethod
  def wall_plane(cls, env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env wall body position and inner-face outward normal (world frame).

    Outward normal points AWAY from the wall body, toward the robot, so a
    point ``P`` outside the wall has ``(P - wall_pos) · n > wall_half``.
    """
    wall = env.scene["wall"]
    wall_pos = wall.data.root_link_pos_w  # (E, 3)
    quat = wall.data.root_link_quat_w  # (E, 4) wxyz
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # Rotate local (-1, 0, 0) by quat to get outward normal in world.
    # For v=(-1,0,0): nx = -(1 - 2(y^2+z^2)); ny = -2(x*y + w*z); nz = -2(x*z - w*y)
    nx = -(1.0 - 2.0 * (y * y + z * z))
    ny = -2.0 * (x * y + w * z)
    nz = -2.0 * (x * z - w * y)
    normal = torch.stack([nx, ny, nz], dim=-1)
    return wall_pos, normal

  @classmethod
  def surface_distances(cls, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Per-proxy closest-surface-to-wall signed distance. (E, 16).
    Positive=gap (contact surface hasn't reached wall yet).
    Negative=penetration.
    Zero=touching.
    """
    c = cls.get(env)
    wall_pos, normal = cls.wall_plane(env)  # (E,3), (E,3)
    geom_pos = env.sim.data.geom_xpos[:, c.geom_ids, :]  # (E, 16, 3)
    delta = geom_pos - wall_pos.unsqueeze(1)  # (E, 16, 3)
    # Signed perpendicular distance from geom center to wall plane:
    signed = (delta * normal.unsqueeze(1)).sum(dim=-1) - c.wall_half  # (E, 16)
    # Box support along the wall normal depends on its full orientation;
    # treating size[0] as a radius is incorrect, especially at tilted poses.
    rotations = env.sim.data.geom_xmat[:, c.geom_ids].reshape(-1, len(c.geom_ids), 3, 3)
    local_normal = torch.einsum("egij,ei->egj", rotations, normal)
    extent = (local_normal.abs() * c.sizes.unsqueeze(0)).sum(-1)
    extent = torch.where(c.is_box.unsqueeze(0), extent, c.sizes[:, 0].unsqueeze(0))
    return signed - extent


# ---------------------------------------------------------------------------
# Tracking terms read the climb_motion command and use exp(-error/std**2).
# Body poses align in XY/yaw; body velocities stay in the world frame.
# ---------------------------------------------------------------------------


def _motion(env: ManagerBasedRlEnv, command_name: str) -> "ClimbMotionCommand":
  return env.command_manager.get_term(command_name)


def _graced_mismatch(
  motion: "ClimbMotionCommand",
  grace_time_s: float | None,
  grace_fraction: float | None,
) -> torch.Tensor:
  """Per-EE mismatch scaled by the edge-distance grace ramp, (E, H).

  ``grace_fraction`` prices the grace as a fraction of the CURRENT scheduled
  run's length (per EE, per frame): a weld-through bill is then a constant
  fraction of the violated window no matter how the clip is timed. The fixed
  ``grace_time_s`` alternative silently discounted welding when the clip was
  compressed — at 0.25 s over a 0.6 s foot window only the middle 0.1 s
  billed full price (motionref16's weld economics). Fraction takes
  precedence when both are given.
  """
  mismatch = (motion.ref_engaged != motion.robot_engaged).float()
  if grace_fraction is not None and grace_fraction > 0.0:
    edge_seconds = motion.ref_edge_distance * motion.cycle_time_s
    grace_seconds = (
      motion.ref_run_time * motion.cycle_time_s * grace_fraction
    ).clamp(min=1.0e-6)
    return mismatch * (edge_seconds / grace_seconds).clamp(max=1.0)
  if grace_time_s is not None and grace_time_s > 0.0:
    edge_seconds = motion.ref_edge_distance * motion.cycle_time_s
    return mismatch * (edge_seconds / grace_time_s).clamp(max=1.0)
  return mismatch


def _engage_match_gate(
  motion: "ClimbMotionCommand",
  floor: float,
  grace_time_s: float,
  grace_fraction: float | None = None,
) -> torch.Tensor:
  """Schedule-adherence income gate in [floor, 1], per env.

  The mirror of ``motion_engage_mismatch``: each EE's mismatch is priced by
  the same edge-distance ramp the bill uses (near a scheduled flip it is
  forgiven — instant compliance is physically impossible), and the gate is
  1 - mean(billed mismatch), squeezed above ``floor``. Pose-tracking income
  multiplied by this gate flows in full only while the latch pattern follows
  the chart; a limb-welding statue keeps at most ``floor`` + its accidental
  match share. The floor keeps early pose learning funded (RSI births start
  fully matched, so the gradient toward keeping the match exists from step
  one).
  """
  mismatch = _graced_mismatch(motion, grace_time_s, grace_fraction)
  # R4: pending reseat EEs are exempt here too — a frozen-clock recovery
  # must not throttle the whole robot's tracking income through the gate.
  mismatch = mismatch * (~motion._affordance().dr_release_pending).float()
  match = 1.0 - mismatch.mean(dim=-1)
  return floor + (1.0 - floor) * match


def motion_joint_pos_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.2,
  engage_gate_floor: float | None = None,
  engage_gate_grace_s: float = 0.25,
  engage_gate_grace_fraction: float | None = None,
) -> torch.Tensor:
  """exp kernel on the mean squared clip-joint angle error (rad).

  With ``engage_gate_floor`` the payout is scaled by the schedule-adherence
  gate (see ``_engage_match_gate``): looking right only pays in full while
  gripping right — the statue's engagement-blind pose income was the last
  exploit pillar (motionref9: welded diagonal tracked the near-stance cycle
  at ~0.09 rad forever while ignoring the latch chart).
  """
  motion = _motion(env, command_name)
  error = (motion.ref_joint_pos - motion.robot_joint_pos).square().mean(dim=-1)
  reward = torch.exp(-error / std**2)
  if engage_gate_floor is not None:
    reward = reward * _engage_match_gate(
      motion, engage_gate_floor, engage_gate_grace_s, engage_gate_grace_fraction
    )
  return reward


def motion_body_position_error_exp(
  env, command_name="climb_motion", std=0.1, engage_gate_floor=None,
  engage_gate_grace_s=0.25, engage_gate_grace_fraction=None,
):
  """BeyondMimic mean body-origin position error after XY/yaw alignment."""
  motion = _motion(env, command_name)
  reference, _ = motion.aligned_body_reference()
  error = (reference - motion.robot_body_pos_w).square().sum(-1).mean(-1)
  reward = torch.exp(-error / std**2)
  if engage_gate_floor is not None:
    reward *= _engage_match_gate(motion, engage_gate_floor,
      engage_gate_grace_s, engage_gate_grace_fraction)
  return reward


def motion_body_orientation_error_exp(
  env, command_name="climb_motion", std=0.3, engage_gate_floor=None,
  engage_gate_grace_s=0.25, engage_gate_grace_fraction=None,
):
  """BeyondMimic mean squared geodesic body orientation error."""
  from mjlab.utils.lab_api.math import quat_error_magnitude
  motion = _motion(env, command_name)
  _, reference = motion.aligned_body_reference()
  error = quat_error_magnitude(reference, motion.robot_body_quat_w).square().mean(-1)
  reward = torch.exp(-error / std**2)
  if engage_gate_floor is not None:
    reward *= _engage_match_gate(motion, engage_gate_floor,
      engage_gate_grace_s, engage_gate_grace_fraction)
  return reward


def motion_body_linear_velocity_error_exp(env, command_name="climb_motion", std=1.0):
  motion = _motion(env, command_name)
  error = (motion.ref_body_lin_vel_w - motion.robot_body_lin_vel_w).square().sum(-1).mean(-1)
  return torch.exp(-error / std**2)


def motion_body_angular_velocity_error_exp(env, command_name="climb_motion", std=3.14):
  motion = _motion(env, command_name)
  error = (motion.ref_body_ang_vel_w - motion.robot_body_ang_vel_w).square().sum(-1).mean(-1)
  return torch.exp(-error / std**2)


def undesired_body_contacts(env, sensor_name="climb_undesired_contact", threshold=1.0):
  """Count unwanted body contacts exceeding threshold during a policy step.

  The sensor excludes expected pad/toe bodies; a peak over substeps retains
  brief impacts. maxforce reduction prevents opposing contacts cancelling.
  """
  data = env.scene.sensors[sensor_name].data
  forces = data.force_history
  if forces is None:
    forces = data.force.unsqueeze(2)
  return (forces.norm(dim=-1).amax(-1) > threshold).float().sum(-1)


def motion_joint_vel_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 1.5,
) -> torch.Tensor:
  """exp kernel on the mean squared clip-joint velocity error (rad/s)."""
  motion = _motion(env, command_name)
  error = (motion.ref_joint_vel - motion.robot_joint_vel).square().mean(dim=-1)
  return torch.exp(-error / std**2)


def motion_ee_pos_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.1,
  engage_gate_floor: float | None = None,
  engage_gate_grace_s: float = 0.25,
  engage_gate_grace_fraction: float | None = None,
) -> torch.Tensor:
  """exp kernel on root-frame end-effector position error (m).

  Root-relative on both sides, so this shapes the limb geometry of the cycle
  (reach, swing arcs) without ever pinning the base to the clip's world
  trajectory. ``engage_gate_floor``: see ``_engage_match_gate``.
  """
  motion = _motion(env, command_name)
  error = (
    (motion.ref_ee_pos_r - motion.robot_ee_pos_r).square().sum(dim=-1).mean(dim=-1)
  )
  reward = torch.exp(-error / std**2)
  if engage_gate_floor is not None:
    reward = reward * _engage_match_gate(
      motion, engage_gate_floor, engage_gate_grace_s, engage_gate_grace_fraction
    )
  return reward


def motion_root_ori_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.3,
  engage_gate_floor: float | None = None,
  engage_gate_grace_s: float = 0.25,
  engage_gate_grace_fraction: float | None = None,
) -> torch.Tensor:
  """exp kernel on the root orientation geodesic error (rad).

  Orientation only — the reference base tilts and rights itself through the
  hand swings, and that posture is part of the motion. Position stays free.
  ``engage_gate_floor``: see ``_engage_match_gate``.
  """
  from mjlab.utils.lab_api.math import quat_error_magnitude

  motion = _motion(env, command_name)
  error = quat_error_magnitude(
    motion.ref_root_quat_w, motion.robot_root_quat_w
  ).square()
  reward = torch.exp(-error / std**2)
  if engage_gate_floor is not None:
    reward = reward * _engage_match_gate(
      motion, engage_gate_floor, engage_gate_grace_s, engage_gate_grace_fraction
    )
  return reward


def motion_engage_mismatch(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  grace_time_s: float | None = None,
  grace_fraction: float | None = None,
) -> torch.Tensor:
  """Per-EE bill for a latch state disagreeing with the clip's schedule.

  Pair with a negative weight. This is the DeepMimic-style replacement for
  the old commanded-gait machinery: the clip says which pads are anchored at
  each phase; the policy must physically release/press to match it (the
  affordance still owns the latch).

  With ``grace_time_s`` the bill scales linearly with the time to the
  NEAREST schedule edge, saturating at full price ``grace_time_s`` away from
  it. A latch transition can never be instantaneous (0.06 s engage debounce,
  0.20 s re-engage cooldown), so full price right at a scheduled flip bills
  the physically impossible. The graded bill is a piecewise-linear function
  of the same countdown channels the policy observes in ``motion_command``,
  so the schedule -> action mapping is learnable from the chart itself —
  and transfers to a different clip — rather than memorized against the
  phase clock.

  ``grace_fraction`` prices the grace as a fraction of the violated window's
  own length instead of a fixed number of seconds (see ``_graced_mismatch``)
  — the weld-through bill then survives clip retiming unchanged.
  """
  motion = _motion(env, command_name)
  bill = _graced_mismatch(motion, grace_time_s, grace_fraction)
  # R4: EEs lost to a dr-forced release are exempt while their reseat hold
  # is armed — the frozen clock exists so recovery is not billed as truancy.
  pending = motion._affordance().dr_release_pending
  return (bill * (~pending).float()).sum(dim=-1)


def motion_scheduled_press_shaping(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  press_norm_n: float = 12.0,
  regrip_lead_time_s: float = 0.3,
) -> torch.Tensor:
  """Seat-phase step credit: pay pad press force inside the regrip window.

  Pair with a small positive weight. Seven runs in a row parked a limb pair
  because a regrip's first motions (reach, press, hold through debounce)
  earn nothing until the latch closes — undirected exploration never strings
  them together. This term pays the press itself, but ONLY per EE inside its
  scheduled seating window: the reference says the pad should be anchored
  (``ref_engaged``) or is about to be (regrip countdown within
  ``regrip_lead_time_s`` — the same observed channel), AND the robot has not
  latched yet. Payment stops the instant the latch closes: an engaged
  microspine anchor carries tension, pressing is only the doorway. Pressing
  outside the window pays zero, and the per-EE ceiling (press_norm_n
  saturation x the configured weight) must stay below the mismatch bill so
  press-and-never-latch is strictly net-negative.
  """
  motion = _motion(env, command_name)
  affordance = motion._affordance()
  lead = regrip_lead_time_s / motion.cycle_time_s
  regrip_soon = motion.ref_edge_countdowns[..., 1] <= lead
  window = (motion.ref_engaged | regrip_soon) & ~motion.robot_engaged
  press = (affordance.pad_press_force / press_norm_n).clamp(max=1.0)
  return (press * window.float()).sum(dim=-1)


def _toe_sensor_row_perm(sensor, toe_geom_names: tuple[str, ...]) -> "torch.Tensor":
  """Map the toe contact sensor's row order onto PHYBOT_C2_FOOT_TOES order."""
  perm = getattr(sensor, "_toe_brace_row_perm", None)
  if perm is None:
    names = getattr(sensor, "primary_names", None)
    if names is None:
      names = list(dict.fromkeys(slot.primary_name for slot in sensor._slots))
    order = {name.split("/")[-1]: i for i, name in enumerate(names)}
    missing = [g for g in toe_geom_names if g not in order]
    if missing:
      raise ValueError(
        f"Toe contact sensor rows {names} are missing geoms {missing}."
      )
    perm = torch.tensor(
      [order[g] for g in toe_geom_names],
      dtype=torch.long,
      device=sensor.data.found.device,
    )
    sensor._toe_brace_row_perm = perm
  return perm


def motion_anchor_landing():
  """Factory: pay each scheduled latch by landing accuracy on the anchor ladder.

  The shape rewards are deliberately root-relative and the engage gate only
  checks latch TIMING, so a policy can execute the whole regrip cycle in
  place — releasing and re-latching each pad at its old spot — and still
  collect nearly every imitation term (the measured iter-1400 basin: match
  0.98, net climb ~20% of the ghost). This term closes that hole with the
  one channel none of them read: WHERE the pad lands. On an engage rising
  edge inside that EE's scheduled window it pays
  ``exp(-(|anchor - ref_anchor|/std)^2)`` once — the reference anchor ladder
  climbs one step height per cycle, so landing at the old spot (30 mm short)
  earns ~0.37 at std 0.03 while the scheduled rung pays 1.0.

  Anti-farming: one payment per (EE, schedule window), cleared only when the
  reference window closes — release-and-relatch inside a window collects
  nothing extra (and still pays the mismatch/unsafe-release bills).
  Off-schedule latches never pay.
  """
  state: dict[str, torch.Tensor | None] = {"paid": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "climb_motion",
    std: float = 0.03,
  ) -> torch.Tensor:
    motion = _motion(env, command_name)
    affordance = motion._affordance()
    executor = affordance._get_executor()
    ref_on = motion.ref_engaged
    paid = state["paid"]
    if paid is None or paid.shape != ref_on.shape:
      paid = torch.zeros_like(ref_on)
    # A window already open at spawn (RSI seeds the latch mid-window) is
    # marked PAID: its honest play is "stay latched" with no landing event,
    # so leaving it eligible would pay a release-and-relatch churn (+0.8
    # gross vs ~0.26 of mismatch bill at these weights). Eligibility starts
    # at each EE's first fresh window.
    fresh = (env.episode_length_buf <= 1).unsqueeze(-1)
    paid = torch.where(fresh, affordance._engaged & ref_on, paid)
    paid = paid & ref_on  # window closed -> eligibility restored
    landed = affordance.engaged_this_step & ref_on & ~paid
    # The executor captures the weld anchor one step after the latch
    # decision, so judge the CURRENT pad center — the exact point the
    # capture will pin next step (anchor projection is off in this task).
    error = (
      executor.pad_center_world() - motion.ref_anchor_pos_w
    ).norm(dim=-1)
    pay = torch.exp(-(error / std) ** 2)
    state["paid"] = paid | landed
    return (pay * landed.float()).sum(dim=-1)

  return call


def motion_swing_approach(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  std: float = 0.05,
) -> torch.Tensor:
  """Dense shaping toward the upcoming landing rung during scheduled swings.

  The anchor-landing bonus pays once, at the latch, judged on the anchor
  ladder — but reaching a rung one step height UP requires the whole
  preceding swing to aim there, and none of those intermediate motions earn
  anything on their own (measured motionref16: landing income stuck at 3-5%
  of its ceiling because exploration never sampled a high landing). This
  pays the approach itself: while an EE is scheduled OFF (its swing window)
  and actually released, an exp kernel on the pad-center distance to
  ``ref_anchor_pos_w`` — which during a swing IS the upcoming landing
  target (see ClimbMotionCommand.ref_anchor_pos_w). Latching ends the pay
  (the landing bonus takes over); a released pad during a scheduled ON
  window earns nothing here — that mismatch is billed, not rewarded.
  """
  motion = _motion(env, command_name)
  affordance = motion._affordance()
  executor = affordance._get_executor()
  active = (~motion.ref_engaged) & (~motion.robot_engaged)
  error = (executor.pad_center_world() - motion.ref_anchor_pos_w).norm(dim=-1)
  pay = torch.exp(-((error / std) ** 2))
  return (pay * active.float()).sum(dim=-1)


def motion_toe_brace_match(
  env: ManagerBasedRlEnv,
  command_name: str = "climb_motion",
  sensor_name: str = "climb_toe_contact",
  toe_geom_names: tuple[str, ...] = (
    "left_ankle_roll_toe",
    "right_ankle_roll_toe",
  ),
  press_norm_n: float = 8.0,
  off_schedule_cost: float = 0.3,
) -> torch.Tensor:
  """Pay the scheduled foot-toe wall brace; bill unscheduled toe pressing.

  The reference's foot regrips brace the articulated foot's toe against the
  wall while the released spine pad travels clear of it (``toe_braced``
  clip channel). Like the pad press shaping, the brace's first motions earn
  nothing on their own — this pays normal toe force inside the scheduled
  window, saturating at ``press_norm_n`` (the audit needs only ~16 N).
  Pressing OUTSIDE the window costs ``off_schedule_cost`` of the same
  saturated scale: a toe reacting against the wall while its own pad is
  latched loads that fresh anchor toward reverse-shear/tension (the cheap
  release directions) — the exact interaction the reference's hover handoff
  exists to prevent.
  """
  motion = _motion(env, command_name)
  scheduled = motion.ref_toe_braced
  sensor = env.scene[sensor_name]
  data = sensor.data
  if data.found is None or data.force is None:
    raise ValueError(
      f"Contact sensor '{sensor_name}' must provide 'found' and 'force'."
    )
  perm = _toe_sensor_row_perm(sensor, toe_geom_names)
  found = data.found[:, perm] > 0
  # CONTACT sensor force is [normal, tangent1, tangent2].
  normal = data.force[:, perm, 0].clamp(min=0.0)
  press = (normal / press_norm_n).clamp(max=1.0) * found.float()
  on = scheduled.float()
  return (press * on - off_schedule_cost * press * (1.0 - on)).sum(dim=-1)


def motion_ref_height_tracking():
  """Factory: exp kernel on (robot climb - reference climb) since settling.

  The climbing DRIVE as an imitation term. Every free-standing height income
  ("any rise pays, no questions asked") funded a reference-ignoring gait —
  scramble, arms-only, legs-only — and each takeoff bought climb income with
  falling schedule adherence. Here the money follows the ghost instead: the
  reference's cumulative climb since this episode settled (``ref_root_pos_w``
  accrues one step height per loop) is the target, and the payout decays as
  the robot falls BEHIND or runs AHEAD of it symmetrically. RSI places the
  robot exactly on the profile, so holding a proper stance and moving with
  the ghost pays full from the first settled step — the honest path finally
  has dense income. Bounded in (0, 1]: the per-step caps and budget
  machinery of the free-standing terms have nothing left to guard.
  """
  state: dict[str, torch.Tensor | None] = {
    "fresh": None,
    "z_base": None,
    "ref_base": None,
  }

  def call(
    env: ManagerBasedRlEnv,
    *,
    asset_cfg: SceneEntityCfg,
    motion_command_name: str = "climb_motion",
    std: float = 0.05,
    settle_time_s: float = 0.5,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    motion = env.command_manager.get_term(motion_command_name)
    # Along-wall climb coordinates (world z on a vertical wall, bit-identical
    # there). BOTH sides project onto the same per-env wall axis — rotating
    # only one of them would silently misprice every tilted rung.
    up = motion.wall_up_w
    z = (asset.data.root_link_pos_w * up).sum(dim=-1)
    ref_z = (motion.ref_root_pos_w * up).sum(dim=-1)

    if state["fresh"] is None or state["fresh"].shape != z.shape[:1]:
      state["fresh"] = torch.zeros(
        z.shape[0], dtype=torch.bool, device=z.device
      )
      state["z_base"] = z.clone()
      state["ref_base"] = ref_z.clone()
      return torch.ones_like(z)

    fresh = state["fresh"]
    z_base = state["z_base"]
    ref_base = state["ref_base"]
    assert z_base is not None and ref_base is not None
    settling = env.episode_length_buf.float() * env.step_dt <= settle_time_s
    initialize = fresh | settling
    z_base = torch.where(initialize, z, z_base)
    ref_base = torch.where(initialize, ref_z, ref_base)
    state["z_base"] = z_base
    state["ref_base"] = ref_base
    fresh.zero_()

    dh = (z - z_base) - (ref_z - ref_base)
    return torch.exp(-(dh**2) / std**2)

  def reset(env_ids=None) -> dict:
    if state["fresh"] is None:
      return {}
    ids = slice(None) if env_ids is None else env_ids
    state["fresh"][ids] = True
    return {}

  call.reset = reset  # type: ignore[attr-defined]
  return call


# ---------------------------------------------------------------------------
# Task drive: height and velocity. These pay for actual climbing, independent
# of how closely the motion is imitated.
# ---------------------------------------------------------------------------


def climb_height_progress():
  """Factory: reward new height after the reset pose has physically settled.

  UNMOUNTED (superseded by motion_ref_height) and NOT wall-frame aware: if
  revived, the world-z high-water mark and ``command[:, 2]`` reads must
  become along-wall projections (see motion_ref_height_tracking).

  The term tracks the best base height reached in each episode and rewards
  only increases of that high-water mark. A policy that thrusts the hips up
  and falls back down can receive this reward once for the first peak, but not
  repeatedly for the same oscillation. During ``settle_time_s`` the high-water
  mark follows the base without reward, preventing the unavoidable reset sag
  from creating a several-centimeter reward dead zone.
  """
  state: dict[str, torch.Tensor | None] = {
    "best_z": None,
    "fresh": None,
    "budget": None,
  }

  def call(
    env: ManagerBasedRlEnv,
    *,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    target_vz: float = 0.08,
    command_threshold: float = 0.01,
    max_normalized_progress: float = 1.5,
    settle_time_s: float = 0.5,
    rate_budget_scale: float | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]

    if state["best_z"] is None or state["best_z"].shape != z.shape:
      state["best_z"] = z.clone()
      state["fresh"] = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
      state["budget"] = torch.zeros_like(z)
      return torch.zeros_like(z)

    fresh = state["fresh"]
    prev_best = state["best_z"]
    assert fresh is not None and prev_best is not None
    settling = env.episode_length_buf.float() * env.step_dt <= settle_time_s
    initialize = fresh | settling
    prev_best = torch.where(initialize, z, prev_best)
    new_best = torch.maximum(prev_best, z)
    progress_m = new_best - prev_best
    state["best_z"] = torch.where(initialize, z, new_best)
    fresh.zero_()

    if command_name is not None:
      cmd_z = env.command_manager.get_command(command_name)[:, 2].abs()
    else:
      cmd_z = torch.full_like(z, abs(target_vz))

    active = cmd_z > command_threshold
    target_step = (cmd_z * env.step_dt).clamp_min(1.0e-6)
    progress = (progress_m / target_step).clamp(max=max_normalized_progress)
    progress = torch.where(
      active & ~initialize, progress, torch.zeros_like(progress)
    )
    if rate_budget_scale is not None:
      # Cumulative income cap: total normalized progress paid per episode may
      # not exceed rate_budget_scale x the commanded climb distance over the
      # same (active, settled) time. The reference's within-cycle burst spends
      # headroom banked during its non-climbing phases, so honest cycles pay
      # in full while sustained overspeed scrambling is rate-limited — the
      # per-step cap above cannot distinguish the two.
      budget = state["budget"]
      assert budget is not None
      budget = budget + (active & ~initialize).float() * rate_budget_scale
      paid = torch.minimum(progress, budget)
      state["budget"] = budget - paid
      progress = paid
    return progress

  def reset(env_ids=None) -> dict:
    if state["fresh"] is None:
      return {}
    if env_ids is None:
      state["fresh"].fill_(True)
      if state["budget"] is not None:
        state["budget"].zero_()
    else:
      state["fresh"][env_ids] = True
      if state["budget"] is not None:
        state["budget"][env_ids] = 0.0
    return {}

  call.reset = reset  # type: ignore[attr-defined]
  return call


def signed_climb_height_progress():
  """Factory: dense signed base-height change after initial settling.

  UNMOUNTED (superseded by motion_ref_height) and NOT wall-frame aware: if
  revived, its world-z deltas must become along-wall projections.

  Unlike a positive-only velocity reward, a closed up/down cycle has roughly
  zero undiscounted return. Its weight should stay below the high-water term:
  it supplies a recovery gradient below the previous maximum, while genuinely
  new height remains the primary objective.

  Two mutually exclusive upward-income caps: ``reference_budget_command``
  bounds cumulative net income by the motion reference's own climb profile
  (the cap IS the reference's value at this instant), while
  ``rate_budget_scale`` is the legacy flat-rate budget. Descent is never
  budgeted in either mode.
  """
  state: dict[str, torch.Tensor | None] = {
    "prev_z": None,
    "fresh": None,
    "budget": None,
    "ref_base": None,
    "net": None,
  }

  def call(
    env: ManagerBasedRlEnv,
    *,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    target_vz: float = 0.08,
    command_threshold: float = 0.01,
    max_normalized_progress: float = 1.0,
    settle_time_s: float = 0.5,
    rate_budget_scale: float | None = None,
    reference_budget_command: str | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]

    if state["prev_z"] is None or state["prev_z"].shape != z.shape:
      state["prev_z"] = z.clone()
      state["fresh"] = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
      state["budget"] = torch.zeros_like(z)
      state["ref_base"] = torch.zeros_like(z)
      state["net"] = torch.zeros_like(z)
      return torch.zeros_like(z)

    prev_z = state["prev_z"]
    fresh = state["fresh"]
    assert prev_z is not None and fresh is not None
    dz = z - prev_z
    state["prev_z"] = z.clone()

    if command_name is not None:
      cmd_z = env.command_manager.get_command(command_name)[:, 2]
    else:
      cmd_z = torch.full_like(z, target_vz)
    active = cmd_z.abs() > command_threshold
    target_step = (cmd_z.abs() * env.step_dt).clamp_min(1.0e-6)
    progress = (dz / target_step).clamp(
      min=-max_normalized_progress,
      max=max_normalized_progress,
    )

    settling = env.episode_length_buf.float() * env.step_dt <= settle_time_s
    initialize = fresh | settling
    fresh.zero_()
    out = torch.where(active & ~initialize, progress, torch.zeros_like(progress))
    if reference_budget_command is not None:
      # Upper cap = the reference's own value: cumulative net income may not
      # exceed the height the motion reference itself has gained since this
      # episode settled (``ref_root_pos_w`` accrues one step height per loop,
      # so the cap keeps climbing with the schedule). Climbing ahead of the
      # schedule earns nothing until the reference catches up; descent bills
      # in full and restores the same headroom, so a sag-and-recover loop
      # nets zero instead of bleeding.
      motion = env.command_manager.get_term(reference_budget_command)
      ref_z = motion.ref_root_pos_w[:, 2]
      ref_base = state["ref_base"]
      net = state["net"]
      assert ref_base is not None and net is not None
      ref_base = torch.where(initialize, ref_z, ref_base)
      net = torch.where(initialize, torch.zeros_like(net), net)
      state["ref_base"] = ref_base
      allowance = ((ref_z - ref_base) / target_step - net).clamp_min(0.0)
      paid = torch.minimum(out.clamp_min(0.0), allowance)
      out = paid + out.clamp_max(0.0)
      state["net"] = net + out
    elif rate_budget_scale is not None:
      # Same cumulative cap as climb_height_progress, applied to the upward
      # side only: descent keeps its full recovery gradient.
      budget = state["budget"]
      assert budget is not None
      budget = budget + (active & ~initialize).float() * rate_budget_scale
      paid = torch.minimum(out.clamp_min(0.0), budget)
      state["budget"] = budget - paid
      out = paid + out.clamp_max(0.0)
    return out

  def reset(env_ids=None) -> dict:
    if state["fresh"] is None:
      return {}
    ids = slice(None) if env_ids is None else env_ids
    state["fresh"][ids] = True
    if state["budget"] is not None:
      state["budget"][ids] = 0.0
    if state["net"] is not None:
      state["net"][ids] = 0.0
    return {}

  call.reset = reset  # type: ignore[attr-defined]
  return call


def climb_lateral_velocity_y_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  velocity_scale: float = 0.1,
  max_error: float = 4.0,
) -> torch.Tensor:
  """Penalize wall-tangent lateral drift without suppressing wall-normal motion."""
  asset: Entity = env.scene[asset_cfg.name]
  error = asset.data.root_link_lin_vel_w[:, 1] / velocity_scale
  return error.square().clamp(max=max_error)


def wall_distance_band_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  target_distance: float = 0.255,
  tolerance: float = 0.08,
  distance_scale: float = 0.1,
  max_error: float = 4.0,
  wall_half_thickness: float = 0.5,
) -> torch.Tensor:
  """Penalize only base-to-wall distances outside a permissive dead band."""
  asset: Entity = env.scene[asset_cfg.name]
  wall_pos, outward_normal = _SphereGeomCache.wall_plane(env)
  base_offset = asset.data.root_link_pos_w - wall_pos
  distance = (
    (base_offset * outward_normal).sum(dim=-1) - wall_half_thickness
  )
  error = (distance - target_distance).abs().sub(tolerance).clamp_min(0.0)
  return (error / distance_scale).square().clamp(max=max_error)


# ---------------------------------------------------------------------------
# Microspine physics contract: minimum support, controlled release, no slide,
# wall-facing pads. These are unchanged by the DeepMimic redesign — tracking
# teaches the motion, these guarantee the physical rules while doing it.
# ---------------------------------------------------------------------------


def _hand_body_ids(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  body_names: tuple[str, ...] = ("left_hand", "right_hand"),
) -> list[int]:
  asset: Entity = env.scene[asset_cfg.name]
  ids = []
  for name in body_names:
    idx, _ = asset.find_bodies([name])
    ids.append(idx[0])
  return ids


def hand_slide_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  event_term_name: str = "microspine_grip",
  hand_body_names: tuple[str, ...] = ("left_hand", "right_hand"),
) -> torch.Tensor:
  """Penalise tangential wall-plane velocity of ENGAGED end-effectors.

  Gate the slip speed by the actual ``engaged`` latch state rather than a
  desired-contact schedule, so the penalty tracks the real "grip should not
  slide" condition. Returns ``Σ_h e[h] · ‖v_h,tan‖`` (raw m/s).

  Pair with a negative weight. Normal-component motion (elastic wall
  compression, engage / release impact) is not penalised.
  """
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = _hand_body_ids(env, asset_cfg, hand_body_names)
  _, normal_w = _SphereGeomCache.wall_plane(env)  # (E, 3)

  term_cfg = env.event_manager.get_term_cfg(event_term_name)
  engaged = term_cfg.func.engaged.float()  # (E, H)

  total = torch.zeros(env.num_envs, device=env.device)
  for h, body_id in enumerate(body_ids):
    v = asset.data.body_link_lin_vel_w[:, body_id, :]  # (E, 3)
    v_tan = v - (v * normal_w).sum(-1, keepdim=True) * normal_w
    total = total + engaged[:, h] * v_tan.norm(dim=-1)
  return total


# ``hand_slide_penalty`` accepts any EE body list, so passing the 4 spine
# names produces the generalised "engaged-EE tangential velocity" penalty.
ee_slide_penalty = hand_slide_penalty


def _ee_face_alignment_scores(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  ee_body_names: tuple[str, ...],
  face_local_axis: tuple[float, float, float],
) -> torch.Tensor:
  """Return per-EE wall-facing alignment scores in ``[0, 1]``."""
  asset: Entity = env.scene[asset_cfg.name]
  _, wall_normal = _SphereGeomCache.wall_plane(env)  # (E, 3) outer normal
  target = -wall_normal  # wall inward direction (robot → wall)

  face_local = torch.tensor(face_local_axis, dtype=torch.float32, device=env.device)
  vx, vy, vz = face_local[0], face_local[1], face_local[2]

  per_ee: list[torch.Tensor] = []
  for bname in ee_body_names:
    ids, _ = asset.find_bodies([bname])
    body_id = ids[0]
    quat = asset.data.body_link_quat_w[:, body_id, :]  # (E, 4) wxyz
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # Rotate face_local by quat (q · v · q^-1).
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    fx = vx + w * tx + (y * tz - z * ty)
    fy = vy + w * ty + (z * tx - x * tz)
    fz = vz + w * tz + (x * ty - y * tx)
    face = torch.stack([fx, fy, fz], dim=-1)  # (E, 3)
    dot = (face * target).sum(dim=-1).clamp(-1.0, 1.0)
    per_ee.append((dot + 1.0) * 0.5)
  return torch.stack(per_ee, dim=-1)


def ee_face_parallel_error(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  ee_body_names: tuple[str, ...],
  event_term_name: str = "microspine_grip",
  command_name: str | None = "microspine_affordance",
  face_local_axis: tuple[float, float, float] = (0.0, -1.0, 0.0),
) -> torch.Tensor:
  """Mean face-alignment error for engaged or wall-contacting EEs only.

  A remote swing limb is free to reorient. Once it touches the wall it joins
  the mask, providing approach alignment without paying a constant positive
  reward for an already-static pose.
  """
  scores = _ee_face_alignment_scores(
    env, asset_cfg, ee_body_names, face_local_axis
  )
  physics = env.event_manager.get_term_cfg(event_term_name).func
  mask = physics.engaged.clone()
  if command_name is not None:
    affordance = env.command_manager.get_term(command_name)
    mask |= affordance.contact_count > 0
  mask_f = mask.float()
  return ((1.0 - scores) * mask_f).sum(dim=-1) / mask_f.sum(
    dim=-1
  ).clamp_min(1.0)


def engaged_tilt_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
  threshold_deg: float = 3.0,
  scale_deg: float = 5.0,
) -> torch.Tensor:
  """Tax engaged-pad rotation from its engage capture, approaching the peel.

  Companion shaping for the tilt-peel release (reason 5): the physical
  budget is the spine-suspension travel — le_max 5 mm folded over the rake
  span gives ~3.8 deg of chassis rotation about the loaded edge before the
  tension-side row pulls out (tilt_peel_max_deg_range, DR 3.5-4.5). This
  term prices the approach so the cliff is felt as a gradient, not
  discovered by falling: capture-RELATIVE rotation (the same quantity the
  release judges) beyond ``threshold_deg`` costs quadratically. Swing pads
  are untaxed — reorienting mid-air is the job. The reference uses <=1.1
  deg, so schedule-true execution pays nothing.
  """
  affordance = env.command_manager.get_term(command_name)
  tilt_deg = torch.rad2deg(affordance.tilt_from_capture)
  over = ((tilt_deg - threshold_deg).clamp(min=0.0) / scale_deg).square()
  return (over * affordance.engaged.float()).sum(dim=-1)


def microspine_support_loss(
  env: ManagerBasedRlEnv,
  event_term_name: str = "microspine_grip",
  minimum_supports: int = 2,
) -> torch.Tensor:
  """Penalty kernel for having fewer than the minimum safe anchors."""
  engaged = env.event_manager.get_term_cfg(event_term_name).func.engaged
  shortfall = minimum_supports - engaged.sum(dim=-1)
  return shortfall.clamp_min(0).float()


def microspine_release_load_reward():
  """Factory: reward scheduled unloading, penalize tangentially loaded release.

  Uses the empirical judge's PRE-release unit loads, including twist-induced
  shear/lateral loads that can cancel in the pad resultant. Normal force is
  not an unloading target. The physical release rules remain with the judge.
  Each scheduled release window (lead-in plus OFF interval) can pay once;
  repeated releases still incur the load cost. External DR releases pay/cost
  nothing. Divide by dt so the configured weight is a per-event amount.
  """
  state: dict[str, torch.Tensor | None] = {"paid": None}

  def call(
    env: ManagerBasedRlEnv,
    *,
    command_name: str = "microspine_affordance",
    motion_command_name: str = "climb_motion",
    shear_scale_n: float = 2.0,
    lateral_scale_n: float = 2.0,
    shear_cost: float = 0.25,
    lateral_cost: float = 0.1,
    max_load_cost: float = 4.0,
    minimum_supports: int = 3,
    release_lead_time_s: float = 0.25,
    settle_time_s: float = 0.5,
  ) -> torch.Tensor:
    """Scales are N per active unit; the most loaded unit sets the score."""
    if min(shear_scale_n, lateral_scale_n) <= 0 or max_load_cost <= 0:
      raise ValueError("Release-load scales and maximum cost must be positive")
    if min(shear_cost, lateral_cost, release_lead_time_s) < 0:
      raise ValueError("Release-load costs and lead time must be nonnegative")
    affordance = env.command_manager.get_term(command_name)
    motion = _motion(env, motion_command_name)
    # Populated before the latch falling edge; do not read post-release
    # contacts, whose force has already vanished. Axes are [shear,lateral,normal].
    loads = affordance.unit_load_tln
    finite = torch.isfinite(loads).all(dim=-1).all(dim=-1)
    shear_sq = (loads[..., 0] / shear_scale_n).square()
    lateral_sq = (loads[..., 1] / lateral_scale_n).square()
    quality = torch.exp(-(shear_sq + lateral_sq).amax(dim=-1))
    load_cost = (shear_cost*shear_sq + lateral_cost*lateral_sq).amax(dim=-1)
    load_cost = torch.where(finite, load_cost.clamp(max=max_load_cost), max_load_cost)

    reason = affordance.release_reason
    released = affordance.forced_release_this_step & (reason > 0) & (reason != 6)
    release_soon = motion.ref_edge_countdowns[..., 0] * motion.cycle_time_s <= release_lead_time_s
    window = ~motion.ref_engaged | release_soon
    paid = state["paid"]
    if paid is None or paid.shape != window.shape or paid.device != window.device:
      paid = torch.zeros_like(window)
    # An RSI/reset born inside an existing swing cannot earn a fresh release
    # bonus by relatching there. Partial resets only touch their own env rows.
    fresh = (env.episode_length_buf <= 1).unsqueeze(-1)
    paid = torch.where(fresh, window, paid) & window
    support_ok = affordance.engaged.sum(dim=-1) >= minimum_supports
    recovering = affordance.dr_release_pending.any(dim=-1)
    # Unknown/early empirical failures are not successful controlled releases.
    controlled_reason = (reason >= 1) & (reason <= 5)
    pay = (released & controlled_reason & finite & window & ~paid
           & support_ok.unsqueeze(-1) & ~recovering.unsqueeze(-1))
    # Consume even an unsuccessful first release: relatch/release retries in
    # the same window cannot turn a bad release into extra positive income.
    state["paid"] = paid | (released & window)
    score = (torch.where(pay, quality, 0.) - torch.where(released, load_cost, 0.)).sum(dim=-1)
    settling = env.episode_length_buf.float() * env.step_dt <= settle_time_s
    return torch.where(settling, 0., score / env.step_dt)

  return call


def unsafe_microspine_release_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
  minimum_supports: int = 2,
  settle_time_s: float = 0.5,
  reverse_release_cost: float = 0.0,
  tension_release_cost: float = 0.25,
  lateral_release_cost: float = 0.25,
  ultimate_release_cost: float = 1.0,
  tilt_release_cost: float = 0.5,
  support_loss_cost: float = 1.0,
  motion_command_name: str | None = None,
  scheduled_reverse_cost: float = 0.3,
  scheduled_reverse_grace_s: float = 0.25,
) -> torch.Tensor:
  """Per-event cost that distinguishes controlled and failed disengagement.

  Reason 1 (reverse shear) is the passive spine's intended unhooking path and
  has no base reason cost while sufficient support remains. The separate
  release-load reward can still penalize its tangential load. Pull-off, lateral, and ultimate
  failures carry increasing costs. Dividing by ``step_dt`` makes the configured
  weight an actual one-shot event cost after RewardManager's dt scaling.

  With ``motion_command_name`` set, a reverse-shear unhook of an EE the clip
  schedules ENGAGED bills ``scheduled_reverse_cost`` — scaled down to zero
  within ``scheduled_reverse_grace_s`` of the nearest schedule edge, so the
  on-time unhook at a scheduled release stays free. This prices the
  latch-jiggle-unhook churn (motionref16b: 43-45 free reverse releases per
  30 s on the abandoned right foot) without touching legitimate releases.
  """
  affordance = env.command_manager.get_term(command_name)
  released = affordance.forced_release_this_step
  reason = affordance.release_reason
  cost = torch.zeros_like(reason, dtype=torch.float32)
  reason_costs = (
    (1, reverse_release_cost),
    (2, tension_release_cost),
    (3, lateral_release_cost),
    (4, ultimate_release_cost),
    (5, tilt_release_cost),  # tilt-peel: rotated the patch off its grip
    (7, ultimate_release_cost),  # empirical measured-domain rejection
    (8, tension_release_cost),  # empirical early-loading failure
  )
  for reason_id, value in reason_costs:
    cost += (released & (reason == reason_id)).float() * value

  if motion_command_name is not None and scheduled_reverse_cost > 0.0:
    motion = _motion(env, motion_command_name)
    off_schedule = released & (reason == 1) & motion.ref_engaged
    edge_seconds = motion.ref_edge_distance * motion.cycle_time_s
    ramp = (
      (edge_seconds / max(scheduled_reverse_grace_s, 1.0e-6)).clamp(max=1.0)
      if scheduled_reverse_grace_s > 0.0
      else torch.ones_like(edge_seconds)
    )
    cost += off_schedule.float() * ramp * scheduled_reverse_cost

  engaged_count = affordance.engaged.sum(dim=-1)
  unsafe_support = released.any(dim=-1) & (engaged_count < minimum_supports)
  event_cost = cost.sum(dim=-1) + unsafe_support.float() * support_loss_cost
  settling = env.episode_length_buf.float() * env.step_dt <= settle_time_s
  return torch.where(
    settling,
    torch.zeros_like(event_cost),
    event_cost / env.step_dt,
  )


def microspine_forced_release_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Count of EEs released by the limit surface this step, shape (E,).

  Penalizing forced releases teaches the policy to keep anchor loads inside
  the hold envelope (lean into shear, avoid pull-off/lateral peeling) instead
  of treating grip as unconditional.
  """
  affordance = env.command_manager.get_term(command_name)
  return affordance.forced_release_this_step.float().sum(dim=1)


# ---------------------------------------------------------------------------
# Regularization.
# ---------------------------------------------------------------------------


def torque_headroom_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  threshold: float = 0.7,
) -> torch.Tensor:
  """Price actuator torque spent above a duty threshold, per joint per step.

  Normalized quadratic in the remaining headroom: an actuator at the
  ``threshold`` costs 0, a SATURATED actuator costs 1.0 — so the configured
  weight is the per-second price of holding one joint at its limit.
  Torques below the threshold are free: this deliberately does not tax
  normal support duty (~0.5 of limit at 90 deg) or the gentle
  weight-shift-then-peel unhook (measured 0.45 peak on model_2175).

  Motivation (2026-07-30, motionref17 iter ~4100): with spike torque
  unpriced (dof_torques_l2 at -1e-5 is noise), the policy drifted from the
  weight-shift unhook to a brute-force yank — cancelling a hand anchor's
  ~85 N gravity share by raw shoulder torque, doubling the release-step
  peak from 0.45 to 0.95-1.00 of the 58 Nm cap for a +7% climb gain. The
  deploy stack clamps at these limits, so a saturated yank on hardware
  loses its impulse mid-release.
  """
  asset: Entity = env.scene[asset_cfg.name]
  torque = asset.data.actuator_force
  limits = getattr(asset, "_torque_headroom_limits", None)
  if limits is None or limits.shape[0] != torque.shape[1]:
    # Model-level actuator force ranges.
    fr = torch.as_tensor(env.sim.model.actuator_forcerange)
    if fr.ndim == 3:
      fr = fr[0]
    limits = (
      fr[asset.indexing.ctrl_ids]
      .abs()
      .amax(dim=-1)
      .to(device=torque.device, dtype=torque.dtype)
    )
    limits = torch.where(
      limits > 0.0, limits, torch.full_like(limits, torch.inf)
    )
    asset._torque_headroom_limits = limits  # type: ignore[attr-defined]
  util = torque.abs() / limits
  span = max(1.0 - threshold, 1.0e-6)
  over = ((util - threshold).clamp(min=0.0) / span).square()
  return over.sum(dim=-1)


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize base angular velocity around body x and y axes.

  Direct port of Anymal's ``ang_vel_xy_l2``. Discourages the policy from
  flailing roll/pitch — yaw rate (z) is intentionally not penalised here.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=-1)


def bounded_joint_acc_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  acceleration_scale: float = 100.0,
  max_cost: float = 10.0,
) -> torch.Tensor:
  """Mean squared joint acceleration with a finite influence bound.

  Constraint release and failed episodes can produce brief acceleration
  impulses several orders of magnitude above normal motion. An unbounded
  squared cost lets one such sample dominate an entire PPO batch. Normalize
  before averaging across joints and cap only that regularizer's influence;
  the termination and grip-safety terms still represent the failure itself.
  """
  if acceleration_scale <= 0.0:
    raise ValueError("acceleration_scale must be positive")
  if max_cost <= 0.0:
    raise ValueError("max_cost must be positive")

  asset: Entity = env.scene[asset_cfg.name]
  joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
  cost = torch.mean(torch.square(joint_acc / acceleration_scale), dim=1)
  cost = torch.nan_to_num(cost, nan=max_cost, posinf=max_cost, neginf=max_cost)
  return torch.clamp(cost, max=max_cost)
