"""Custom observation functions for the wall-climbing task."""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, TYPE_CHECKING

import torch

from mjlab.envs.mdp.observations import (
  builtin_sensor,
  generated_commands,
  joint_pos_rel,
  joint_vel_rel,
  last_action,
)
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.noise import UniformNoiseCfg as Unoise

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def EMAObsFilter(base_func: Callable, alpha: float = 0.35):
  """Per-term first-order IIR low-pass filter, applied to any obs func.

  Matches the formula from the wall-climbing paper (arXiv:2510.20174):
    o_filtered = (1 - alpha) * o_raw + alpha * o_filtered_prev
  with alpha=0.35 by default — i.e. mostly raw signal, 35% history.

  Reduces obs noise from CCD jitter / contact-induced velocity spikes
  without distorting steady-state values.

  Implemented as a closure (not a class) so tyro's CLI introspection
  doesn't try to expose the captured ``base_func`` as a CLI argument.
  Returns a callable with a ``.reset`` attribute, which is the duck-typed
  interface mjlab's ObservationManager uses to detect class-style terms
  (see observation_manager.py:437).
  """
  alpha = float(alpha)
  state = {"buf": None, "fresh": None}

  def call(env: "ManagerBasedRlEnv", **kwargs) -> torch.Tensor:
    raw = base_func(env, **kwargs)
    if state["buf"] is None or state["buf"].shape != raw.shape:
      state["buf"] = raw.clone()
      state["fresh"] = torch.zeros(raw.shape[0], dtype=torch.bool, device=raw.device)
      return raw
    fresh = state["fresh"]
    if fresh.any():
      state["buf"][fresh] = raw[fresh]
      fresh.zero_()
    state["buf"] = (1.0 - alpha) * raw + alpha * state["buf"]
    return state["buf"]

  def reset(env_ids: torch.Tensor | slice | None = None) -> dict:
    if state["fresh"] is None:
      return {}
    if env_ids is None:
      state["fresh"].fill_(True)
    else:
      state["fresh"][env_ids] = True
    return {}

  call.reset = reset  # type: ignore[attr-defined]
  return call


def projected_gravity_from_framequat_sensor(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Project the current simulator gravity direction into the base frame.

  ``EntityData.gravity_vec_w`` is a nominal world-down unit vector and does
  not track the gravity-tilt curriculum. Read the live MuJoCo/Warp gravity
  instead, then normalize it so the observation remains a unit direction.
  """
  del asset_cfg  # Kept for API compatibility with older env configs.
  import warp as wp

  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  quat_w = sensor.data
  quat_w = quat_w / torch.linalg.norm(quat_w, dim=-1, keepdim=True).clamp_min(1.0e-6)

  gravity_w = wp.to_torch(env.sim.wp_model.opt.gravity)  # (1, 3) or (E, 3)
  if gravity_w.shape[0] == 1:
    gravity_w = gravity_w.expand(env.num_envs, 3)
  else:
    gravity_w = gravity_w[:env.num_envs]
  gravity_dir_w = gravity_w / torch.linalg.norm(
    gravity_w, dim=-1, keepdim=True
  ).clamp_min(1.0e-6)
  return quat_apply_inverse(quat_w, gravity_dir_w)


def prev_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Action from the previous step (one step before current `last_action`)."""
  return env.action_manager.prev_action


def prev_prev_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Action from two steps ago."""
  return env.action_manager.prev_prev_action


# Gait clock period (seconds). A full cycle = one swing + one stance for each
# hand. Left hand: swing in φ ∈ [0, 0.5), stance in [0.5, 1). Right hand is
# offset by 0.5 so exactly one hand is always swinging.
#
# Sized to match arm reach. Each half cycle the body is pulled up by the
# stance arm's elbow flexion, capped physically at one upper-arm length
# (shoulder -> elbow is roughly 0.20 m on the PHYBot arm). With period = 4 s and
# vz_cmd = 0.08 m/s, per-step lift ≈ 0.16 m ≈ 80 % of that limit.
GAIT_PERIOD_S = 4.0


def gait_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per-env gait phase ∈ [0, 1). Advances with episode_length_buf · step_dt."""
  t = env.episode_length_buf.float() * env.step_dt
  return (t / GAIT_PERIOD_S) % 1.0


def gait_clock(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Gait clock observation (sin/cos × 2 hands) — shape (E, 4).

  The left hand uses the raw phase; the right hand phase is offset by 0.5 so
  the two hands swing in alternation (trot-style).
  """
  p = gait_phase(env)
  p_left = p
  p_right = (p + 0.5) % 1.0
  two_pi = 2.0 * math.pi
  return torch.stack(
    [
      torch.sin(two_pi * p_left),
      torch.cos(two_pi * p_left),
      torch.sin(two_pi * p_right),
      torch.cos(two_pi * p_right),
    ],
    dim=-1,
  )


def height_above_spawn(
  env: ManagerBasedRlEnv,
  baseline_z: float = 2.0,
  motion_command_name: str = "climb_motion",
) -> torch.Tensor:
  """Along-wall base altitude minus a fixed baseline. Shape (E, 1).

  The critic's altitude channel: without it the height income reads as pure
  noise (hidden-state variance in V). Measured ALONG the climbing surface —
  ``(root - wall_base_line) . wall_up`` — which is world z minus baseline on
  a vertical wall (bit-identical to the retired world-z form) and, under
  wall tilt, the same scalar the height rewards pay on. Using raw world z
  here would mix wall-normal standoff into "altitude" at every tilted rung.
  """
  motion = env.command_manager.get_term(motion_command_name)
  robot_pos = env.scene["robot"].data.root_link_pos_w
  altitude = ((robot_pos - motion.wall_pivot_w) * motion.wall_up_w).sum(dim=-1)
  return (altitude - baseline_z).unsqueeze(-1)  # (E, 1)


def velocity_command(
  env: ManagerBasedRlEnv,
  command_name: str | None = None,
  vx_cmd: float = 0.0,
  vy_cmd: float = 0.0,
  vz_cmd: float = 0.08,
) -> torch.Tensor:
  """Commanded linear velocity (world frame), shape (E, 3).

  When ``command_name`` is provided this reads the live command manager term,
  so observation and velocity-tracking rewards use the same per-env target. The
  fixed ``vx_cmd``/``vy_cmd``/``vz_cmd`` path is kept for legacy configs.
  """
  if command_name is not None:
    return env.command_manager.get_command(command_name)

  cmd = torch.tensor(
    [vx_cmd, vy_cmd, vz_cmd],
    device=env.device,
    dtype=torch.float32,
  )
  return cmd.unsqueeze(0).expand(env.num_envs, 3)


def microspine_pad_normal_force(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Per-EE pad-normal reaction force in Newtons, (E, 4) — load-cell model.

  Models the pad-mounted force channel the gimbal sensing co-design will
  provide: the wall-normal reaction the pad transmits, continuous across the
  latch event. Two sim sources are blended because
  ``disable_contacts_while_latched`` retracts the sphere grid once latched —
  ``pad_press_force`` reads structurally zero exactly when the anchor
  carries its biggest load. Released pads report the sphere-grid compression
  sum; latched pads the anchor equality-constraint normal reaction
  (``anchor_load_tnl[..., 2]`` is f_n, positive = pull-off tension, so
  compression is its negation clamped at zero). Privileged tensors serve
  only as the signal SOURCE; the output is the physically continuous
  quantity a real load cell measures. Noise/latency/quantization live on
  the ObservationTermCfg, to be recalibrated against the real sensor.
  """
  affordance = env.command_manager.get_term(command_name)
  anchor_compression = (-affordance.anchor_load_tnl[..., 2]).clamp(min=0.0)
  return torch.where(
    affordance.engaged, anchor_compression, affordance.pad_press_force
  )


def microspine_engaged_state(
  env: ManagerBasedRlEnv,
  event_term_name: str = "microspine_grip",
  command_name: str | None = None,
) -> torch.Tensor:
  """Per-EE microspine latch state, shape ``(E, num_ee)`` in ``{0, 1}``.

  The affordance command is preferred because it is the source of truth and is
  already initialized on the reset observation; the physics event mirrors that
  state into equalities one event pass later. ``event_term_name`` retains the
  legacy physics-state path.
  """
  if command_name is not None:
    engaged = env.command_manager.get_term(command_name).engaged
  else:
    term_cfg = env.event_manager.get_term_cfg(event_term_name)
    engaged = term_cfg.func.engaged  # (E, H) bool
  return engaged.float()


# Backward-compatible name retained for old climb configs.
hand_engaged_state = microspine_engaged_state


def microspine_contact_fraction(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
  spheres_per_ee: int = 4,
) -> torch.Tensor:
  """Per-EE engaged fraction of the virtual spine field (seating quality).

  Semantics upgrade (motionref9): the raw touching-sphere count was a dead
  channel — a rigid pad on a flat wall generically touches with exactly one
  sphere, so the old value pinned at 0.25 whenever pressing. The virtual
  spine fraction is the signal the engage gate and the capacity scaling
  actually consume ('how flat is the pad'), so the policy sees the same
  quantity it is judged and paid on. Same shape, ckpts across the change are
  incompatible (fresh runs only, as with every obs semantic change).
  """
  affordance = env.command_manager.get_term(command_name)
  return affordance.spine_fraction.clamp(0.0, 1.0)


def microspine_cooldown_obs(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
  cooldown_steps: int = 30,
) -> torch.Tensor:
  """Normalized per-EE re-engagement cooldown known to the grip controller."""
  affordance = env.command_manager.get_term(command_name)
  return affordance.cooldown_steps.float() / max(1, cooldown_steps)


def ee_disengage_time_obs(
  env: ManagerBasedRlEnv,
  event_term_name: str = "microspine_grip",
  clip: float = 0.5,
) -> torch.Tensor:
  """Normalized per-EE continuous disengage time, shape ``(E, H)``.

  The timer is maintained by ``MicrospineConstraint`` and reset to 0 while an
  EE is engaged. Clipping keeps the observation bounded and matches the
  max-time gate used by the capped disengage-time gait reward.
  """
  term_cfg = env.event_manager.get_term_cfg(event_term_name)
  physics = term_cfg.func
  if not hasattr(physics, "current_disengage_time"):
    return torch.zeros_like(physics.engaged, dtype=torch.float32)

  denom = max(float(clip), 1.0e-6)
  return physics.current_disengage_time.clamp(max=denom) / denom


# ---------------------------------------------------------------------------
# Privileged microspine state (critic-only).
#
# These read the limit-surface diagnostics that MicrospineAffordance maintains
# They are sim-only quantities (a real robot cannot sense its hold margin), so
# they belong in the critic group, never the actor.
# ---------------------------------------------------------------------------


def microspine_hold_margin(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Per-EE normalized distance to the hold envelope, shape (E, H).

  1 = unloaded, 0 = on the envelope boundary, < 0 = violating (about to
  release once debounced).
  """
  affordance = env.command_manager.get_term(command_name)
  return affordance.hold_margin


def microspine_anchor_load(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Per-EE anchor load in the engage frame (f_t, f_l, f_n), shape (E, 3H).

  Raw Newtons — pair with ``scale=0.01`` in the term cfg to keep the
  normalized magnitude O(1).
  """
  affordance = env.command_manager.get_term(command_name)
  return affordance.anchor_load_tnl.flatten(1)


def microspine_forced_release(
  env: ManagerBasedRlEnv,
  command_name: str = "microspine_affordance",
) -> torch.Tensor:
  """Per-EE limit-surface release flag this step, shape (E, H) in {0, 1}."""
  affordance = env.command_manager.get_term(command_name)
  return affordance.forced_release_this_step.float()


def privileged_critic_group(
  actor_group: ObservationGroupCfg,
  command_name: str = "microspine_affordance",
) -> ObservationGroupCfg:
  """Asymmetric actor-critic group: actor terms + privileged microspine state.

  The extras (hold margin, anchor loads, forced releases) are what a real
  robot cannot sense. Term cfgs are re-instantiated (``dataclasses.replace``)
  so the two groups never share mutable cfg objects; the shallow copies
  deliberately SHARE the ``params`` dicts with the actor terms (preexisting
  behavior — do not deep-copy). Call AFTER every actor-term edit so the
  copies pick them up.
  """
  critic_terms = {
    name: dataclasses.replace(term) for name, term in actor_group.terms.items()
  }
  critic_terms["microspine_hold_margin"] = ObservationTermCfg(
    func=microspine_hold_margin,
    params={"command_name": command_name},
  )
  critic_terms["microspine_anchor_load"] = ObservationTermCfg(
    func=microspine_anchor_load,
    params={"command_name": command_name},
    scale=0.01,  # Newtons → O(1).
  )
  critic_terms["microspine_forced_release"] = ObservationTermCfg(
    func=microspine_forced_release,
    params={"command_name": command_name},
  )
  return ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )


def student_observation_group(
  policy_joint_cfg: SceneEntityCfg,
  motion_command_name: str,
  affordance_command_name: str,
  gyro_sensor_name: str,
  quat_sensor_name: str,
) -> ObservationGroupCfg:
  """Deployable observation contract for the distilled student.

  Every term is hardware-measurable: IMU gyro + gravity, joint encoders, the
  policy's own action, the motion command (a pure 50 Hz clock lookup exported
  alongside the policy), and the gimbal load cell (``pad_normal_force`` — the
  one non-proprioceptive channel, also the deploy-side trigger for the R4
  clock pause). Deliberately absent: base_lin_vel / height_above_spawn (no
  estimator on the robot), the microspine latch state (no engagement
  sensing), the constant velocity_command, and prev/prev_prev_action (the
  history covers them). Full design narrative: docs/climb_env_cfg_notes.md
  ("student group").
  """
  student_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=builtin_sensor,
      params={"sensor_name": gyro_sensor_name},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity_from_framequat_sensor,
      params={"sensor_name": quat_sensor_name},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos_rel": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": dataclasses.replace(policy_joint_cfg)},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel_rel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": dataclasses.replace(policy_joint_cfg)},
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "last_action": ObservationTermCfg(func=last_action),
    "motion_command": ObservationTermCfg(
      func=generated_commands,
      params={"command_name": motion_command_name},
    ),
    # APPEND-ONLY from here: inserting mid-dict shifts every downstream
    # block of the exported C++ offset table. Pipeline reminder: noise is
    # applied per-frame BEFORE the history append; the 1-2 step delay
    # models load-cell/CAN latency and deliberately survives play/export
    # (the real sensor is late too — delay is not corruption-gated).
    "pad_normal_force": ObservationTermCfg(
      func=microspine_pad_normal_force,
      params={"command_name": affordance_command_name},
      noise=Unoise(n_min=-5.0, n_max=5.0),
      clip=(0.0, 200.0),
      scale=0.01,
      delay_min_lag=1,
      delay_max_lag=2,
      delay_per_env=True,
    ),
  }
  return ObservationGroupCfg(
    terms=student_terms,
    concatenate_terms=True,
    enable_corruption=True,
    # 50 frames = 1.0 s: a 0.2 s window cannot span the 0.6 s reengage
    # cooldown (latch timers uninferable); 1 s covers debounce + typical
    # reseat. 89 dims/frame x 50 = 4450 — the C++ packer contract.
    # (notes: student history_length)
    history_length=50,
  )
