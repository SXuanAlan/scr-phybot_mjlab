"""Custom observation functions for the wall-climbing task."""

from __future__ import annotations

import math
from typing import Callable, TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

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
# (shoulder → elbow ≈ 0.20 m on PHYBot Big V2). With period = 4 s and
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
  env: ManagerBasedRlEnv, baseline_z: float = 2.0
) -> torch.Tensor:
  """Robot base z-coordinate relative to spawn height. Shape (E, 1).

  CRITICAL for learning the ``climb_height`` reward. Without this, the
  critic cannot estimate V(s) based on current altitude, so the
  position-based reward turns into pure noise from the policy's point of
  view — every state looks the same to it, but the reward varies ±360 with
  invisible z. Adding z-above-spawn as an observation collapses that
  hidden-state variance and lets both V and π learn height-conditioned
  behaviour.
  """
  z = env.scene["robot"].data.root_link_pos_w[:, 2]  # (E,)
  return (z - baseline_z).unsqueeze(-1)  # (E, 1)


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


def hand_engaged_state(
  env: ManagerBasedRlEnv,
  event_term_name: str = "microspine_adhesion",
) -> torch.Tensor:
  """Per-hand microspine latch state. Shape (E, 2) float in {0, 1}.

  Tells the policy whether each hand is currently gripping the wall. This
  is the same ``engaged`` flag that ``airborne_penalty`` reads from the
  event manager — exposing it as an observation lets the policy
  coordinate motion with latch transitions (e.g. don't try to reach with
  a hand that's still engaged).
  """
  term_cfg = env.event_manager.get_term_cfg(event_term_name)
  engaged = term_cfg.func.engaged  # (E, H) bool
  return engaged.float()


def ee_disengage_time_obs(
  env: ManagerBasedRlEnv,
  event_term_name: str = "microspine_adhesion",
  clip: float = 0.5,
) -> torch.Tensor:
  """Normalized per-EE continuous disengage time, shape ``(E, H)``.

  The timer is maintained by ``MicrospineController`` and reset to 0 while an
  EE is engaged. Clipping keeps the observation bounded and matches the
  max-time gate used by the capped disengage-time gait reward.
  """
  term_cfg = env.event_manager.get_term_cfg(event_term_name)
  controller = term_cfg.func
  if not hasattr(controller, "current_disengage_time"):
    return torch.zeros_like(controller.engaged, dtype=torch.float32)

  denom = max(float(clip), 1.0e-6)
  return controller.current_disengage_time.clamp(max=denom) / denom
