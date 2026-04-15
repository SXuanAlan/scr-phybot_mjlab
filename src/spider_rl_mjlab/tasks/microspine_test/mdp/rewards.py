"""Custom reward functions for the microspine test task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from spider_rl_mjlab.assets.microspine.controller import MicrospineController


def _get_controller(
  env: ManagerBasedRlEnv, term_name: str
) -> MicrospineController:
  return env.event_manager.get_term_cfg(term_name).func  # type: ignore[return-value]


def hand_on_wall(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  wall_x: float = 0.0,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  root_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  dist = torch.abs(root_x - wall_x)
  return torch.exp(-dist / 0.05)


def hand_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2]


def engage_progress(
  env: ManagerBasedRlEnv,
  microspine_term_name: str = "microspine_adhesion",
  contact_threshold: float = 0.5,
) -> torch.Tensor:
  """Peak soft-count × (1 + current force level).

  Uses the controller's running-max ``peak_soft_count`` (the best
  simultaneous multi-sphere contact ratio achieved so far this episode)
  instead of the instantaneous soft_count. This way early steps with few
  contacts don't dilute the reward — once the policy achieves 8/8 contact
  even briefly, the peak stays at 1.0 for the rest of the episode and the
  reward shifts purely to the ``force_level`` term (press harder).

  ``contact_threshold`` (default 0.5 N) must be high enough to ignore the
  gap/margin-induced ghost forces that appear before physical contact.

  Max per-step reward = 1.0 × (1 + 1.0) = 2.0.
  """
  ctrl = _get_controller(env, microspine_term_name)
  engage_th = max(ctrl._mcfg.engage_force_threshold, 1e-6)
  forces = ctrl.last_normal_forces  # (E, S)

  force_level = (forces / engage_th).clamp(0.0, 1.0).mean(dim=1)

  return ctrl.peak_soft_count * (1.0 + force_level)


def engaged_bonus(
  env: ManagerBasedRlEnv,
  microspine_term_name: str = "microspine_adhesion",
) -> torch.Tensor:
  """+1 per engaged hand, averaged over hands. Sparse bonus once latched."""
  ctrl = _get_controller(env, microspine_term_name)
  return ctrl.engaged.float().mean(dim=1)


def friction_up_penalty(
  env: ManagerBasedRlEnv,
  microspine_term_name: str = "microspine_adhesion",
  threshold: float = 50.0,
  power: float = 2.0,
) -> torch.Tensor:
  """Soft penalty = (smoothed_fric_up / threshold)^power.

  Uses ``last_friction_up_smoothed`` (EMA) for low variance, then raises to
  ``power`` (default 2) to compress low values:
    fric_up=10 N → 0.04 (almost free, normal contact friction)
    fric_up=25 N → 0.25
    fric_up=40 N → 0.64
    fric_up=50 N → 1.0  (full)

  This avoids continuously punishing the policy for normal contact friction
  (which is huge with μ=30) while still creating a strong gradient near the
  disengage cliff.
  """
  ctrl = _get_controller(env, microspine_term_name)
  fric_up = (ctrl.last_friction_up_smoothed.clamp(min=0.0) / threshold).clamp(0.0, 1.0)
  return (fric_up ** power).mean(dim=1)


def contact_lost_penalty(
  env: ManagerBasedRlEnv,
  microspine_term_name: str = "microspine_adhesion",
  streak_norm: float = 100.0,
) -> torch.Tensor:
  """Streak-weighted penalty fired the step physical contact is lost.

  Per-sphere ``last_streak_at_loss`` → take the max across spheres within
  each hand (the longest-held sphere that just lost contact) → sum hands.
  - Brief contacts (streak < ``streak_norm``) → fractional penalty
  - Long contacts → full -1 per hand (× weight)
  """
  ctrl = _get_controller(env, microspine_term_name)
  factor = (ctrl.last_streak_at_loss.float() / streak_norm).clamp(0.0, 1.0)  # (E, S)
  per_hand = factor.view(factor.shape[0], ctrl._num_hands, -1).max(dim=-1).values  # (E, H)
  return per_hand.sum(dim=1)


def disengage_penalty(
  env: ManagerBasedRlEnv,
  microspine_term_name: str = "microspine_adhesion",
  streak_norm: float = 100.0,
) -> torch.Tensor:
  """Streak-weighted penalty: free for short attempts, full for long engages.

  Returns (per env) ``sum_h disengaged_now[h] * clamp(streak[h] / streak_norm, 0, 1)``.
  - Disengaging within ``streak_norm`` steps of engage → fractional penalty
  - Disengaging from ``streak_norm`` or more contiguous steps → full -1 (per hand)

  This avoids the "avoid engage to skip the penalty" trap: brief exploratory
  engages cost nothing, only abandoning a stable long contact hurts.
  """
  ctrl = _get_controller(env, microspine_term_name)
  is_disengaging = ctrl.disengaged_this_step.float()  # (E, H)
  # Per-sphere streak → max across spheres per hand.
  streak = ctrl.contact_current_streak.float()  # (E, S)
  streak_per_hand = streak.view(
    streak.shape[0], ctrl._num_hands, -1
  ).max(dim=-1).values  # (E, H)
  streak_factor = (streak_per_hand / streak_norm).clamp(0.0, 1.0)
  return (is_disengaging * streak_factor).sum(dim=1)


def contact_continuous_time(
  env: ManagerBasedRlEnv,
  microspine_term_name: str = "microspine_adhesion",
  top_k: int = 4,
  tau: float = 0.5,
) -> torch.Tensor:
  """Saturating per-step reward on the top-k longest per-sphere contact streaks.

  Per-sphere streaks are tracked in the controller. For each hand we take
  the ``top_k`` longest sphere streaks and average them, then pass through
  ``tanh(streak / tau)``. With ``tau=0.5`` (default) the reward hits
  ~0.995 of max at 1.5 s of contact (= 3·tau), which matches the phase-1
  engage window. Longer streaks give no extra credit, so the policy is
  pushed to *start* earning the reward quickly rather than dragging out
  a single sphere for the whole episode.

  Returns ``tanh(topk_mean_streak_seconds / tau)`` averaged across hands.
  """
  ctrl = _get_controller(env, microspine_term_name)
  streak_seconds = ctrl.contact_current_streak.float() * env.step_dt  # (E, S)
  # Reshape (E, S) → (E, H, spheres_per_hand), take top-k mean per hand.
  per_hand = streak_seconds.view(streak_seconds.shape[0], ctrl._num_hands, -1)
  k = min(top_k, per_hand.shape[-1])
  per_hand_topk = per_hand.topk(k, dim=-1).values.mean(dim=-1)  # (E, H)
  return torch.tanh(per_hand_topk / tau).mean(dim=1)


def engaged_time(
  env: ManagerBasedRlEnv,
  microspine_term_name: str = "microspine_adhesion",
  tau: float = 0.5,
) -> torch.Tensor:
  """Saturating reward for continuous time in the ENGAGED (latched) state.

  Uses the controller's per-hand engage step counter. Returns
  ``tanh(engage_seconds / tau)`` averaged across hands, so the reward ramps
  quickly for the first ~3·tau seconds of continuous engagement and then
  saturates. This gives the policy a strong incentive to *maintain*
  engagement once achieved, not just trigger it briefly.
  """
  ctrl = _get_controller(env, microspine_term_name)
  engage_seconds = ctrl._engage_steps.float() * env.step_dt  # (E, H)
  return torch.tanh(engage_seconds / tau).mean(dim=1)


def is_engaged_obs(
  env: ManagerBasedRlEnv,
  microspine_term_name: str = "microspine_adhesion",
) -> torch.Tensor:
  """Observation: per-hand engaged latch as float (E, num_hands)."""
  ctrl = _get_controller(env, microspine_term_name)
  return ctrl.engaged.float()


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """Rotate vec by quaternion (wxyz). Batched: quat (E,4), vec (3,) → (E,3)."""
  w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
  vx, vy, vz = vec[0], vec[1], vec[2]
  tx = 2.0 * (y * vz - z * vy)
  ty = 2.0 * (z * vx - x * vz)
  tz = 2.0 * (x * vy - y * vx)
  rx = vx + w * tx + (y * tz - z * ty)
  ry = vy + w * ty + (z * tx - x * tz)
  rz = vz + w * tz + (x * ty - y * tx)
  return torch.stack([rx, ry, rz], dim=-1)


_SPHERE_ID_CACHE: dict[str, list[int]] = {}
_HAND_ID_CACHE: dict[str, int] = {}


def engagement_drag(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  microspine_term_name: str = "microspine_adhesion",
  target_dir: tuple[float, float, float] = (-0.70710678, 0.0, 0.70710678),
  force_clip: float = 200.0,
) -> torch.Tensor:
  """Reward out-of-wall + upward contact reaction force on the hand.

  Microspine engagement requires the hand to press *diagonally into the wall
  and downward* (press: +x into wall, -z drag). When the spines correctly
  engage, the reaction force the hand feels is exactly opposite — diagonal
  *out of the wall and upward* — which is what this reward measures.

  Concretely: sum the world-frame contact force (normal + friction) across
  all spheres on each hand, project onto ``target_dir``, clamp negative
  values to 0, and normalize by ``force_clip`` so the reward is bounded in
  [0, 1]. Averaged across hands.

  Why this is better than the old velocity-based engagement_drag:
  - Rewards the *result* of good engagement (reaction force in the right
    direction) instead of the instantaneous hand velocity. This avoids the
    chatter local optimum where the policy oscillates up/down while barely
    touching the wall to harvest −v_z at contact onset.
  - Bounded to [0, 1] instead of unbounded m/s, so PPO advantage variance
    stays well-behaved.
  - Gated by physics, not by a ``max(force) > 1e-2`` heuristic: if the hand
    isn't actually loading the spines, the projected reaction force is near
    zero and the reward is zero.

  Args:
    target_dir: Desired reaction-force direction in world frame. Default
      ``(-0.707, 0, 0.707)`` = (out-of-wall, 0, upward) for a wall at +x.
    force_clip: Force [N] at which the reward saturates at 1.0.
  """
  ctrl = _get_controller(env, microspine_term_name)
  force_w = ctrl.last_hand_contact_force_world  # (E, H, 3)
  device = force_w.device
  target = torch.tensor(target_dir, dtype=torch.float32, device=device)
  target = target / target.norm().clamp(min=1e-6)
  proj = (force_w * target).sum(dim=-1)  # (E, H)
  reward = (proj / force_clip).clamp(0.0, 1.0)
  return reward.mean(dim=1)


def palm_orientation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  hand_body_names: tuple[str, ...] = ("left_hand", "right_hand"),
  palm_local_dir: tuple[float, float, float] = (-1.0, 0.0, 0.0),
  world_target: tuple[float, float, float] = (1.0, 0.0, 0.0),
  sigma: float = 0.1,
) -> torch.Tensor:
  """Reward the hand palm normal aligning with the wall normal.

  Computes the world direction of ``palm_local_dir`` (phybot hand's palm is
  local -x) and dot-products it with ``world_target`` (+x = into wall).
  Returns ``exp((dot - 1) / sigma)`` averaged across hands.
  """
  asset: Entity = env.scene[asset_cfg.name]
  local_dir = torch.tensor(palm_local_dir, dtype=torch.float32, device=env.device)
  target = torch.tensor(world_target, dtype=torch.float32, device=env.device)

  total = torch.zeros(env.num_envs, device=env.device)
  for hand_body_name in hand_body_names:
    cache_key = f"{asset_cfg.name}:{hand_body_name}"
    if cache_key not in _HAND_ID_CACHE:
      ids, _ = asset.find_bodies([hand_body_name])
      _HAND_ID_CACHE[cache_key] = int(ids[0])
    hand_id = _HAND_ID_CACHE[cache_key]

    hand_quat = asset.data.body_link_quat_w[:, hand_id, :]  # (E, 4)
    palm_world = _quat_rotate(hand_quat, local_dir)  # (E, 3)
    dot = (palm_world * target).sum(dim=1)  # (E,)
    total += torch.exp((dot - 1.0) / sigma)

  return total / len(hand_body_names)


def hand_to_wall_dist(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  wall_asset_name: str = "wall",
  wall_half_x: float = 0.5,
  sphere_body_names: tuple[str, ...] = (
    "left_hand_c_0_0", "left_hand_c_0_1", "left_hand_c_0_2", "left_hand_c_0_3",
    "left_hand_c_1_0", "left_hand_c_1_1", "left_hand_c_1_2", "left_hand_c_1_3",
    "right_hand_c_0_0", "right_hand_c_0_1", "right_hand_c_0_2", "right_hand_c_0_3",
    "right_hand_c_1_0", "right_hand_c_1_1", "right_hand_c_1_2", "right_hand_c_1_3",
  ),
  sigma: float = 0.02,
  overshoot: float = 0.005,
) -> torch.Tensor:
  """Dense guide reward — bring the contact spheres up to and past the wall.

  Signal = ``exp(-signed_gap / sigma)`` where ``signed_gap`` is the distance
  from the mean sphere x to the wall inner face, clipped at ``-overshoot``
  (so penetration up to ``overshoot`` meters keeps growing the reward).
  - gap > 0 (away):           reward in (0, 1)
  - gap = 0 (just touching):  reward = 1
  - gap < 0 (pressing in):    reward in (1, exp(overshoot/sigma)]  ← bonus
  - gap < -overshoot:         clamps so the reward doesn't blow up

  This kills the gap=0 plateau that traps the policy just outside contact.
  """
  asset: Entity = env.scene[asset_cfg.name]
  wall: Entity = env.scene[wall_asset_name]

  cache_key = f"{asset_cfg.name}:{wall_asset_name}"
  if cache_key not in _SPHERE_ID_CACHE:
    ids, _ = asset.find_bodies(list(sphere_body_names))
    _SPHERE_ID_CACHE[cache_key] = list(ids)
  sphere_ids = _SPHERE_ID_CACHE[cache_key]

  sphere_x = asset.data.body_link_pos_w[:, sphere_ids, 0]  # (E, num_spheres)
  mean_x = sphere_x.mean(dim=1)  # (E,)
  wall_inner_x = wall.data.root_link_pos_w[:, 0] - wall_half_x  # (E,)
  signed_gap = (wall_inner_x - mean_x).clamp(min=-overshoot)
  return torch.exp(-signed_gap / sigma)


_BASE_INIT_X_CACHE: dict[str, torch.Tensor] = {}


def base_forward_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  deadzone: float = 0.05,
) -> torch.Tensor:
  """Penalize base moving forward (toward wall, +x) beyond a deadzone.

  Returns ``clamp(base_x - init_x - deadzone, min=0) ** 2`` so small balance
  oscillations are free but sustained leaning toward the wall is penalized.
  """
  asset: Entity = env.scene[asset_cfg.name]
  cache_key = asset_cfg.name
  if cache_key not in _BASE_INIT_X_CACHE:
    # Capture env-local init base x from first call.
    init_x = (asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]).clone()
    _BASE_INIT_X_CACHE[cache_key] = init_x
  init_x = _BASE_INIT_X_CACHE[cache_key]

  current_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  forward = (current_x - init_x - deadzone).clamp(min=0.0)
  return forward ** 2


def torso_upright(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward upright orientation using projected gravity.

  Returns exp(-||g_xy||^2) where g_xy are x,y components of gravity in the
  base frame. Value is 1.0 when perfectly upright, decays toward 0 when tilted.
  """
  asset: Entity = env.scene[asset_cfg.name]
  gravity_b = asset.data.projected_gravity_b
  return torch.exp(-torch.sum(torch.square(gravity_b[:, :2]), dim=1))
