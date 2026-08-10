"""Curriculum functions for the wall-climbing task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import torch

from scr_phybot_mjab.assets.wall import WALL_OFFSET, WALL_PIVOT_X

from .commands import ClimbVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


class RewardWeightStage(TypedDict):
  iter: int
  weight: float


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
  steps_per_iter: int = 24,
) -> torch.Tensor:
  """Update a reward term's weight based on training iteration stages.

  Args:
    steps_per_iter: Number of env.step() calls per training iteration
      (must match num_steps_per_env in rl_cfg).
  """
  del env_ids  # Unused.
  current_iter = env.common_step_counter // steps_per_iter
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if current_iter > stage["iter"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


class WallAngleStage(TypedDict):
  iter: int
  angle_deg: float  # effective wall angle from horizontal. 90 = vertical wall
  # (gravity straight down). 30 = gentle 30°-from-horizontal "slope" — modeled
  # by tilting GRAVITY while keeping the wall geometry vertical.
  jitter_deg: NotRequired[float]  # per-stage override of the curriculum's
  # global ``angle_jitter_deg`` (e.g. a wide rung bridging a removed
  # neighbor). Still subject to ``jitter_below_deg``: the exact-90 stage
  # never jitters.


class GravityCurriculumStage(TypedDict):
  """One performance-gated gravity-tilt stage.

  ``angle_deg`` is the effective slope from horizontal simulated by tilting
  gravity (90 = vertical wall). ``min_height_gain`` is the base rise (m)
  since RSI placement an episode must reach to count as a success.
  """

  angle_deg: float
  min_height_gain: float


class performance_gated_gravity:
  """Advance the gravity-tilt angle from the episode height-gain success rate.

  DeepMimic-redesign replacement for the commanded-gait curriculum: the
  motion reference owns limb sequencing and RSI owns per-episode engagement,
  so the only stage knob left is the tilt angle, and the only success
  criterion is whether the robot actually climbed.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del env
    self.level = 0
    self.evaluated_episodes = 0
    self.successful_episodes = 0
    self.last_success_rate = 0.0
    # Episodes that BEGAN under a previous stage must not grade the current
    # one: stale successes advanced two levels in a single iteration and
    # skipped a designed rung outright (2026-07-17). Every env is stale until
    # its first reset under the freshly applied stage.
    self._stale: torch.Tensor | None = None
    stages = cfg.params.get("stages", [])
    if not stages:
      raise ValueError("performance_gated_gravity requires at least one stage.")

  @staticmethod
  def _set_gravity(
    env: ManagerBasedRlEnv,
    angle_deg: float,
    gravity_magnitude: float,
  ) -> None:
    import warp as wp

    rad = math.radians(angle_deg)
    grav_view = wp.to_torch(env.sim.wp_model.opt.gravity)
    grav_view[:, 0] = gravity_magnitude * math.cos(rad)
    grav_view[:, 1] = 0.0
    grav_view[:, 2] = -gravity_magnitude * math.sin(rad)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    *,
    stages: list[GravityCurriculumStage],
    motion_command_name: str = "climb_motion",
    min_evaluated_episodes: int = 512,
    advance_success_rate: float = 0.7,
    gravity_magnitude: float = 9.81,
  ) -> dict[str, torch.Tensor]:
    if not 0.0 <= advance_success_rate <= 1.0:
      raise ValueError("advance_success_rate must be in [0, 1].")
    if min_evaluated_episodes < 1:
      raise ValueError("min_evaluated_episodes must be positive.")

    # Curriculum compute runs first in ``_reset_idx``, before any state is
    # rewritten, so the terminal height gain of the resetting envs is intact.
    motion = env.command_manager.get_term(motion_command_name)
    stage = stages[self.level]
    if self._stale is None or self._stale.shape[0] != env.num_envs:
      self._stale = torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
      )
    fresh = ~self._stale[env_ids]
    completed = (env.episode_length_buf[env_ids] > 0) & fresh
    successes = (
      motion.height_gained[env_ids] >= stage["min_height_gain"]
    ) & completed
    self.evaluated_episodes += int(completed.sum().item())
    self.successful_episodes += int(successes.sum().item())
    # These envs restart now and will run entirely under the current stage.
    self._stale[env_ids] = False

    if self.evaluated_episodes >= min_evaluated_episodes:
      self.last_success_rate = (
        self.successful_episodes / self.evaluated_episodes
      )
      if (
        self.last_success_rate >= advance_success_rate
        and self.level + 1 < len(stages)
      ):
        self.level += 1
        self._stale.fill_(True)
      self.evaluated_episodes = 0
      self.successful_episodes = 0

    stage = stages[self.level]
    self._set_gravity(env, stage["angle_deg"], gravity_magnitude)

    device = env.device
    return {
      "stage": torch.tensor(float(self.level), device=device),
      "angle_deg": torch.tensor(stage["angle_deg"], device=device),
      "success_rate": torch.tensor(self.last_success_rate, device=device),
      "min_height_gain": torch.tensor(
        float(stage["min_height_gain"]), device=device
      ),
    }


class ClimbVelocityRangeStage(TypedDict, total=False):
  iter: int
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  lin_vel_z: tuple[float, float]
  vx_range: tuple[float, float]
  vy_range: tuple[float, float]
  vz_range: tuple[float, float]


class TiltPeelStage(TypedDict):
  iter: int
  deg_range: tuple[float, float]  # tilt_peel_max_deg_range for this stage
  tax_threshold_deg: NotRequired[float]  # engaged_tilt reward threshold
  tax_scale_deg: NotRequired[float]  # engaged_tilt reward scale


def tilt_peel_schedule(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  stages: list[TiltPeelStage],
  command_name: str = "microspine_affordance",
  tax_term_name: str | None = "engaged_tilt",
  steps_per_iter: int = 24,
) -> torch.Tensor:
  """Anneal the tilt-peel band toward the calibrated physics by iteration.

  The calibrated band (3.5-4.5 deg of engaged rotation from capture) is
  tighter than birth exploration: an untrained policy's flailing rotates
  pads past it before "hold on" can be learned at all (motionref18 cold
  start: every episode peeled apart within ~1 s and 300 iterations moved
  nothing). Like the gravity ladder, the ROAD to the honest physics is
  staged while the destination stays fixed: the cfg's static range is the
  final stage, benches and play builds (empty/pinned curriculum) always run
  the calibrated band, and per-grip draws pick up the current stage at
  their next latch (stage transitions blur naturally).

  The engaged_tilt TAX anneals on the same table (``tax_threshold_deg`` /
  ``tax_scale_deg``): taxing from the final 3 deg while the peel band sat
  at 12-15 priced the multi-anchor closed-chain wiggle beginners need and
  bred a one-anchor-dangle population (motionref18b iter 300: RF occupancy
  0.97, every other limb ~0.02) — the band and its approach warning must
  move together.
  """
  del env_ids  # Stage is a function of training time; draws happen per latch.
  current_iter = env.common_step_counter // steps_per_iter
  if not stages:
    raise ValueError("tilt_peel_schedule requires at least one stage.")
  current = stages[0]
  for stage in sorted(stages, key=lambda item: item["iter"]):
    if current_iter >= stage["iter"]:
      current = stage
    else:
      break
  affordance = env.command_manager.get_term(command_name)
  affordance._ls_cfg.tilt_peel_max_deg_range = tuple(current["deg_range"])
  if tax_term_name is not None and "tax_threshold_deg" in current:
    tax_cfg = env.reward_manager.get_term_cfg(tax_term_name)
    tax_cfg.params["threshold_deg"] = current["tax_threshold_deg"]
    if "tax_scale_deg" in current:
      tax_cfg.params["scale_deg"] = current["tax_scale_deg"]
  return torch.tensor([current["deg_range"][0]], device=env.device)


def resolve_gravity_stage(
  current_iter: int, angle_stages: list[WallAngleStage]
) -> WallAngleStage:
  """Resolve the active gravity curriculum stage by stepwise lookup."""
  if not angle_stages:
    raise ValueError("angle_stages must contain at least one stage.")

  current = angle_stages[0]
  for stage in sorted(angle_stages, key=lambda item: item["iter"]):
    if current_iter >= stage["iter"]:
      current = stage
    else:
      break
  return current


def resolve_gravity_angle(current_iter: int, angle_stages: list[WallAngleStage]) -> float:
  """Resolve the effective gravity curriculum angle by stepwise stage lookup."""
  return resolve_gravity_stage(current_iter, angle_stages)["angle_deg"]


# Training-schedule tables — gravity ladder x tilt-peel anneal, one timeline.
# The iters transcribe the ONE validated trajectory (v3 birth->3300 on the
# 45/75/tilt prefix, v4 bridge 3300->5050 through 85/88/exact-90 to the ship
# gate) and are CEILINGS driven by mastery gates; every gravity hop overlaps
# its neighbor's jitter pool, and the tilt anneal reaches the calibrated
# physics (3000) before the wall gets steep (3550). Full doctrine + design
# rules: docs/climb_env_cfg_notes.md ("Curriculum").
# Consumers must copy at use ([dict(s) for s in ...]): these tuples are shared
# module state and env cfgs are built eagerly at import time.
WALL_ANGLE_STAGES: tuple[WallAngleStage, ...] = (
  {"iter": 0, "angle_deg": 45.0},
  {"iter": 1500, "angle_deg": 75.0, "jitter_deg": 12.0},
  {"iter": 3550, "angle_deg": 85.0, "jitter_deg": 3.0},
  {"iter": 4200, "angle_deg": 88.0, "jitter_deg": 2.0},
  {"iter": 4800, "angle_deg": 90.0},
)
TILT_PEEL_STAGES: tuple[TiltPeelStage, ...] = (
  {
    "iter": 0,
    "deg_range": (12.0, 15.0),
    "tax_threshold_deg": 10.0,
    "tax_scale_deg": 8.0,
  },
  {
    "iter": 1200,
    "deg_range": (8.0, 10.0),
    "tax_threshold_deg": 6.5,
    "tax_scale_deg": 6.0,
  },
  {
    "iter": 2400,
    "deg_range": (5.0, 7.0),
    "tax_threshold_deg": 4.0,
    "tax_scale_deg": 5.0,
  },
  {
    "iter": 3000,
    "deg_range": (3.5, 4.5),
    "tax_threshold_deg": 3.0,
    "tax_scale_deg": 5.0,
  },
)


def pin_curricula_to_final_stage(cfg: ManagerBasedRlEnvCfg) -> None:
  """Pin the wall-angle ladder and tilt-peel anneal to their terminal stages.

  Play/eval and the distill/polish variants run the final task only: the
  single remaining stage is the ladder's last angle and the CALIBRATED
  tilt-peel band (the annealing ladder is a training aid, not physics).
  Deliberately leaves ``angle_jitter_deg`` untouched — play zeroes it
  separately for deterministic rollouts, while distill keeps the training
  jitter so rollouts cover the neighborhood the teacher graduated on.
  """
  final_angle = cfg.curriculum["wall_angle"].params["angle_stages"][-1]
  cfg.curriculum["wall_angle"].params["angle_stages"] = [
    {"iter": 0, "angle_deg": final_angle["angle_deg"]}
  ]
  final_tilt = cfg.curriculum["tilt_peel"].params["stages"][-1]
  cfg.curriculum["tilt_peel"].params["stages"] = [
    {"iter": 0, "deg_range": final_tilt["deg_range"]}
  ]


def resolve_climb_velocity_ranges(
  current_iter: int,
  range_stages: list[ClimbVelocityRangeStage],
) -> ClimbVelocityRangeStage:
  """Resolve command ranges for a training iteration.

  Stages are cumulative: later stages can specify only the axes they change.
  ``vx_range``/``vy_range``/``vz_range`` aliases are accepted for readability
  in climb-wall configs.
  """
  if not range_stages:
    raise ValueError("range_stages must contain at least one stage.")

  resolved: dict = {}
  for stage in sorted(range_stages, key=lambda item: item["iter"]):
    if current_iter >= stage["iter"]:
      resolved.update(stage)
  if not resolved:
    resolved.update(range_stages[0])
  return cast(ClimbVelocityRangeStage, resolved)


def _stage_range(
  stage: ClimbVelocityRangeStage, *keys: str
) -> tuple[float, float] | None:
  for key in keys:
    if key in stage:
      value = stage[key]  # type: ignore[literal-required]
      return (float(value[0]), float(value[1]))
  return None


def _apply_climb_velocity_range_stage(
  cfg: ClimbVelocityCommandCfg,
  stage: ClimbVelocityRangeStage,
) -> None:
  if (value := _stage_range(stage, "lin_vel_x", "vx_range")) is not None:
    cfg.ranges.lin_vel_x = value
  if (value := _stage_range(stage, "lin_vel_y", "vy_range")) is not None:
    cfg.ranges.lin_vel_y = value
  if (value := _stage_range(stage, "lin_vel_z", "vz_range")) is not None:
    cfg.ranges.lin_vel_z = value


def sample_gravity_angles(
  base_deg: float,
  jitter_deg: float,
  jitter_below_deg: float,
  count: int,
  device: torch.device | str = "cpu",
) -> torch.Tensor:
  """Per-episode angle draws for the gravity-tilt domain randomization.

  Uniform in ``base ± jitter`` while the stage base sits below
  ``jitter_below_deg``; at or above it (the vertical-wall stage) every draw
  is exactly the base — randomizing around 90° would mix in overhangs, which
  are a different task, not a harder version of this one.
  """
  if jitter_deg <= 0.0 or base_deg >= jitter_below_deg:
    return torch.full((count,), base_deg, device=device)
  return torch.empty(count, device=device).uniform_(
    base_deg - jitter_deg, base_deg + jitter_deg
  )


def wall_pose_for_angle(
  angle_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Env-local wall-body pose for a wall tilted to ``angle_deg``.

  ``angle_deg`` is the slope from horizontal (90 = vertical wall = the
  compiled identity pose; smaller angles lean the wall away from the robot
  like a ramp). The rotation is ``Rot_y(90 - angle)`` about the base line
  ``(x=WALL_PIVOT_X, z=0)``: ``pos' = pivot + R (offset - pivot)``,
  ``quat' = R`` (wxyz). This is the world-rotation image of the retired
  gravity-tilt representation — at a given angle the robot-relative physics
  is identical, gravity stays physically down.

  Returns ``(pos (N, 3), quat (N, 4))`` in the env-local frame of
  ``EntityData.default_root_state`` (reset events add env_origins).
  """
  alpha = torch.deg2rad(90.0 - angle_deg)
  ca = torch.cos(alpha)
  sa = torch.sin(alpha)
  rel_x = WALL_OFFSET[0] - WALL_PIVOT_X
  rel_z = WALL_OFFSET[2]
  pos = torch.stack(
    [
      WALL_PIVOT_X + rel_x * ca + rel_z * sa,
      torch.full_like(ca, float(WALL_OFFSET[1])),
      -rel_x * sa + rel_z * ca,
    ],
    dim=-1,
  )
  half = alpha * 0.5
  zeros = torch.zeros_like(half)
  quat = torch.stack([torch.cos(half), zeros, torch.sin(half), zeros], dim=-1)
  return pos, quat


def wall_angle(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  angle_stages: list[WallAngleStage],
  steps_per_iter: int = 24,
  angle_jitter_deg: float = 0.0,
  jitter_below_deg: float = 90.0,
  wall_entity_name: str = "wall",
) -> torch.Tensor:
  """Curriculum that tilts the WALL to ``angle_deg`` under real gravity.

  Replaces the gravity-tilt representation: gravity stays ``(0, 0, -g)`` and
  the wall (an auto-wrapped mocap entity) rotates about its base line, with
  the robot spawn and motion reference rotating along (the motion command
  reads the wall pose at reset). At 90° this is bit-identical to the old
  exact-90 world; the terminal rung therefore matches climb_transition's
  world (vertical wall, real gravity).

  Per-episode jitter matches the gravity machinery (per-stage ``jitter_deg``
  override, no jitter at/above ``jitter_below_deg`` — overhangs are a
  different task). One deliberate difference: a resetting env picks up the
  CURRENT stage's pose while running envs keep theirs until their own reset
  — a wall must never rotate under latched anchors, so stage flips blur over
  at most one episode (same spirit as tilt_peel's per-latch pickup).

  Writes ``wall.data.default_root_state[env_ids, 0:7]``; the stock
  ``reset_scene_to_default`` event applies it to the mocap pose (with
  env_origins) later in the same reset pass.
  """
  current_iter = env.common_step_counter // steps_per_iter
  stage = resolve_gravity_stage(current_iter, angle_stages)
  base_deg = float(stage["angle_deg"])
  stage_jitter = float(stage.get("jitter_deg", angle_jitter_deg))
  if env_ids is not None and len(env_ids) > 0:
    angles = sample_gravity_angles(
      base_deg,
      stage_jitter,
      jitter_below_deg,
      len(env_ids),
      device=env.device,
    )
    pos, quat = wall_pose_for_angle(angles)
    wall = env.scene[wall_entity_name]
    wall.data.default_root_state[env_ids, 0:3] = pos
    wall.data.default_root_state[env_ids, 3:7] = quat
  return torch.tensor([base_deg])


def _ensure_per_world_gravity(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Torch view of a PER-WORLD gravity array, expanding it on first use.

  mjwarp physics kernels read ``opt_gravity[worldid % shape[0]]``, so a
  (num_envs,) gravity array is honored per environment — but mjlab allocates
  it shared (shape 1) and its ``expand_model_fields`` only handles top-level
  model fields, not the nested ``opt.gravity``. Expansion replaces the warp
  array, and the captured CUDA graphs still hold the OLD array's pointer, so
  the graphs must be re-captured afterwards (``create_graph``) — the exact
  procedure mjlab documents for replaced arrays. One-time cost at first use.
  """
  import warp as wp

  grav = env.sim.wp_model.opt.gravity
  if grav.shape[0] != env.num_envs:
    tiled = wp.zeros(env.num_envs, dtype=wp.vec3, device=grav.device)
    wp.to_torch(tiled)[:] = wp.to_torch(grav)[0]
    env.sim.wp_model.opt.gravity = tiled
    bridge = getattr(env.sim, "_model_bridge", None)
    if bridge is not None:
      bridge.clear_cache()
    env.sim.create_graph()
  return wp.to_torch(env.sim.wp_model.opt.gravity)


def gravity_angle(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  angle_stages: list[WallAngleStage],
  gravity_magnitude: float = 9.81,
  steps_per_iter: int = 24,
  angle_jitter_deg: float = 0.0,
  jitter_below_deg: float = 90.0,
) -> torch.Tensor:
  """Curriculum that varies gravity direction to simulate climbing on a slope
  of angle ``angle_deg`` from horizontal, while keeping the wall geometry
  vertical and removing the need for a floor.

  At ``angle_deg = 90°``: gravity = (0, 0, -g) — standard vertical wall climb.
  At ``angle_deg = 30°``: gravity = g · (cos 30°, 0, -sin 30°) — tilted so it
  mostly pushes the robot INTO the wall (+x), with only a small component
  along the wall's up direction. This is physically equivalent to the robot
  sitting on a gentle 30° slope.

  With ``angle_jitter_deg`` > 0, stages below ``jitter_below_deg`` run
  per-episode domain randomization: each env resetting NOW draws its own
  angle uniformly in ``base ± jitter`` and keeps it for the whole episode
  (mid-episode gravity never moves). The shared gravity array is expanded to
  per-world on first use (see ``_ensure_per_world_gravity``); jitter-free
  stages write every row to the base, so the 90° rung is exactly vertical
  for everyone the moment it starts.

  Writes gravity in-place on ``env.sim.wp_model.opt.gravity`` via a torch
  view — never reallocates the warp array (CUDA graphs would otherwise still
  read from the old array). ``gravity_vec_w`` on each entity is deliberately
  NOT updated: orientation terms that read ``EntityData.projected_gravity_b``
  continue to measure alignment with world-vertical, which is the correct
  "up-the-wall" reference for climbing. Observation terms that need the
  physical gravity direction should read ``env.sim.wp_model.opt.gravity``.
  """
  import warp as wp

  current_iter = env.common_step_counter // steps_per_iter
  stage = resolve_gravity_stage(current_iter, angle_stages)
  angle_deg = stage["angle_deg"]
  # Per-stage jitter override (e.g. a widened rung bridging a removed
  # neighbor); falls back to the curriculum-wide angle_jitter_deg.
  stage_jitter = float(stage.get("jitter_deg", angle_jitter_deg))

  jitter_wanted = stage_jitter > 0.0 and angle_deg < jitter_below_deg
  if jitter_wanted:
    grav_view = _ensure_per_world_gravity(env)  # (num_envs, 3)
  else:
    grav_view = wp.to_torch(env.sim.wp_model.opt.gravity)  # (N, 3)
  jittered = jitter_wanted and grav_view.shape[0] == env.num_envs
  if jittered:
    if env_ids is not None and len(env_ids) > 0:
      angles = sample_gravity_angles(
        angle_deg,
        stage_jitter,
        jitter_below_deg,
        len(env_ids),
        device=grav_view.device,
      )
      rad = torch.deg2rad(angles)
      grav_view[env_ids, 0] = gravity_magnitude * torch.cos(rad)
      grav_view[env_ids, 1] = 0.0
      grav_view[env_ids, 2] = -gravity_magnitude * torch.sin(rad)
  else:
    rad = math.radians(angle_deg)
    grav_view[:, 0] = gravity_magnitude * math.cos(rad)
    grav_view[:, 1] = 0.0
    grav_view[:, 2] = -gravity_magnitude * math.sin(rad)

  return torch.tensor([angle_deg], device=env.device)


def climb_velocity_command_ranges(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  range_stages: list[ClimbVelocityRangeStage],
  steps_per_iter: int = 24,
) -> dict[str, torch.Tensor]:
  """Curriculum that updates climb velocity command ranges by iteration."""
  del env_ids  # Unused.
  current_iter = env.common_step_counter // steps_per_iter
  stage = resolve_climb_velocity_ranges(current_iter, range_stages)

  command_term = env.command_manager.get_term(command_name)
  cfg = command_term.cfg
  if not isinstance(cfg, ClimbVelocityCommandCfg):
    raise TypeError(
      f"Command term '{command_name}' must use ClimbVelocityCommandCfg, "
      f"got {type(cfg).__name__}."
    )
  _apply_climb_velocity_range_stage(cfg, stage)

  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0], device=env.device),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1], device=env.device),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0], device=env.device),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1], device=env.device),
    "lin_vel_z_min": torch.tensor(cfg.ranges.lin_vel_z[0], device=env.device),
    "lin_vel_z_max": torch.tensor(cfg.ranges.lin_vel_z[1], device=env.device),
  }
