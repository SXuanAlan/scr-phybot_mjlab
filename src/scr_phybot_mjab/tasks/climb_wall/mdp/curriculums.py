"""Curriculum functions for the wall-climbing task."""

from __future__ import annotations

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
  steps_per_iter: int = 96,  # matches BOTH live tasks' rl_cfg
  # num_steps_per_env (was 24 until 08-19 — a fossil from the early climb
  # era; a full-repo audit found every live wiring passes this explicitly:
  # climb via the tested _STEPS_PER_ITER constant, transition via literal
  # 96s plus a test guard. walk has its own step-keyed scheduler).
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
  angle_deg: float  # wall angle from horizontal. 90 = vertical wall; smaller
  # = the wall leans away from the robot like a ramp. Realized by TILTING THE
  # WALL about its base line under real gravity (mdp.wall_angle).
  jitter_deg: NotRequired[float]  # per-stage override of the curriculum's
  # global ``angle_jitter_deg`` (e.g. a wide rung bridging a removed
  # neighbor). Still subject to ``jitter_below_deg``: the exact-90 stage
  # never jitters.


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
  steps_per_iter: int = 96,  # matches BOTH live tasks' rl_cfg
  # num_steps_per_env (was 24 until 08-19 — a fossil from the early climb
  # era; a full-repo audit found every live wiring passes this explicitly:
  # climb via the tested _STEPS_PER_ITER constant, transition via literal
  # 96s plus a test guard. walk has its own step-keyed scheduler).
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


def resolve_wall_stage(
  current_iter: int, angle_stages: list[WallAngleStage]
) -> WallAngleStage:
  """Resolve the active wall-angle curriculum stage by stepwise lookup."""
  if not angle_stages:
    raise ValueError("angle_stages must contain at least one stage.")

  current = angle_stages[0]
  for stage in sorted(angle_stages, key=lambda item: item["iter"]):
    if current_iter >= stage["iter"]:
      current = stage
    else:
      break
  return current


def resolve_wall_angle(current_iter: int, angle_stages: list[WallAngleStage]) -> float:
  """Resolve the effective wall-angle curriculum angle by stepwise lookup."""
  return resolve_wall_stage(current_iter, angle_stages)["angle_deg"]


# Training-schedule tables — wall-angle ladder x tilt-peel anneal, one timeline.
# The iters transcribe the ONE validated trajectory (v3 birth->3300 on the
# 45/75/tilt prefix, v4 bridge 3300->5050 through 85/88/exact-90 to the ship
# gate) and are CEILINGS driven by mastery gates; every angle hop overlaps
# its neighbor's jitter pool, and the tilt anneal reaches the calibrated
# physics (3000) before the wall gets steep (3300+). Full doctrine + design
# rules: docs/archive/climb_env_cfg_notes.md ("Curriculum").
# 2026-08-10 six-stage extension: the 80+-5 rung is inserted for the
# band-teacher matrix (deploy bands 80+-5 / 88+-2 / 90 each get a minted
# teacher at its rung's end — the per-band mount+climb pipeline in
# docs/plans/). Later rungs shift +350; every boundary re-checked >=300
# iters from a tilt-anneal step and every pool still overlaps its neighbor
# (75+-12 = 63-87 covers 80+-5 = 75-85 covers 85+-3 = 82-88).
# Consumers must copy at use ([dict(s) for s in ...]): these tuples are shared
# module state and env cfgs are built eagerly at import time.
WALL_ANGLE_STAGES: tuple[WallAngleStage, ...] = (
  {"iter": 0, "angle_deg": 45.0},
  {"iter": 1500, "angle_deg": 75.0, "jitter_deg": 12.0},
  {"iter": 3300, "angle_deg": 80.0, "jitter_deg": 5.0},
  {"iter": 3900, "angle_deg": 85.0, "jitter_deg": 3.0},
  {"iter": 4550, "angle_deg": 88.0, "jitter_deg": 2.0},
  {"iter": 5150, "angle_deg": 90.0},
)
# Reduced-hardware task: keep the legacy iteration-driven ladder and
# per-reset angle draws, with an intermediate band bounded by the 60-degree
# target. The terminal rung must disable jitter explicitly (unlike 90, it
# is not exempted by sample_wall_angles). Copy these dicts into each cfg.
CORNERS60_WALL_ANGLE_STAGES: tuple[WallAngleStage, ...] = (
  {"iter": 0, "angle_deg": 45.0, "jitter_deg": 5.0},
  {"iter": 1500, "angle_deg": 55.0, "jitter_deg": 5.0},
  {"iter": 3300, "angle_deg": 60.0, "jitter_deg": 0.0},
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
  The pinned stage carries no ``jitter_deg``, and with the terminal rung at
  90 the sampler is jitter-exempt anyway (``jitter_below_deg``), so pinned
  runs execute a HARD single angle — for a band-jittered pinned run use
  :func:`pin_curricula_to_band` instead.
  """
  final_angle = cfg.curriculum["wall_angle"].params["angle_stages"][-1]
  cfg.curriculum["wall_angle"].params["angle_stages"] = [
    {"iter": 0, "angle_deg": final_angle["angle_deg"]}
  ]
  # Gimbal-era cfgs carry no tilt-peel ladder (angle release retired).
  if "tilt_peel" in cfg.curriculum:
    final_tilt = cfg.curriculum["tilt_peel"].params["stages"][-1]
    cfg.curriculum["tilt_peel"].params["stages"] = [
      {"iter": 0, "deg_range": final_tilt["deg_range"]}
    ]


def pin_curricula_to_band(
  cfg: ManagerBasedRlEnvCfg, angle_deg: float, jitter_deg: float
) -> None:
  """Pin the ladder to a single BAND stage — jitter preserved.

  The per-band knob of the band-teacher matrix: soak/distill/eval runs that
  must live inside one deploy band (80+-5, 88+-2) rather than at a hard
  angle. Writes one stage carrying the band half-width and pins tilt-peel
  to the calibrated physics, exactly like the final-stage pin. Note the
  sampler clamps nothing at 90: a band like 88+-2 legitimately samples up
  to 90, and a base at/above ``jitter_below_deg`` (default 90) stays exact.
  """
  cfg.curriculum["wall_angle"].params["angle_stages"] = [
    {"iter": 0, "angle_deg": angle_deg, "jitter_deg": jitter_deg}
  ]
  # Gimbal-era cfgs carry no tilt-peel ladder (angle release retired).
  if "tilt_peel" in cfg.curriculum:
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


def sample_wall_angles(
  base_deg: float,
  jitter_deg: float,
  jitter_below_deg: float,
  count: int,
  device: torch.device | str = "cpu",
) -> torch.Tensor:
  """Per-episode angle draws for the wall-tilt domain randomization.

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
  steps_per_iter: int = 96,  # matches BOTH live tasks' rl_cfg
  # num_steps_per_env (was 24 until 08-19 — a fossil from the early climb
  # era; a full-repo audit found every live wiring passes this explicitly:
  # climb via the tested _STEPS_PER_ITER constant, transition via literal
  # 96s plus a test guard. walk has its own step-keyed scheduler).
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
  stage = resolve_wall_stage(current_iter, angle_stages)
  base_deg = float(stage["angle_deg"])
  stage_jitter = float(stage.get("jitter_deg", angle_jitter_deg))
  if env_ids is not None and len(env_ids) > 0:
    angles = sample_wall_angles(
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


def climb_velocity_command_ranges(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  range_stages: list[ClimbVelocityRangeStage],
  steps_per_iter: int = 96,  # matches BOTH live tasks' rl_cfg
  # num_steps_per_env (was 24 until 08-19 — a fossil from the early climb
  # era; a full-repo audit found every live wiring passes this explicitly:
  # climb via the tested _STEPS_PER_ITER constant, transition via literal
  # 96s plus a test guard. walk has its own step-keyed scheduler).
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
