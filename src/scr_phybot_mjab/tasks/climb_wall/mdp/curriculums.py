"""Curriculum functions for the wall-climbing task."""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict

import torch

from scr_phybot_mjab.assets.wall import WALL_OFFSET, WALL_PIVOT_X

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

__all__ = (
  "CORNERS60_WALL_ANGLE_STAGES",
  "RewardWeightStage",
  "WALL_ANGLE_STAGES",
  "WallAngleStage",
  "pin_curricula_to_final_stage",
  "resolve_wall_angle",
  "resolve_wall_stage",
  "reward_weight",
  "sample_wall_angles",
  "wall_angle",
  "wall_pose_for_angle",
)


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


def resolve_wall_stage(current_iter: int, angle_stages: list[WallAngleStage]) -> WallAngleStage:
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


# Wall-angle ladder. The iterations are ceilings driven by mastery gates, and
# every angle hop overlaps its neighbor's jitter pool (75+-12 = 63-87 covers
# 80+-5 = 75-85 covers 85+-3 = 82-88). Design rules:
# docs/archive/climb_env_cfg_notes.md ("Curriculum").
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


def pin_curricula_to_final_stage(cfg: ManagerBasedRlEnvCfg) -> None:
  """Pin the wall-angle ladder to its terminal stage.

  Play/eval runs the final task only. The pinned stage carries no
  ``jitter_deg``, and with the terminal rung at 90 the sampler is
  jitter-exempt anyway (``jitter_below_deg``), so pinned runs execute a HARD
  single angle.
  """
  final_angle = cfg.curriculum["wall_angle"].params["angle_stages"][-1]
  cfg.curriculum["wall_angle"].params["angle_stages"] = [{"iter": 0, "angle_deg": final_angle["angle_deg"]}]


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
  return torch.empty(count, device=device).uniform_(base_deg - jitter_deg, base_deg + jitter_deg)


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
  at most one episode.

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
