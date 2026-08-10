"""DeepMimic-style motion reference for the wall-climbing task.

Consumes the IK cycle clip written by ``scripts/gen_climb_clip.py``
(``climb_cycle.npy``, the AMP/motion ``.npy`` dict schema plus climb keys).
The command owns three things:

  * the per-env phase index into one regrip cycle (advances every policy
    step and wraps — the reference is a loop, not a finite take);
  * reference-state initialization (RSI): on reset each env is placed at a
    sampled clip frame (root pose + joint state + finite-difference
    velocities) and the microspine latches are seeded from the frame's
    ``ee_engaged`` flags;
  * the reference quantities the tracking rewards compare against.

Deliberately NOT provided: a global root position target. The clip's root
trajectory is used only to place the robot at reset and to draw the ghost;
climbing progress is driven by the height/velocity rewards, so the tracking
terms are all root-relative (joint space, root-frame EE positions, root
orientation).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_mul,
  sample_uniform,
)

from scr_phybot_mjab.assets.wall import WALL_PIVOT_X

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_GHOST_COLOR = (0.5, 0.7, 0.5, 0.5)


def vertical_height_spawn_offset(
  clip_root_pos: torch.Tensor,
  wall_quat: torch.Tensor,
  pivot_x: float,
) -> torch.Tensor:
  """Along-wall shift restoring each frame's VERTICAL-wall spawn height.

  The pure base-line rotation moves a spawn's world z with the wall (down at
  shallow angles — ever closer to the floor). Sliding the whole RSI reference
  along the tilted surface by this offset puts the root back at the world
  height the same clip frame has on the vertical wall, while pads, anchors
  and pose keep conforming to the surface (on-wall points recover their 90°
  world heights to within the base-standoff correction). Solves
  ``-rel_x sin(a) + (z + h) cos(a) = z`` for ``h``; exactly zero at identity.
  """
  alpha = 2.0 * torch.atan2(wall_quat[:, 2], wall_quat[:, 0])
  ca = torch.cos(alpha)
  sa = torch.sin(alpha)
  rel_x = clip_root_pos[:, 0] - pivot_x
  target_z = clip_root_pos[:, 2]
  shift = (target_z + rel_x * sa) / ca - target_z
  vertical = wall_quat[:, 0] >= 1.0 - 1.0e-9
  return torch.where(vertical, torch.zeros_like(shift), shift)

# Nominal cycle climb rate (m/s) used only when the clip file is absent at
# config-build time (fresh checkout before ``gen_climb_clip.py`` ran). Env
# construction still fails loudly without the file — this fallback exists so
# building the *config* (task registration, the gen script itself) does not
# require the asset it is about to generate. 0.06 m / 4.3 s cycle.
_FALLBACK_CLIMB_RATE = 0.06 / 4.3


def _np_quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Hamilton product of wxyz quaternion arrays (..., 4)."""
  aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
  bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  return np.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    axis=-1,
  )


def _np_quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate vectors ``v`` (T, N, 3) by the inverse of wxyz quats ``q`` (T, 4)."""
  w = q[:, None, 0:1]
  im = -q[:, None, 1:4]  # conjugate
  t = 2.0 * np.cross(im, v)
  return v + w * t + np.cross(im, t)


def _lerp_frames(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
  """Linear interpolation of a (T, ...) array at fractional frame indices."""
  last = arr.shape[0] - 1
  i0 = np.clip(np.floor(pos).astype(np.int64), 0, last)
  i1 = np.minimum(i0 + 1, last)
  w = (pos - i0).reshape((-1,) + (1,) * (arr.ndim - 1))
  return (1.0 - w) * arr[i0] + w * arr[i1]


def _nearest_frames(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
  last = arr.shape[0] - 1
  return arr[np.clip(np.round(pos).astype(np.int64), 0, last)]


def _edge_frame_countdowns(engaged: np.ndarray) -> np.ndarray:
  """Per-frame countdowns (in frames) to each EE's next schedule edges.

  For frame ``c`` and EE ``h``, entry ``[c, h, 0]`` counts the frames until
  the schedule next turns that EE OFF (its next release) and ``[c, h, 1]``
  until it next turns ON (its next regrip), both wrapping around the cycle.
  Together with the current engage bit these two edges reconstruct the EE's
  entire periodic schedule row — the whole raster chart ahead, not point
  samples of it. A row with no edge (an EE the clip never releases) reads
  the full cycle length: "a whole cycle away", forever.
  """
  cycle, num_ee = engaged.shape
  out = np.full((cycle, num_ee, 2), cycle, dtype=np.int64)
  frames = np.arange(cycle)
  for h in range(num_ee):
    now = engaged[:, h]
    nxt = np.roll(now, -1)
    for k, edge in enumerate((now & ~nxt, ~now & nxt)):  # falling, rising
      event_frames = (np.flatnonzero(edge) + 1) % cycle
      if event_frames.size == 0:
        continue
      dist = (event_frames[None, :] - frames[:, None]) % cycle
      dist[dist == 0] = cycle
      out[:, h, k] = dist.min(axis=1)
  return out


def _run_lengths(engaged: np.ndarray) -> np.ndarray:
  """Per-frame length (frames) of the circular ON/OFF run containing it.

  For frame ``c`` and EE ``h`` this is the length of the maximal constant
  stretch of ``engaged[:, h]`` that frame ``c`` sits in, with wrap-around
  (the schedule is a loop). An EE the clip never flips reads the full cycle.
  Consumed by the window-proportional mismatch grace: a weld-through bill
  then costs a constant fraction of the violated window regardless of how
  the clip is timed.
  """
  cycle, num_ee = engaged.shape
  out = np.full((cycle, num_ee), cycle, dtype=np.int64)
  for h in range(num_ee):
    col = engaged[:, h]
    starts = np.flatnonzero(col != np.roll(col, 1))  # circular run starts
    if starts.size == 0:
      continue
    for k, s in enumerate(starts):
      e = starts[(k + 1) % starts.size]
      length = (e - s) % cycle
      if length == 0:
        length = cycle
      out[(np.arange(length) + s) % cycle, h] = length
  return out


def _cycle_frames(data: dict) -> int:
  """Frames of one loopable cycle: cycle 0 without the ``end_hold`` tail.

  ``gen_climb_clip`` appends ``end_hold`` (== next cycle's ``hold_start``
  shifted one step height) only after the last cycle, so cycle 0 minus
  ``end_hold`` is the loop period regardless of ``--cycles``.
  """
  names = list(data["segment_names"])
  end_id = names.index("end_hold")
  labels = np.asarray(data["phase_segment"])
  cycles = np.asarray(data["cycle_index"])
  return int(np.sum((cycles == 0) & (labels != end_id)))


def clip_average_climb_rate(
  motion_file: str | Path, fallback: float = _FALLBACK_CLIMB_RATE
) -> float:
  """Average upward base speed of one reference cycle (m/s).

  Used at config-build time to point the climb velocity command at the rate
  the clip actually delivers, so the height-progress normalization and the
  commanded-velocity observation agree with the motion being imitated.
  """
  path = Path(motion_file)
  if not path.exists():
    return float(fallback)
  data = np.load(path, allow_pickle=True).item()
  cycle = _cycle_frames(data)
  return float(data["step_height"]) * float(data["fps"]) / cycle


@dataclass
class ClimbMotionClip:
  """One loopable regrip cycle on the policy-step time grid (torch tensors)."""

  joint_names: tuple[str, ...]
  ee_names: tuple[str, ...]
  fps: float
  step_height: float
  cycle_frames: int
  joint_pos: torch.Tensor  # (C, J)
  joint_vel: torch.Tensor  # (C, J)
  root_pos: torch.Tensor  # (C, 3) world, as generated (env origin added later)
  root_quat: torch.Tensor  # (C, 4) wxyz
  root_lin_vel: torch.Tensor  # (C, 3)
  root_ang_vel: torch.Tensor  # (C, 3)
  ee_pos_r: torch.Tensor  # (C, H, 3) EE body origins in the root frame
  ee_engaged: torch.Tensor  # (C, H) bool
  ee_edge_times: torch.Tensor  # (C, H, 2) countdown to next release/regrip
  ee_edge_distance: torch.Tensor  # (C, H) |time| to the NEAREST edge
  ee_run_time: torch.Tensor  # (C, H) length of the containing ON/OFF run
  toe_braced: torch.Tensor  # (C, 2) bool: foot-toe wall brace schedule
  ee_anchor_pos: torch.Tensor  # (C, H, 3) scheduled anchor, clip world frame

  @classmethod
  def from_file(
    cls, motion_file: str | Path, step_dt: float, device: str
  ) -> ClimbMotionClip:
    data = np.load(Path(motion_file), allow_pickle=True).item()
    return cls.from_dict(data, step_dt=step_dt, device=device)

  @classmethod
  def from_dict(
    cls, data: dict, step_dt: float, device: str
  ) -> ClimbMotionClip:
    required = (
      "joints_list",
      "joint_positions",
      "root_position",
      "root_quaternion",
      "fps",
      "ee_list",
      "ee_positions",
      "ee_engaged",
      "phase_segment",
      "segment_names",
      "cycle_index",
      "step_height",
    )
    missing = [key for key in required if key not in data]
    if missing:
      raise KeyError(f"Climb motion clip is missing keys: {missing}")

    src_fps = float(data["fps"])
    target_fps = 1.0 / step_dt
    joint_pos = np.asarray(data["joint_positions"], dtype=np.float64)
    root_pos = np.asarray(data["root_position"], dtype=np.float64)
    # The AMP/motion schema stores xyzw; mjlab is wxyz throughout.
    root_quat = np.asarray(data["root_quaternion"], dtype=np.float64)[
      :, [3, 0, 1, 2]
    ]
    ee_pos = np.asarray(data["ee_positions"], dtype=np.float64)
    engaged = np.asarray(data["ee_engaged"]) != 0
    # Foot-toe wall-brace schedule (gen_climb_clip 2026-07-29). Older clips
    # predate the bracing toes; they never schedule a brace.
    toe_braced = (
      np.asarray(data["toe_braced"]) != 0
      if "toe_braced" in data
      else np.zeros((engaged.shape[0], 2), dtype=bool)
    )
    # Scheduled anchor positions (clip world frame): the weld target while
    # engaged, the upcoming landing target while swinging. Consumed by the
    # landing-accuracy reward.
    ee_anchor = np.asarray(data["ee_anchor_positions"], dtype=np.float64)
    labels = np.asarray(data["phase_segment"])
    cycles = np.asarray(data["cycle_index"])

    if abs(src_fps - target_fps) > 1.0e-6:
      # Resample onto the policy-step grid. Adjacent frames are close, so
      # lerp (+ renormalize for quats) is adequate; flags use nearest.
      duration = (joint_pos.shape[0] - 1) / src_fps
      n_new = int(np.floor(duration * target_fps)) + 1
      pos = np.arange(n_new) / target_fps * src_fps
      joint_pos = _lerp_frames(joint_pos, pos)
      root_pos = _lerp_frames(root_pos, pos)
      root_quat = _lerp_frames(root_quat, pos)
      root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)
      ee_pos = _lerp_frames(ee_pos, pos)
      engaged = _nearest_frames(engaged, pos)
      toe_braced = _nearest_frames(toe_braced, pos)
      ee_anchor = _lerp_frames(ee_anchor, pos)
      labels = _nearest_frames(labels, pos)
      cycles = _nearest_frames(cycles, pos)

    cycle = _cycle_frames(
      {
        "segment_names": data["segment_names"],
        "phase_segment": labels,
        "cycle_index": cycles,
      }
    )
    total = joint_pos.shape[0]
    if cycle < 2:
      raise ValueError(f"Climb motion cycle has {cycle} frames; need >= 2.")
    if cycle >= total:
      raise ValueError(
        f"Climb motion clip has no frame beyond the cycle ({cycle} of "
        f"{total}); finite-difference velocities need the end_hold tail."
      )

    # Forward differences over [0, C) use frame C — present by construction
    # (end_hold, or the next cycle), and equal to frame 0 shifted one step
    # height, so the wrap velocity is the true loop velocity.
    joint_vel = (joint_pos[1 : cycle + 1] - joint_pos[:cycle]) * target_fps
    root_lin_vel = (root_pos[1 : cycle + 1] - root_pos[:cycle]) * target_fps
    q_conj = root_quat[:cycle] * np.array([1.0, -1.0, -1.0, -1.0])
    q_rel = _np_quat_mul(root_quat[1 : cycle + 1], q_conj)
    q_rel *= np.where(q_rel[:, 0:1] >= 0.0, 1.0, -1.0)
    vec_norm = np.linalg.norm(q_rel[:, 1:4], axis=-1)
    angle = 2.0 * np.arctan2(vec_norm, q_rel[:, 0])
    axis = q_rel[:, 1:4] / np.maximum(vec_norm, 1.0e-12)[:, None]
    root_ang_vel = axis * (angle * target_fps)[:, None]

    ee_pos_r = _np_quat_rotate_inverse(
      root_quat[:cycle], ee_pos[:cycle] - root_pos[:cycle, None, :]
    )

    # Countdown to the next release/regrip edge (normalized: the obs chart)
    # plus the symmetric distance to the NEAREST edge, past or future (the
    # graded-mismatch bill). The backward pass is the same countdown on the
    # time-reversed schedule, minus one because "0 frames ago" reads as 1 in
    # countdown form. EEs the clip never flips read 1.0 everywhere: no edge
    # is ever near, so their mismatch always bills full price.
    engaged_cycle = engaged[:cycle]
    future = _edge_frame_countdowns(engaged_cycle)
    past = _edge_frame_countdowns(engaged_cycle[::-1])[::-1]
    edge_distance = (
      np.minimum(future.min(axis=-1), past.min(axis=-1) - 1) / cycle
    ).astype(np.float32)
    edge_distance[:, future.min(axis=(0, -1)) >= cycle] = 1.0
    run_time = (_run_lengths(engaged_cycle) / cycle).astype(np.float32)

    def _t(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
      return torch.tensor(np.ascontiguousarray(arr), dtype=dtype, device=device)

    return cls(
      joint_names=tuple(data["joints_list"]),
      ee_names=tuple(data["ee_list"]),
      fps=target_fps,
      step_height=float(data["step_height"]),
      cycle_frames=cycle,
      joint_pos=_t(joint_pos[:cycle]),
      joint_vel=_t(joint_vel),
      root_pos=_t(root_pos[:cycle]),
      root_quat=_t(root_quat[:cycle]),
      root_lin_vel=_t(root_lin_vel),
      root_ang_vel=_t(root_ang_vel),
      ee_pos_r=_t(ee_pos_r),
      ee_engaged=_t(engaged_cycle, dtype=torch.bool),
      ee_edge_times=_t(future.astype(np.float32) / cycle),
      ee_edge_distance=_t(edge_distance),
      ee_run_time=_t(run_time),
      toe_braced=_t(toe_braced[:cycle], dtype=torch.bool),
      ee_anchor_pos=_t(ee_anchor[:cycle]),
    )



class ClimbMotionCommand(CommandTerm):
  """Phase-indexed climb-cycle reference with reference-state initialization.

  Reset places the robot AT a sampled clip frame and seeds the microspine
  latches from that frame, so the reward landscape at t=0 is exactly "keep
  doing the reference". Requires the microspine affordance command to be
  registered BEFORE this term (its own reset would otherwise clobber the
  seeded latch state); the ordering is asserted on first reset.
  """

  cfg: ClimbMotionCommandCfg

  def __init__(self, cfg: ClimbMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.clip = ClimbMotionClip.from_file(
      cfg.motion_file, step_dt=env.step_dt, device=str(self.device)
    )
    if tuple(cfg.ee_body_names) != self.clip.ee_names:
      raise ValueError(
        f"Clip EE order {self.clip.ee_names} does not match the configured "
        f"microspine order {tuple(cfg.ee_body_names)}."
      )

    joint_ids, joint_names = self.robot.find_joints(
      list(self.clip.joint_names), preserve_order=True
    )
    if tuple(joint_names) != self.clip.joint_names:
      raise ValueError(
        "Clip joints could not be mapped onto the robot in order: "
        f"{joint_names} vs {self.clip.joint_names}."
      )
    self._joint_ids = torch.tensor(
      joint_ids, dtype=torch.long, device=self.device
    )
    ee_body_ids = [
      self.robot.find_bodies([name])[0][0] for name in cfg.ee_body_names
    ]
    self._ee_body_ids = torch.tensor(
      ee_body_ids, dtype=torch.long, device=self.device
    )

    self.time_steps = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    # Offset from clip-frame root position to this env's reference position
    # in the CANONICAL clip frame (vertical wall, no env origin): reset
    # height jitter plus one step height per completed loop. World placement
    # applies the per-env wall frame + env origin via ``_to_world_point``.
    # Used for RSI placement bookkeeping and the ghost only — no reward
    # reads a global reference position.
    self.root_offset = torch.zeros(self.num_envs, 3, device=self.device)
    # Per-env wall frame (wxyz; identity = vertical wall), cached at reset
    # from the wall entity's default root state — the wall_angle curriculum
    # writes that earlier in the same reset pass, and the wall never moves
    # mid-episode, so reset-time caching is exact. All world-facing outputs
    # (RSI write, ref_* properties, ghost) rotate through this frame about
    # the wall base line; at identity the transform is a bit-exact no-op.
    self.wall_quat = torch.zeros(self.num_envs, 4, device=self.device)
    self.wall_quat[:, 0] = 1.0
    self._pivot_local = torch.tensor(
      [WALL_PIVOT_X, 0.0, 0.0], device=self.device
    )
    self.spawn_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self._order_checked = False
    self._ghost_model = None

    self.metrics["error_joint_pos"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_ee_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["engage_match"] = torch.zeros(self.num_envs, device=self.device)

  # -- reference (clip) side --------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """Phase clock + engage schedule chart, shape (E, 2 + 3H + 2).

    Layout: ``[cos, sin, engaged(H), (t_release, t_regrip) x H,
    toe_braced(2)]``. The countdown pairs are the schedule's upcoming edges
    per EE (normalized by the cycle, sliding linearly with the clock), so
    the policy reads each raster row's whole future — when this pad must
    let go and when it must re-latch — instead of only the bar under the
    playhead. The trailing pair flags the foot-toe wall-brace windows (the
    released foot's support during its own regrip).
    """
    phase = (
      self.time_steps.float() / self.clip.cycle_frames * (2.0 * torch.pi)
    )
    edges = self.clip.ee_edge_times[self.time_steps]
    return torch.cat(
      [
        torch.cos(phase).unsqueeze(-1),
        torch.sin(phase).unsqueeze(-1),
        self.ref_engaged.float(),
        edges.flatten(1, 2),
        self.ref_toe_braced.float(),
      ],
      dim=-1,
    )

  @property
  def ref_joint_pos(self) -> torch.Tensor:
    return self.clip.joint_pos[self.time_steps]

  @property
  def ref_joint_vel(self) -> torch.Tensor:
    return self.clip.joint_vel[self.time_steps]

  @property
  def ref_root_quat_w(self) -> torch.Tensor:
    return self._to_world_quat(self.clip.root_quat[self.time_steps])

  @property
  def ref_ee_pos_r(self) -> torch.Tensor:
    return self.clip.ee_pos_r[self.time_steps]

  @property
  def ref_engaged(self) -> torch.Tensor:
    return self.clip.ee_engaged[self.time_steps]

  @property
  def ref_toe_braced(self) -> torch.Tensor:
    """Scheduled foot-toe wall-brace flags, (E, 2) in PHYBOT_C2_FOOT_TOES
    order (left foot, right foot)."""
    return self.clip.toe_braced[self.time_steps]

  @property
  def ref_anchor_pos_w(self) -> torch.Tensor:
    """Scheduled anchor positions in sim world coordinates, (E, H, 3).

    Clip-frame anchors shifted by the accumulated per-loop step height and
    mapped through the per-env wall frame (+ env origin), i.e. exactly the
    ghost's anchor ladder. While an EE swings this is its upcoming landing
    target."""
    return self._to_world_point(
      self.clip.ee_anchor_pos[self.time_steps] + self.root_offset.unsqueeze(1)
    )

  @property
  def ref_edge_distance(self) -> torch.Tensor:
    """Normalized time to the NEAREST schedule edge per EE, (E, H).

    Consumed by the edge-graced engage mismatch bill, which thereby stays a
    piecewise-linear function of the countdown channels the policy observes.
    """
    return self.clip.ee_edge_distance[self.time_steps]

  @property
  def ref_run_time(self) -> torch.Tensor:
    """Normalized length of each EE's CURRENT scheduled ON/OFF run, (E, H).

    Multiplied by ``cycle_time_s`` this is the containing window's duration
    in seconds — the denominator of the window-proportional mismatch grace.
    """
    return self.clip.ee_run_time[self.time_steps]

  @property
  def ref_edge_countdowns(self) -> torch.Tensor:
    """Normalized countdowns to the next release/regrip edge, (E, H, 2).

    The same table the ``motion_command`` observation exposes; reward terms
    gated on it (seat-phase press shaping) therefore pay along signals the
    policy can see.
    """
    return self.clip.ee_edge_times[self.time_steps]

  @property
  def cycle_time_s(self) -> float:
    """Duration of one reference cycle in seconds."""
    return self.clip.cycle_frames / self.clip.fps

  # -- per-env wall frame -----------------------------------------------------
  # The clip is canonical (vertical wall, gravity down). Under the wall-angle
  # representation each env's wall is tilted about the base line
  # (x = WALL_PIVOT_X, z = 0); every world-facing quantity applies that
  # rotation. Envs at exactly 90° take a bit-exact identity fast path so the
  # terminal rung reproduces the retired gravity representation verbatim.

  @property
  def wall_up_w(self) -> torch.Tensor:
    """Along-wall climb direction per env, (E, 3) — ẑ on a vertical wall."""
    z_axis = torch.zeros(self.num_envs, 3, device=self.device)
    z_axis[:, 2] = 1.0
    return quat_apply(self.wall_quat, z_axis)

  @property
  def wall_pivot_w(self) -> torch.Tensor:
    """World position of the wall base (hinge) line per env, (E, 3)."""
    return self._env.scene.env_origins + self._pivot_local

  def _vertical_mask(self, env_ids: torch.Tensor | None) -> torch.Tensor:
    quat = self.wall_quat if env_ids is None else self.wall_quat[env_ids]
    return quat[:, 0] >= 1.0 - 1.0e-9

  def _to_world_point(
    self, points: torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Canonical clip-frame points -> world: rotate about the base line, add
    the env origin. Accepts (N, 3) or (N, H, 3)."""
    quat = self.wall_quat if env_ids is None else self.wall_quat[env_ids]
    origins = self._env.scene.env_origins
    if env_ids is not None:
      origins = origins[env_ids]
    vertical = self._vertical_mask(env_ids)
    pivot = self._pivot_local
    if points.dim() == 3:
      quat_b = quat.unsqueeze(1).expand(-1, points.shape[1], -1)
      rotated = pivot + quat_apply(quat_b, points - pivot)
      out = torch.where(vertical.view(-1, 1, 1), points, rotated)
      return out + origins.unsqueeze(1)
    rotated = pivot + quat_apply(quat, points - pivot)
    out = torch.where(vertical.view(-1, 1), points, rotated)
    return out + origins

  def _to_world_quat(
    self, quats: torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> torch.Tensor:
    wall = self.wall_quat if env_ids is None else self.wall_quat[env_ids]
    vertical = self._vertical_mask(env_ids)
    return torch.where(vertical.view(-1, 1), quats, quat_mul(wall, quats))

  def _to_world_vec(
    self, vecs: torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> torch.Tensor:
    wall = self.wall_quat if env_ids is None else self.wall_quat[env_ids]
    # quat_apply with an identity quaternion is exact — no fast path needed.
    return quat_apply(wall, vecs)

  @property
  def ref_root_pos_w(self) -> torch.Tensor:
    """Ghost/RSI-only global reference position (untracked by rewards)."""
    return self._to_world_point(
      self.clip.root_pos[self.time_steps] + self.root_offset
    )

  # -- robot side -------------------------------------------------------------

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos[:, self._joint_ids]

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel[:, self._joint_ids]

  @property
  def robot_root_quat_w(self) -> torch.Tensor:
    return self.robot.data.root_link_quat_w

  @property
  def robot_ee_pos_r(self) -> torch.Tensor:
    """EE body origins in the robot's root frame, (E, H, 3)."""
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    ee_pos = self.robot.data.body_link_pos_w[:, self._ee_body_ids]
    delta = ee_pos - root_pos.unsqueeze(1)
    quat = root_quat.unsqueeze(1).expand(-1, len(self.cfg.ee_body_names), -1)
    return quat_apply_inverse(quat, delta)

  @property
  def robot_engaged(self) -> torch.Tensor:
    return self._affordance().engaged

  @property
  def height_gained(self) -> torch.Tensor:
    """Along-wall base rise since the episode's RSI placement, (E,).

    Curriculum input. Projection onto ``wall_up_w`` — world z on a vertical
    wall (bit-identical to the retired z-difference there)."""
    delta = self.robot.data.root_link_pos_w - self.spawn_pos
    return (delta * self.wall_up_w).sum(dim=-1)

  def _affordance(self):
    return self._env.command_manager.get_term(self.cfg.affordance_command_name)

  # -- lifecycle --------------------------------------------------------------

  def _check_term_order(self) -> None:
    """The affordance must reset before this term seeds it (dict order)."""
    if self._order_checked:
      return
    names = self._env.command_manager.active_terms
    own = next(
      name
      for name in names
      if self._env.command_manager.get_term(name) is self
    )
    if names.index(self.cfg.affordance_command_name) > names.index(own):
      raise RuntimeError(
        f"Command '{self.cfg.affordance_command_name}' must be registered "
        f"before '{own}': its reset would clobber the RSI-seeded latches."
      )
    self._order_checked = True

  def _update_metrics(self) -> None:
    self.metrics["error_joint_pos"] = (
      self.ref_joint_pos - self.robot_joint_pos
    ).abs().mean(dim=-1)
    self.metrics["error_ee_pos"] = (
      (self.ref_ee_pos_r - self.robot_ee_pos_r).norm(dim=-1).mean(dim=-1)
    )
    self.metrics["engage_match"] = (
      (self.ref_engaged == self.robot_engaged).float().mean(dim=-1)
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._check_term_order()
    count = len(env_ids)
    if self.cfg.sampling_mode == "start":
      frames = torch.zeros(count, dtype=torch.long, device=self.device)
    else:
      frames = torch.randint(
        0, self.clip.cycle_frames, (count,), device=self.device
      )
    self.time_steps[env_ids] = frames

    # --- reference state initialization -------------------------------------
    # Per-env wall frame first: the wall_angle curriculum wrote this env's
    # pose into the wall's default_root_state earlier in this reset pass
    # (identity when no wall_angle term is installed — vertical wall).
    wall: Entity = self._env.scene[self.cfg.wall_entity_name]
    self.wall_quat[env_ids] = wall.data.default_root_state[
      env_ids, 3:7
    ].clone()
    # Canonical-frame offset: spawn-height jitter (along the wall) plus the
    # vertical-height compensation; env origins are applied by the world
    # transform.
    offset = torch.zeros(count, 3, device=self.device)
    offset[:, 2] = sample_uniform(
      self.cfg.height_offset_range[0],
      self.cfg.height_offset_range[1],
      (count,),
      device=self.device,
    )
    if self.cfg.spawn_match_vertical_height:
      offset[:, 2] += vertical_height_spawn_offset(
        self.clip.root_pos[frames],
        self.wall_quat[env_ids],
        float(self._pivot_local[0].item()),
      )
    self.root_offset[env_ids] = offset
    root_pos = self._to_world_point(
      self.clip.root_pos[frames] + offset, env_ids
    )
    self.spawn_pos[env_ids] = root_pos

    joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, self._joint_ids] = self.clip.joint_pos[frames]
    if self.cfg.joint_pos_noise > 0.0:
      joint_pos[:, self._joint_ids] += sample_uniform(
        -self.cfg.joint_pos_noise,
        self.cfg.joint_pos_noise,
        (count, len(self._joint_ids)),
        device=self.device,
      )
      limits = self.robot.data.soft_joint_pos_limits[env_ids]
      joint_pos = torch.clip(joint_pos, limits[..., 0], limits[..., 1])
    joint_vel = torch.zeros_like(joint_pos)
    joint_vel[:, self._joint_ids] = self.clip.joint_vel[frames]

    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    root_state = torch.cat(
      [
        root_pos,
        self._to_world_quat(self.clip.root_quat[frames], env_ids),
        self._to_world_vec(self.clip.root_lin_vel[frames], env_ids),
        self._to_world_vec(self.clip.root_ang_vel[frames], env_ids),
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.clear_state(env_ids=env_ids)

    # Seed the microspine latches from the frame's engage schedule. The pads
    # of engaged EEs sit on the wall in every clip frame, so the constraint
    # executor anchors them at physically consistent poses on the next step.
    self._affordance().seed_engaged(env_ids, self.clip.ee_engaged[frames])

  def _update_command(self) -> None:
    # R4 reseat hold (2026-08-03): while a dr-forced release awaits
    # re-seating, the phase clock waits with it — all 16 command dims freeze
    # together, so recovery tracks a self-consistent standing target instead
    # of racing the schedule. Economics: frozen ref height = paused climb
    # income is the only time cost; the pending EE's mismatch is exempted at
    # the bill, and the press-shaping window is already open for it (frozen
    # ref says engaged, robot says not). Deploy mirror: the C++ phase player
    # needs the same pause keyed on a loss signal (gimbal sensing); a
    # no-pause fallback is exactly pre-R4 behavior.
    hold = self._affordance().reseat_hold
    next_steps = torch.where(hold, self.time_steps, self.time_steps + 1)
    wrapped = next_steps >= self.clip.cycle_frames
    self.time_steps = torch.where(
      wrapped, torch.zeros_like(next_steps), next_steps
    )
    # The loop's state-5-equals-state-1 closure is one step height up; keep
    # the ghost/RSI reference continuous across the wrap.
    self.root_offset[wrapped, 2] += self.clip.step_height

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    if self._ghost_model is None:
      self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
      self._ghost_model.geom_rgba[:] = np.array(_GHOST_COLOR, dtype=np.float32)

    indexing = self.robot.indexing
    free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
    joint_q_adr = indexing.joint_q_adr.cpu().numpy()
    for batch in env_indices:
      qpos = np.zeros(self._env.sim.mj_model.nq)
      qpos[free_joint_q_adr[0:3]] = self.ref_root_pos_w[batch].cpu().numpy()
      qpos[free_joint_q_adr[3:7]] = self.ref_root_quat_w[batch].cpu().numpy()
      full_joint_pos = self.robot.data.default_joint_pos[batch].clone()
      full_joint_pos[self._joint_ids] = self.ref_joint_pos[batch]
      qpos[joint_q_adr] = full_joint_pos.cpu().numpy()
      visualizer.add_ghost_mesh(
        qpos, model=self._ghost_model, label=f"ghost_{batch}"
      )


@dataclass(kw_only=True)
class ClimbMotionCommandCfg(CommandTermCfg):
  """Configuration for :class:`ClimbMotionCommand`."""

  entity_name: str = "robot"
  motion_file: str
  affordance_command_name: str = "microspine_affordance"
  # Wall entity whose per-env pose defines the wall frame (tilt about the
  # base line); read once per reset AFTER the wall_angle curriculum writes it.
  wall_entity_name: str = "wall"
  ee_body_names: tuple[str, ...]
  # "uniform" = DeepMimic RSI (sample a random cycle phase per episode);
  # "start" = always begin at frame 0 (evaluation/play).
  sampling_mode: Literal["uniform", "start"] = "uniform"
  # Uniform extra spawn height (m) on top of the clip's root z, decoupling
  # absolute altitude from cycle phase.
  height_offset_range: tuple[float, float] = (0.0, 0.0)
  # Slide the RSI reference along a TILTED wall so the spawn keeps the same
  # WORLD height as the vertical-wall spawn (floor clearance stops shrinking
  # with the angle). Exactly zero at 90° — the bit-identity path holds.
  # Disable to reproduce the pure-rotation world (equivalence audits).
  spawn_match_vertical_height: bool = True
  # Uniform RSI joint-angle noise (rad); clamped to soft limits.
  joint_pos_noise: float = 0.0
  # The reference loops; it is never resampled mid-episode.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> ClimbMotionCommand:
    return ClimbMotionCommand(self, env)
