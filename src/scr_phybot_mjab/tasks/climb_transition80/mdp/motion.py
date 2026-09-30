"""One-shot 80-degree mount prologue reference with BeyondMimic body tables.

``TransitionMotionCommand`` specialises the climb cycle command
(``ClimbMotionCommand``: RSI, latch seeding, reference properties, the R4
reseat freeze, the per-env wall frame) for a single acyclic clip:

* TERMINAL HOLD — the clock clamps on the last playable frame (the all-four
  ``handoff_hold`` stance) instead of wrapping;
* ACYCLIC ENGAGE TABLES — edge countdowns / run lengths / edge distances are
  rebuilt without wrap so the seam never manufactures phantom edges;
* BLIND-DELIVERY SPAWN SCATTER — RSI frames with no engaged pads get uniform
  base x/y/yaw noise while the reference and its anchors stay canonical;
  seeded frames keep the exact clip pose (welded pads cannot move);
* RSI BIAS WINDOW — ``spawn_early_fraction`` of births land inside the
  segments named by ``spawn_bias_segments`` (the exam scripts pin windows
  this way); off by default — births are ADAPTIVE instead;
* FK BODY TABLES — the clip carries qpos only, so every tracked body's
  reference pose/velocity is computed once at load with MuJoCo forward
  kinematics (canonical frame, rotated into the tilted world through the
  parent's wall-frame machinery); body tracking is GLOBAL (the wall and its
  anchors are world-fixed) with the mjlab xy+yaw re-anchoring available as
  ``relative_frame="anchor_yaw"``;
* ADAPTIVE SAMPLING — BeyondMimic failure-weighted start bins (one per
  clip-second): births concentrate where episodes die;
* MOTION_COMMAND 10 — ``[cos, sin, engaged(4), t_next_edge(4)]``: the
  parent's 16-lane chart carried six constants on the a80 prologue
  (toe_braced is generator-zero; only the left hand ever releases, so the
  other three t_release lanes sat on the 1.0 sentinel). Each pad now reads
  the countdown to ITS next edge (release while engaged, regrip while not).

The acyclic-table / window / scatter helpers are this package's own copies
of the climb_transition originals (2026-08-09 .. 08-15); the package does
not import climb_transition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_angle_axis,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from scr_phybot_mjab.tasks.climb_wall.mdp.motion import (
  ClimbMotionClip,
  ClimbMotionCommand,
  ClimbMotionCommandCfg,
  _lerp_frames,
  _np_quat_mul,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = (
  "birth_mask_from_ranges",
  "rake_windows_from_engaged",
  "ARM_TRACKING_BODY_NAMES",
  "DEFAULT_TRACKING_BODY_NAMES",
  "TransitionMotionCommand",
  "TransitionMotionCommandCfg",
  "adaptive_bin_probabilities",
  "apply_spawn_pose_noise",
  "build_fk_tables",
  "finite_difference_velocities",
  "rebuild_acyclic_schedule",
  "resolve_segment_window",
)

# The 17 tracked bodies (phybot_c2 link names, no entity prefix; the
# floating base is ``torso``). Pads are included so the tracking kernels and
# the pad-only term share one table; the pad CENTER (contact law, Phi) is a
# different point — see ``executor.pad_center_world()``.
DEFAULT_TRACKING_BODY_NAMES: tuple[str, ...] = (
  "torso",
  "torso_upper",
  "neck_pitch",
  "left_shoulder_roll",
  "left_elbow_pitch",
  "left_hand_spine",
  "right_shoulder_roll",
  "right_elbow_pitch",
  "right_hand_spine",
  "left_hip_yaw",
  "left_knee",
  "left_ankle_roll",
  "left_foot_spine",
  "right_hip_yaw",
  "right_knee",
  "right_ankle_roll",
  "right_foot_spine",
)
# Arm links for the optional LadderMan Eq.(1) asymmetric split.
ARM_TRACKING_BODY_NAMES: tuple[str, ...] = (
  "left_shoulder_roll",
  "left_elbow_pitch",
  "right_shoulder_roll",
  "right_elbow_pitch",
)


# --- acyclic engage tables ------------------------------------------------------


def _acyclic_edge_frame_countdowns(engaged: np.ndarray) -> np.ndarray:
  """Frames to each EE's next falling/rising edge without wrap; sentinel =
  full clip length meaning "never"."""
  cycle, num_ee = engaged.shape
  out = np.full((cycle, num_ee, 2), cycle, dtype=np.int64)
  frames = np.arange(cycle)
  for h in range(num_ee):
    now = engaged[:, h]
    nxt = np.append(now[1:], now[-1])
    for k, edge in enumerate((now & ~nxt, ~now & nxt)):  # falling, rising
      event_frames = np.flatnonzero(edge) + 1
      if event_frames.size == 0:
        continue
      dist = event_frames[None, :] - frames[:, None]
      dist = np.where(dist <= 0, cycle, dist)
      out[:, h, k] = dist.min(axis=1)
  return out


def _acyclic_run_lengths(engaged: np.ndarray) -> np.ndarray:
  """Run lengths without the wrap join (frame 0 starts a run, the last run
  ends at the clip end)."""
  cycle, num_ee = engaged.shape
  out = np.empty((cycle, num_ee), dtype=np.int64)
  for h in range(num_ee):
    col = engaged[:, h]
    starts = np.flatnonzero(np.append(True, col[1:] != col[:-1]))
    bounds = np.append(starts, cycle)
    for s, e in zip(bounds[:-1], bounds[1:]):
      out[s:e, h] = e - s
  return out


def rebuild_acyclic_schedule(clip: ClimbMotionClip) -> None:
  """Replace a loaded clip's circular engage tables with acyclic ones
  (composes with ``ClimbMotionClip.from_dict`` unchanged)."""
  engaged = clip.ee_engaged.cpu().numpy().astype(bool)
  cycle = engaged.shape[0]
  device = clip.ee_edge_times.device
  future = _acyclic_edge_frame_countdowns(engaged)
  past = _acyclic_edge_frame_countdowns(engaged[::-1])[::-1]
  edge_frames = np.minimum(future.min(axis=-1), past.min(axis=-1) - 1)
  edge_distance = (edge_frames / cycle).astype(np.float32)
  # A frame whose nearest "edge" on both sides is the no-event sentinel must
  # read exactly 1.0 (covers EEs that never flip at all).
  edge_distance[edge_frames >= cycle - 1] = 1.0
  run_time = (_acyclic_run_lengths(engaged) / cycle).astype(np.float32)

  def _t(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(arr), device=device)

  clip.ee_edge_times = _t(future.astype(np.float32) / cycle)
  clip.ee_edge_distance = _t(edge_distance)
  clip.ee_run_time = _t(run_time)


def resolve_segment_window(motion_file: str, segments: tuple[str, ...]) -> tuple[float, float] | None:
  """Frame-fraction window [lo, hi) of the named clip segments (None when
  the file is absent or no frame matches)."""
  path = Path(motion_file)
  if not path.exists():
    return None
  data = np.load(path, allow_pickle=True).item()
  names = list(data["segment_names"])
  try:
    wanted = {names.index(seg) for seg in segments}
  except ValueError:
    return None
  labels = np.asarray(data["phase_segment"])
  cycles = np.asarray(data["cycle_index"])
  end_id = names.index("end_hold")
  cycle_mask = (cycles == cycles.min()) & (labels != end_id)
  cycle = int(cycle_mask.sum())
  hit = np.flatnonzero(cycle_mask & np.isin(labels, list(wanted)))
  if hit.size == 0 or cycle == 0:
    return None
  return float(hit.min()) / cycle, float(hit.max() + 1) / cycle


def apply_spawn_pose_noise(
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  disengaged: torch.Tensor,
  xy_noise: tuple[float, float],
  yaw_noise: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Uniform base x/y/yaw noise on rows where ``disengaged`` is True."""
  if xy_noise[0] == 0.0 and xy_noise[1] == 0.0 and yaw_noise == 0.0:
    return root_pos, root_quat
  count = root_pos.shape[0]
  device = root_pos.device
  gate = disengaged.to(root_pos.dtype)
  pos = root_pos.clone()
  pos[:, 0] += gate * sample_uniform(-xy_noise[0], xy_noise[0], (count,), device=device)
  pos[:, 1] += gate * sample_uniform(-xy_noise[1], xy_noise[1], (count,), device=device)
  yaw = gate * sample_uniform(-yaw_noise, yaw_noise, (count,), device=device)
  axis = torch.zeros(count, 3, device=device, dtype=root_pos.dtype)
  axis[:, 2] = 1.0
  quat = quat_mul(quat_from_angle_axis(yaw, axis), root_quat)
  return pos, quat


# --- FK body tables -------------------------------------------------------------


def build_fk_tables(
  mj_model: mujoco.MjModel,
  base_qpos: np.ndarray,
  root_qadr: np.ndarray,
  joint_qadr: np.ndarray,
  body_ids: np.ndarray,
  root_pos: np.ndarray,
  root_quat_wxyz: np.ndarray,
  joint_pos: np.ndarray,
  *,
  passive_qadr: np.ndarray | None = None,
  passive_pos: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Forward-kinematics tables for a clip: body positions (C, B, 3) and
  orientations (C, B, 4, wxyz) in the clip's own frame. ``base_qpos``
  supplies coordinates the clip does not carry. Optional passive coordinates
  reproduce the CAD gimbal state without adding policy action targets.
  ``root_qadr`` are the 7 free-joint addresses, ``joint_qadr`` the clip's
  policy-joint addresses in clip order."""
  data = mujoco.MjData(mj_model)
  root_qadr = np.asarray(root_qadr, dtype=np.int64)
  joint_qadr = np.asarray(joint_qadr, dtype=np.int64)
  body_ids = np.asarray(body_ids, dtype=np.int64)
  frames = root_pos.shape[0]
  if (passive_qadr is None) != (passive_pos is None):
    raise ValueError("Passive FK needs both coordinate addresses and positions")
  if passive_pos is not None and passive_pos.shape != (frames, len(passive_qadr)):
    raise ValueError("Passive FK positions must match the frame and joint counts")
  pos = np.zeros((frames, body_ids.shape[0], 3), dtype=np.float64)
  quat = np.zeros((frames, body_ids.shape[0], 4), dtype=np.float64)
  for t in range(frames):
    q = np.array(base_qpos, dtype=np.float64, copy=True)
    q[root_qadr[0:3]] = root_pos[t]
    q[root_qadr[3:7]] = root_quat_wxyz[t]
    q[joint_qadr] = joint_pos[t]
    if passive_qadr is not None:
      q[passive_qadr] = passive_pos[t]
    data.qpos[:] = q
    mujoco.mj_kinematics(mj_model, data)
    pos[t] = data.xpos[body_ids]
    quat[t] = data.xquat[body_ids]
  return pos, quat


def finite_difference_velocities(pos: np.ndarray, quat: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
  """Forward-difference linear (C, B, 3) and angular (C, B, 3) velocities;
  the last row is zero (the clip ends in a static hold the clock clamps on)."""
  lin = np.zeros_like(pos)
  lin[:-1] = (pos[1:] - pos[:-1]) * fps
  ang = np.zeros_like(pos)
  q_conj = quat[:-1] * np.array([1.0, -1.0, -1.0, -1.0])
  q_rel = _np_quat_mul(quat[1:], q_conj)
  q_rel = q_rel * np.where(q_rel[..., 0:1] >= 0.0, 1.0, -1.0)
  vec_norm = np.linalg.norm(q_rel[..., 1:4], axis=-1)
  angle = 2.0 * np.arctan2(vec_norm, q_rel[..., 0])
  axis = q_rel[..., 1:4] / np.maximum(vec_norm, 1.0e-12)[..., None]
  ang[:-1] = axis * (angle * fps)[..., None]
  return lin, ang


def adaptive_bin_probabilities(
  bin_failed: torch.Tensor,
  uniform_ratio: float,
  max_bin_prob: float | None = None,
  kernel: torch.Tensor | None = None,
) -> torch.Tensor:
  """BeyondMimic start-bin distribution: failure histogram + uniform floor,
  optional non-causal smoothing kernel, optional per-bin probability cap."""
  n = bin_failed.shape[0]
  probs = bin_failed + uniform_ratio / float(n)
  if kernel is not None and kernel.numel() > 1:
    padded = torch.nn.functional.pad(probs.view(1, 1, -1), (0, kernel.numel() - 1), mode="replicate")
    probs = torch.nn.functional.conv1d(padded, kernel.view(1, 1, -1)).view(-1)
  probs = probs / probs.sum()
  if max_bin_prob is not None and max_bin_prob * n > 1.0 and max_bin_prob < 1.0:
    for _ in range(n):
      over = probs > max_bin_prob
      if not bool(over.any()):
        break
      excess = (probs[over] - max_bin_prob).sum()
      probs = torch.where(over, torch.full_like(probs, max_bin_prob), probs)
      under = ~over
      share = probs[under]
      if bool(under.any()) and float(share.sum()) > 0.0:
        probs[under] = share + excess * share / share.sum()
  return probs


# --- the command ------------------------------------------------------------------


def birth_mask_from_ranges(cycle_frames: int, ranges: tuple[tuple[int, int], ...]) -> torch.Tensor:
  """(C,) bool: frames RSI may birth into. ``ranges`` are (start, end)
  with end exclusive; raises if nothing stays allowed."""
  allowed = torch.ones(int(cycle_frames), dtype=torch.bool)
  for lo, hi in ranges:
    lo_i, hi_i = max(0, int(lo)), min(int(cycle_frames), int(hi))
    if hi_i > lo_i:
      allowed[lo_i:hi_i] = False
  if not bool(allowed.any()):
    raise ValueError("birth_exclude_frames leaves no frame to birth into.")
  return allowed


def rake_windows_from_engaged(engaged: "np.ndarray | torch.Tensor", rake_frames: int) -> tuple[tuple[int, int], ...]:
  """Classroom windows [edge - rake_frames, edge) for every engage rising
  edge of every EE (sorted, may overlap across EEs)."""
  arr = engaged.detach().cpu().numpy() if torch.is_tensor(engaged) else np.asarray(engaged)
  arr = arr.astype(bool)
  out: list[tuple[int, int]] = []
  for h in range(arr.shape[1]):
    rises = np.flatnonzero(np.diff(arr[:, h].astype(np.int8)) > 0) + 1
    for r in rises:
      out.append((int(max(0, r - int(rake_frames))), int(r)))
  return tuple(sorted(out))


class TransitionMotionCommand(ClimbMotionCommand):
  """One-shot prologue playback + BeyondMimic body tables + adaptive RSI."""

  cfg: TransitionMotionCommandCfg

  def __init__(self, cfg: TransitionMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    rebuild_acyclic_schedule(self.clip)
    # First frame with any scheduled engagement: the boundary of the
    # standing stretch (stand -> raise -> place).
    self._first_engage_frame = int(self.clip.ee_engaged.any(dim=-1).int().argmax().item())
    self._bias_window = self._bias_window_for(cfg, self.clip, self._first_engage_frame, cfg.motion_file)
    self._bias_segments_active: tuple[str, ...] | None = None

    names = tuple(cfg.body_names) or DEFAULT_TRACKING_BODY_NAMES
    missing = [n for n in cfg.ee_body_names if n not in names]
    if missing:
      raise ValueError(f"Tracking body set must contain every pad body; missing {missing}.")
    self._tracking_body_names = names
    ids = [self.robot.find_bodies([name])[0][0] for name in names]
    self._body_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
    self._pad_slots = torch.tensor(
      [names.index(n) for n in cfg.ee_body_names],
      dtype=torch.long,
      device=self.device,
    )
    self._fk_pos: torch.Tensor | None = None
    self._fk_quat: torch.Tensor | None = None
    self._fk_lin_vel: torch.Tensor | None = None
    self._fk_ang_vel: torch.Tensor | None = None
    # Pad FACE centers from the generator (clip key ``pad_positions``, the
    # contact-sphere centroid the contact law and Phi use), on the policy
    # step grid, canonical frame. The pad BODY origin sits 0.16-0.21 m
    # behind the face, so origin tracking alone let the face park 46 cm
    # from its anchor while the origin read 27 cm (v1f probe).
    self._pad_center_canon = self._load_pad_centers(cfg.motion_file, env.step_dt)

    # Adaptive sampling state: one bin per clip-second.
    self._birth_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._bin_count = int(self.clip.cycle_frames * env.step_dt) + 1
    self._bin_failed = torch.zeros(self._bin_count, device=self.device)
    self._current_bin_failed = torch.zeros(self._bin_count, device=self.device)
    kernel = torch.tensor(
      [cfg.adaptive_lambda**i for i in range(max(1, cfg.adaptive_kernel_size))],
      device=self.device,
    )
    self._adaptive_kernel = kernel / kernel.sum()
    self._birth_freeze_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._birth_allowed = birth_mask_from_ranges(self.clip.cycle_frames, cfg.birth_exclude_frames).to(self.device)
    # Per-bin tables of allowed frames (adaptive draws stay inside the bin).
    bins_of = torch.arange(self.clip.cycle_frames) * self._bin_count // max(self.clip.cycle_frames, 1)
    rows = []
    for b in range(self._bin_count):
      rows.append(torch.nonzero((bins_of == b) & self._birth_allowed.cpu()).squeeze(-1))
    width = max(1, max(int(r.numel()) for r in rows))
    table = torch.zeros(self._bin_count, width, dtype=torch.long)
    counts = torch.zeros(self._bin_count, dtype=torch.long)
    for b, r in enumerate(rows):
      counts[b] = r.numel()
      if r.numel():
        table[b, : r.numel()] = r
    self._bin_allowed_table = table.to(self.device)
    self._bin_allowed_count = counts.to(self.device)
    self.metrics["birth_frozen"] = torch.zeros(self.num_envs, device=self.device)
    # Ignition classroom windows (see the cfg field).
    if cfg.spawn_bias_frame_windows:
      windows = tuple((int(a), int(b)) for a, b in cfg.spawn_bias_frame_windows)
    elif cfg.spawn_bias_auto_rake_frames > 0:
      windows = rake_windows_from_engaged(self.clip.ee_engaged, cfg.spawn_bias_auto_rake_frames)
    else:
      windows = ()
    self._classroom_windows: tuple[tuple[int, int], ...] = windows
    cmask = torch.zeros(self.clip.cycle_frames, dtype=torch.bool)
    for a, b in windows:
      cmask[max(0, a) : min(self.clip.cycle_frames, b)] = True
    self._classroom_mask = cmask.to(self.device)
    self.metrics["birth_in_classroom"] = torch.zeros(self.num_envs, device=self.device)

    for key in (
      "sampling_entropy",
      "sampling_top1_bin",
      "error_anchor_pos",
      "error_body_pos",
      "error_pad_pos",
    ):
      self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

  def _load_pad_centers(self, motion_file: str, step_dt: float) -> torch.Tensor:
    data = np.load(Path(motion_file), allow_pickle=True).item()
    if "pad_positions" not in data:
      raise KeyError(f"{motion_file} carries no 'pad_positions' (gen_mount_clip writes them).")
    pads = np.asarray(data["pad_positions"], dtype=np.float64)
    src_fps = float(data["fps"])
    target_fps = 1.0 / step_dt
    if abs(src_fps - target_fps) > 1.0e-6:
      duration = (pads.shape[0] - 1) / src_fps
      n_new = int(np.floor(duration * target_fps)) + 1
      pads = _lerp_frames(pads, np.arange(n_new) / target_fps * src_fps)
    pads = pads[: self.clip.cycle_frames]
    if pads.shape[0] != self.clip.cycle_frames:
      raise ValueError("pad_positions shorter than the playable clip.")
    return torch.tensor(np.ascontiguousarray(pads), dtype=torch.float32, device=self.device)

  # -- RSI windows -------------------------------------------------------------

  @staticmethod
  def _bias_window_for(
    cfg: TransitionMotionCommandCfg,
    clip: ClimbMotionClip,
    first_engage_frame: int,
    motion_file: str,
  ) -> tuple[int, int]:
    """Biased-spawn frame window (local frames): the named segments, else
    everything before the first scheduled engagement."""
    window = None
    if cfg.spawn_bias_segments:
      window = resolve_segment_window(motion_file, cfg.spawn_bias_segments)
    if window is None:
      return (0, max(first_engage_frame, 1))
    cycle = clip.cycle_frames
    lo = int(window[0] * cycle)
    hi = max(int(window[1] * cycle), lo + 1)
    return (lo, min(hi, cycle))

  def set_bias_segments(self, segments: tuple[str, ...]) -> None:
    """Re-aim the biased window at the named segments (exam scripts)."""
    window = resolve_segment_window(self.cfg.motion_file, segments)
    cycle = self.clip.cycle_frames
    if window is None:
      self._bias_window = (0, cycle)
    else:
      lo = int(window[0] * cycle)
      self._bias_window = (lo, min(max(int(window[1] * cycle), lo + 1), cycle))
    self._bias_segments_active = tuple(segments)
    self._classroom_windows = ()  # an exam's segment window replaces the classroom

  def _draw_rsi_frames(self, count: int) -> torch.Tensor:
    """Uniform or adaptive draw, with the biased-window overlay."""
    if self.cfg.sampling_mode == "adaptive":
      frames = self._adaptive_draw(count)
    else:
      allowed_idx = torch.nonzero(self._birth_allowed).squeeze(-1)
      pick = torch.randint(0, allowed_idx.numel(), (count,), device=self.device)
      frames = allowed_idx[pick]
    if self.cfg.spawn_early_fraction > 0.0:
      pool = self._bias_pool()
      if pool.numel():
        early = torch.rand(count, device=self.device) < self.cfg.spawn_early_fraction
        pick = torch.randint(0, pool.numel(), (int(early.sum().item()),), device=self.device)
        frames[early] = pool[pick]
    return frames

  def _bias_pool(self) -> torch.Tensor:
    """Frames the biased share may birth into: the classroom windows (union)
    when configured, else the single segment window; always inside the
    birth mask."""
    if self._classroom_windows:
      pool_mask = self._classroom_mask & self._birth_allowed
    else:
      lo, hi = self._bias_window
      pool_mask = torch.zeros_like(self._birth_allowed)
      if hi > lo:
        pool_mask[lo:hi] = True
      pool_mask &= self._birth_allowed
    return torch.nonzero(pool_mask).squeeze(-1)

  # -- adaptive sampling ---------------------------------------------------------

  def _record_failures(self, env_ids: torch.Tensor) -> None:
    manager = getattr(self._env, "termination_manager", None)
    if manager is None:
      return
    failed = manager.terminated[env_ids]
    if not bool(failed.any()):
      return
    source = self._birth_frames if self.cfg.adaptive_attribution == "birth" else self.time_steps
    frames = source[env_ids][failed]
    bins = (frames * self._bin_count) // max(self.clip.cycle_frames, 1)
    bins = bins.clamp(0, self._bin_count - 1)
    self._current_bin_failed += torch.bincount(bins, minlength=self._bin_count).float()

  def _adaptive_draw(self, count: int) -> torch.Tensor:
    probs = adaptive_bin_probabilities(
      self._bin_failed,
      self.cfg.adaptive_uniform_ratio,
      self.cfg.adaptive_max_bin_prob,
      self._adaptive_kernel,
    )
    # Tenable-birth mask: bins with no allowed frame draw nothing; inside a
    # bin the frame is uniform over its allowed frames.
    probs = probs * (self._bin_allowed_count > 0).to(probs.dtype)
    probs = probs / probs.sum().clamp_min(1.0e-12)
    bins = torch.multinomial(probs, count, replacement=True)
    u = torch.rand(count, device=self.device)
    counts = self._bin_allowed_count[bins]
    idx = (u * counts.to(u.dtype)).long().clamp(max=(counts - 1).clamp_min(0))
    frames = self._bin_allowed_table[bins, idx]
    entropy = -(probs * (probs + 1.0e-12).log()).sum()
    norm = math.log(self._bin_count) if self._bin_count > 1 else 1.0
    self.metrics["sampling_entropy"][:] = entropy / norm
    self.metrics["sampling_top1_bin"][:] = probs.argmax().float() / self._bin_count
    return frames.clamp(0, self.clip.cycle_frames - 1)

  # -- lifecycle -------------------------------------------------------------------

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Mirror of ClimbMotionCommand._resample_command with the transition
    # deltas marked <DELTA>. Failures are recorded BEFORE the clock is
    # rewritten: mjlab resets the command manager before the termination
    # manager, so ``terminated`` still carries this step's verdicts.
    self._check_term_order()
    if self.cfg.sampling_mode == "adaptive":
      self._record_failures(env_ids)
    count = len(env_ids)

    from mjlab.entity import Entity  # local: typing only

    wall: Entity = self._env.scene[self.cfg.wall_entity_name]
    self.wall_quat[env_ids] = wall.data.default_root_state[env_ids, 3:7].clone()

    if self.cfg.sampling_mode == "start":
      frames = torch.zeros(count, dtype=torch.long, device=self.device)
    else:
      frames = self._draw_rsi_frames(count)  # <DELTA> biased / adaptive draw
    self.time_steps[env_ids] = frames
    self._birth_frames[env_ids] = frames
    self.metrics["birth_in_classroom"][env_ids] = self._classroom_mask[frames].to(
      self.metrics["birth_in_classroom"].dtype
    )
    seeded_birth = self.clip.ee_engaged[frames].any(dim=-1)
    self._birth_freeze_left[env_ids] = torch.where(
      seeded_birth,
      torch.full_like(frames, int(self.cfg.birth_freeze_steps)),
      torch.zeros_like(frames),
    )
    offset = torch.zeros(count, 3, device=self.device)
    offset[:, 2] = sample_uniform(
      self.cfg.height_offset_range[0],
      self.cfg.height_offset_range[1],
      (count,),
      device=self.device,
    )
    # spawn_match_vertical_height deliberately NOT applied: the native-angle
    # clip carries true world geometry; the cycle's compensation would lift
    # standing frames off the floor.
    self.root_offset[env_ids] = offset
    root_pos = self._to_world_point(self.clip.root_pos[frames] + offset, env_ids)
    self.spawn_pos[env_ids] = root_pos

    joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, self._joint_ids] = self.clip.joint_pos[frames]
    # <DELTA> joint noise on DISENGAGED rows only: a seeded birth welds the
    # pad at the anchor while noised joints put it mm-cm away.
    disengaged = ~self.clip.ee_engaged[frames].any(dim=-1)
    if self.cfg.joint_pos_noise > 0.0:
      joint_pos[:, self._joint_ids] += sample_uniform(
        -self.cfg.joint_pos_noise,
        self.cfg.joint_pos_noise,
        (count, len(self._joint_ids)),
        device=self.device,
      ) * disengaged.to(joint_pos.dtype).unsqueeze(-1)
      limits = self.robot.data.soft_joint_pos_limits[env_ids]
      joint_pos = torch.clip(joint_pos, limits[..., 0], limits[..., 1])
    joint_vel = torch.zeros_like(joint_pos)
    joint_vel[:, self._joint_ids] = self.clip.joint_vel[frames]
    if self.clip.passive_joint_names:
      joint_pos[:, self._passive_joint_ids] = self.clip.passive_joint_pos[frames]
      joint_vel[:, self._passive_joint_ids] = self.clip.passive_joint_vel[frames]
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # <DELTA> blind-delivery scatter on standing spawns (world x/y/yaw: the
    # walk delivery error lives on the horizontal floor).
    root_pos, root_quat = apply_spawn_pose_noise(
      root_pos,
      self._to_world_quat(self.clip.root_quat[frames], env_ids),
      disengaged=disengaged,
      xy_noise=self.cfg.spawn_xy_noise,
      yaw_noise=self.cfg.spawn_yaw_noise,
    )
    root_state = torch.cat(
      [
        root_pos,
        root_quat,
        self._to_world_vec(self.clip.root_lin_vel[frames], env_ids),
        self._to_world_vec(self.clip.root_ang_vel[frames], env_ids),
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.clear_state(env_ids=env_ids)
    self._affordance().seed_engaged(env_ids, self.clip.ee_engaged[frames])

  def _update_command(self) -> None:
    # R4 reseat freeze preserved; past the end the clock CLAMPS on the last
    # playable frame (no root_offset step bump: the prologue never tiles).
    frozen = self._birth_freeze_left > 0
    hold = self._affordance().reseat_hold | frozen
    next_steps = torch.where(hold, self.time_steps, self.time_steps + 1)
    self.time_steps = next_steps.clamp(max=self.clip.cycle_frames - 1)
    self._birth_freeze_left = (self._birth_freeze_left - 1).clamp(min=0)
    self.metrics["birth_frozen"][:] = frozen.to(self.metrics["birth_frozen"].dtype)
    if self.cfg.sampling_mode == "adaptive":
      alpha = self.cfg.adaptive_alpha
      self._bin_failed = alpha * self._current_bin_failed + (1.0 - alpha) * self._bin_failed
      self._current_bin_failed.zero_()

  @property
  def birth_frozen(self) -> torch.Tensor:
    """(E,) bool: inside the seeded-birth freeze window (clock held)."""
    return self._birth_freeze_left > 0

  @property
  def at_terminal_hold(self) -> torch.Tensor:
    """(E,) bool: the clock sits on the last playable frame."""
    return self.time_steps >= (self.clip.cycle_frames - 1)

  # -- FK tables -------------------------------------------------------------------

  @property
  def tracking_body_names(self) -> tuple[str, ...]:
    return self._tracking_body_names

  def tracking_body_indices(self, body_names: tuple[str, ...] | None) -> list[int]:
    if body_names is None:
      return list(range(len(self._tracking_body_names)))
    return [i for i, n in enumerate(self._tracking_body_names) if n in body_names]

  def _ensure_fk_tables(self) -> None:
    if self._fk_pos is not None:
      return
    mj_model = self._env.sim.mj_model
    indexing = self.robot.indexing
    root_qadr = indexing.free_joint_q_adr.cpu().numpy()
    joint_qadr_all = indexing.joint_q_adr.cpu().numpy()
    joint_qadr_clip = joint_qadr_all[self._joint_ids.cpu().numpy()]
    passive_qadr = joint_qadr_all[self._passive_joint_ids.cpu().numpy()]
    base_qpos = np.array(mj_model.qpos0, dtype=np.float64, copy=True)
    base_qpos[joint_qadr_all] = self.robot.data.default_joint_pos[0].detach().cpu().numpy()
    body_ids = [mj_model.body(f"{self.cfg.entity_name}/{name}").id for name in self._tracking_body_names]
    pos, quat = build_fk_tables(
      mj_model,
      base_qpos,
      root_qadr,
      joint_qadr_clip,
      np.asarray(body_ids),
      self.clip.root_pos.detach().cpu().numpy().astype(np.float64),
      self.clip.root_quat.detach().cpu().numpy().astype(np.float64),
      self.clip.joint_pos.detach().cpu().numpy().astype(np.float64),
      passive_qadr=passive_qadr if len(passive_qadr) else None,
      passive_pos=(
        self.clip.passive_joint_pos.detach().cpu().numpy().astype(np.float64) if len(passive_qadr) else None
      ),
    )
    lin, ang = finite_difference_velocities(pos, quat, self.clip.fps)

    def _t(arr: np.ndarray) -> torch.Tensor:
      return torch.tensor(np.ascontiguousarray(arr), dtype=torch.float32, device=self.device)

    self._fk_pos, self._fk_quat = _t(pos), _t(quat)
    self._fk_lin_vel, self._fk_ang_vel = _t(lin), _t(ang)

  def _world_quats(self, quats: torch.Tensor) -> torch.Tensor:
    wall = self.wall_quat.unsqueeze(1).expand(-1, quats.shape[1], -1)
    rotated = quat_mul(wall.reshape(-1, 4), quats.reshape(-1, 4)).reshape(quats.shape)
    vertical = self._vertical_mask(None).view(-1, 1, 1)
    return torch.where(vertical, quats, rotated)

  def _world_vecs(self, vecs: torch.Tensor) -> torch.Tensor:
    wall = self.wall_quat.unsqueeze(1).expand(-1, vecs.shape[1], -1)
    return quat_apply(wall.reshape(-1, 4), vecs.reshape(-1, 3)).reshape(vecs.shape)

  # -- reference side (world) ------------------------------------------------------

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    """Reference root position, world (E, 3) — tracked here, unlike climb."""
    return self.ref_root_pos_w

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.ref_root_quat_w

  @property
  def ref_joint_pos_rel(self) -> torch.Tensor:
    """Reference joint positions minus the default pose, (E, 21) — the
    ``joint_pos_rel`` basis (same 21-joint order, both resolved from
    PHYBOT_C2_POLICY_JOINT_NAMES with preserve_order)."""
    default = self.robot.data.default_joint_pos[:, self._joint_ids]
    return self.ref_joint_pos - default

  @property
  def ref_next_edge_countdown(self) -> torch.Tensor:
    """(E, 4): normalised countdown to each pad's NEXT scheduled edge —
    t_release while the chart says engaged, t_regrip while it says not
    (acyclic tables, sentinel 1.0 = never)."""
    edges = self.clip.ee_edge_times[self.time_steps]  # (E, 4, 2)
    return torch.where(self.ref_engaged, edges[..., 0], edges[..., 1])

  @property
  def command(self) -> torch.Tensor:
    """Transition80 motion_command, (E, 10) = [cos, sin, engaged(4),
    t_next_edge(4)] — see the module docstring (MOTION_COMMAND 10).
    ``ref_edge_countdowns`` (both lanes) stays available for the reward /
    termination / Phi code."""
    phase = self.time_steps.float() / self.clip.cycle_frames * (2.0 * torch.pi)
    return torch.cat(
      [
        torch.cos(phase).unsqueeze(-1),
        torch.sin(phase).unsqueeze(-1),
        self.ref_engaged.float(),
        self.ref_next_edge_countdown,
      ],
      dim=-1,
    )

  @property
  def ref_pad_center_w(self) -> torch.Tensor:
    """Reference pad FACE centers, world (E, 4, 3)."""
    return self._to_world_point(self._pad_center_canon[self.time_steps] + self.root_offset.unsqueeze(1))

  @property
  def robot_pad_center_w(self) -> torch.Tensor:
    """Robot pad FACE centers, world (E, 4, 3) — the contact law's point."""
    return self._affordance()._get_executor().pad_center_world()

  @property
  def body_pos_w(self) -> torch.Tensor:
    self._ensure_fk_tables()
    assert self._fk_pos is not None
    return self._to_world_point(self._fk_pos[self.time_steps] + self.root_offset.unsqueeze(1))

  @property
  def body_quat_w(self) -> torch.Tensor:
    self._ensure_fk_tables()
    assert self._fk_quat is not None
    return self._world_quats(self._fk_quat[self.time_steps])

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    self._ensure_fk_tables()
    assert self._fk_lin_vel is not None
    return self._world_vecs(self._fk_lin_vel[self.time_steps])

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    self._ensure_fk_tables()
    assert self._fk_ang_vel is not None
    return self._world_vecs(self._fk_ang_vel[self.time_steps])

  def _reanchored(self) -> tuple[torch.Tensor, torch.Tensor]:
    """mjlab ``update_relative_body_poses``: the reference skeleton
    re-planted at the robot root's xy + yaw (reference z, roll, pitch)."""
    body_pos, body_quat = self.body_pos_w, self.body_quat_w
    num_bodies = body_pos.shape[1]
    anchor_pos = self.anchor_pos_w.unsqueeze(1).expand(-1, num_bodies, -1)
    anchor_quat = self.anchor_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)
    robot_pos = self.robot_anchor_pos_w.unsqueeze(1).expand(-1, num_bodies, -1)
    robot_quat = self.robot_anchor_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)
    delta_pos = robot_pos.clone()
    delta_pos[..., 2] = anchor_pos[..., 2]
    delta_ori = yaw_quat(quat_mul(robot_quat.reshape(-1, 4), quat_inv(anchor_quat.reshape(-1, 4)))).reshape(
      -1, num_bodies, 4
    )
    quat_rel = quat_mul(delta_ori.reshape(-1, 4), body_quat.reshape(-1, 4)).reshape(body_quat.shape)
    pos_rel = delta_pos + quat_apply(delta_ori.reshape(-1, 4), (body_pos - anchor_pos).reshape(-1, 3)).reshape(
      body_pos.shape
    )
    return pos_rel, quat_rel

  @property
  def body_pos_target_w(self) -> torch.Tensor:
    return self.body_pos_w if self.cfg.relative_frame == "world" else self._reanchored()[0]

  @property
  def body_quat_target_w(self) -> torch.Tensor:
    return self.body_quat_w if self.cfg.relative_frame == "world" else self._reanchored()[1]

  # -- robot side -----------------------------------------------------------------

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.root_link_pos_w

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.root_link_quat_w

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self._body_ids]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self._body_ids]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self._body_ids]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self._body_ids]

  @property
  def pad_slots(self) -> torch.Tensor:
    """Slots of the four pad bodies inside the tracked set, (4,)."""
    return self._pad_slots

  # -- metrics ----------------------------------------------------------------------

  def _update_metrics(self) -> None:
    super()._update_metrics()
    self.metrics["error_anchor_pos"] = (self.anchor_pos_w - self.robot_anchor_pos_w).norm(dim=-1)
    body_err = (self.body_pos_target_w - self.robot_body_pos_w).norm(dim=-1)
    self.metrics["error_body_pos"] = body_err.mean(dim=-1)
    self.metrics["error_pad_pos"] = (self.ref_pad_center_w - self.robot_pad_center_w).norm(dim=-1).mean(dim=-1)


@dataclass(kw_only=True)
class TransitionMotionCommandCfg(ClimbMotionCommandCfg):
  """Configuration for :class:`TransitionMotionCommand`."""

  # "adaptive" = BeyondMimic failure-weighted start bins; "uniform" = plain
  # DeepMimic RSI; "start" = frame 0 (evaluation).
  sampling_mode: Literal["uniform", "start", "adaptive"] = "adaptive"
  # Blind-delivery spawn scatter on RSI frames with no engaged pads: base
  # x/y offset (m) and yaw (rad). Reference and anchors do NOT follow.
  spawn_xy_noise: tuple[float, float] = (0.0, 0.0)
  spawn_yaw_noise: float = 0.0
  # Fraction of RSI draws forced into the biased window (exam scripts pin
  # segments this way); 0 = no overlay.
  spawn_early_fraction: float = 0.0
  spawn_bias_segments: tuple[str, ...] = ()
  # The native-angle clip carries true world geometry — never lift standing
  # frames off the floor.
  spawn_match_vertical_height: bool = False
  # Tracked bodies (must contain every ``ee_body_names`` entry).
  body_names: tuple[str, ...] = DEFAULT_TRACKING_BODY_NAMES
  # "world": reference world poses (wall-fixed task); "anchor_yaw": mjlab
  # re-anchoring at the robot root xy+yaw.
  relative_frame: Literal["world", "anchor_yaw"] = "world"
  adaptive_alpha: float = 0.001
  adaptive_uniform_ratio: float = 0.1
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_max_bin_prob: float | None = 0.35
  # Which bin a failure is charged to. BeyondMimic charges the DEATH frame;
  # on a one-shot clip whose clock clamps on the last frame every late
  # death (contact-law executions at edge + grace for the last two edges)
  # lands in the final bin and the sampler spends its cap on handoff-hold
  # births that teach nothing about the earlier welds (v1e: 35 % of births
  # there, sampling entropy 0.69). "birth" charges the frame the episode
  # STARTED from — "which starts lead to death" — which is the signal an
  # acyclic clip needs.
  adaptive_attribution: Literal["birth", "death"] = "birth"
  # v3 (2026-08-22) SEEDED-BIRTH FREEZE: for this many steps after a birth
  # that seeds at least one weld, the clock holds on the birth frame (the
  # R4 reseat-freeze path) while the plant's seed judge guard covers the same
  # window — the RSI settle transient is absorbed with the reference standing
  # still and the seeds unjudged; clock and guard release together. 0 = off.
  # Births without a seeded pad never freeze.
  birth_freeze_steps: int = 0
  # v3 TENABLE-BIRTH MASK: (start, end) frame ranges (end exclusive) RSI never
  # births into — the passive-hold feasibility map's infeasible frames. Those
  # frames stay reachable by playing through from earlier births.
  birth_exclude_frames: tuple[tuple[int, int], ...] = ()
  # v4 (2026-08-22) IGNITION CLASSROOM: the ``spawn_early_fraction`` share of
  # births lands uniformly in explicit frame windows (start, end exclusive)
  # instead of the segment-name window. ``spawn_bias_auto_rake_frames`` > 0
  # derives the windows from the clip's engage rising edges as
  # [edge - n, edge) — the pre-landing rakes, where the pad already touches
  # the wall and the reference drags it down: the weld is one press-and-hold
  # away. Explicit windows take precedence over the auto rakes; both take
  # precedence over ``spawn_bias_segments``. The birth mask still applies.
  spawn_bias_frame_windows: tuple[tuple[int, int], ...] = ()
  spawn_bias_auto_rake_frames: int = 0

  def build(self, env: ManagerBasedRlEnv) -> TransitionMotionCommand:
    return TransitionMotionCommand(self, env)
