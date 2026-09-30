"""Acyclic mount-prologue playback: the climb motion command, specialized.

``MountMotionCommand`` inherits the whole climb machinery (RSI, latch
seeding, reference properties, the R4 reseat freeze) and adds exactly the
three behaviors a one-shot prologue needs — implemented HERE so the cycle
task's module is never touched:

* TERMINAL HOLD — the phase clock clamps on the last playable frame (the
  clip's all-four ``handoff_hold`` stance) instead of wrapping; a wrap
  would teleport the reference back to the standing start under a hanging
  robot.
* ACYCLIC ENGAGE TABLES — the cycle loader computes edge countdowns / run
  lengths / edge distances circularly. On a one-shot clip the wrap
  manufactures phantom edges at the seam: measured on the mount prologue,
  every EE's release countdown read "imminent" near the terminal frame and
  the mismatch bill was grace-free at both ends (letting go at the clamped
  handoff stance was nearly unbilled). The tables are rebuilt without wrap
  after loading; events that never happen read the "never" sentinel (1.0).
* BLIND-DELIVERY SPAWN SCATTER — RSI frames with no engaged pads (standing
  starts) get uniform base x/y/yaw noise while the reference and its wall
  anchors stay canonical: the policy must absorb the walk handoff error
  through contact and square the yaw back up during the mount. Frames with
  seeded latches keep the exact clip pose — welded pads cannot move with
  the base.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.utils.lab_api.math import (
  quat_from_angle_axis,
  quat_mul,
  sample_uniform,
)
from scr_phybot_mjab.tasks.climb_wall.mdp.motion import (
  ClimbMotionClip,
  ClimbMotionCommand,
  ClimbMotionCommandCfg,
  vertical_height_spawn_offset,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _acyclic_edge_frame_countdowns(engaged: np.ndarray) -> np.ndarray:
  """One-shot counterpart of the cycle loader's edge countdowns.

  Same output convention — frames to each EE's next falling/rising edge,
  sentinel = full clip length meaning "never" — but computed without wrap:
  the next-frame array is padded with its own last value so the seam can
  never fire an edge, and past events never count.
  """
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
  """One-shot counterpart of the cycle loader's run lengths.

  Frame 0 always starts a run and the last run ends at the clip end — no
  wrap join, which in the circular version fuses the first and last runs
  of an acyclic prologue.
  """
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
  """Replace a loaded clip's circular engage tables with acyclic ones.

  Operates on the already-resampled cycle tensors, so it composes with
  ``ClimbMotionClip.from_dict`` unchanged (fps resampling, velocity
  differencing and the end_hold trim all happen before this).
  """
  engaged = clip.ee_engaged.cpu().numpy().astype(bool)
  cycle = engaged.shape[0]
  device = clip.ee_edge_times.device

  future = _acyclic_edge_frame_countdowns(engaged)
  past = _acyclic_edge_frame_countdowns(engaged[::-1])[::-1]
  edge_frames = np.minimum(future.min(axis=-1), past.min(axis=-1) - 1)
  edge_distance = (edge_frames / cycle).astype(np.float32)
  # Sentinel hygiene: a frame whose nearest "edge" on both sides is the
  # no-event sentinel must read exactly 1.0, not (cycle-1)/cycle. This also
  # covers EEs that never flip at all.
  edge_distance[edge_frames >= cycle - 1] = 1.0
  run_time = (_acyclic_run_lengths(engaged) / cycle).astype(np.float32)

  def _t(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(arr), device=device)

  clip.ee_edge_times = _t(future.astype(np.float32) / cycle)
  clip.ee_edge_distance = _t(edge_distance)
  clip.ee_run_time = _t(run_time)


def resolve_segment_window(
  motion_file: str, segments: tuple[str, ...]
) -> tuple[float, float] | None:
  """Frame-fraction window [lo, hi) of the named clip segments.

  Reads the raw clip (the loader discards segment labels) and returns the
  contiguous span covering every frame labeled with one of ``segments``,
  as fractions of the playable cycle — robust to retiming the clip. None
  when the file is absent (config-build before the asset exists) or no
  frame matches.
  """
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


def _wall_angle_from_wall_quat(quat: torch.Tensor) -> torch.Tensor:
  """Wall angle in degrees from the wall entity's wxyz quaternion, (E,).

  The wall_angle curriculum writes a pure Rot_y(90 - a) about the base
  line, so theta = 2*atan2(q_y, q_w) and a = 90 - deg(theta). Identity
  quaternion -> exactly 90.
  """
  theta = 2.0 * torch.atan2(quat[:, 2], quat[:, 0])
  return 90.0 - torch.rad2deg(theta)


def _merge_passive_state(clips: list[ClimbMotionClip]) -> dict:
  """Preserve CAD suspension state when combining old and regenerated clips.

  Legacy generators fixed the unrecorded gimbals at zero. Populate those
  rows explicitly when a family also contains a clip with passive state.
  """
  names = tuple(dict.fromkeys(name for clip in clips for name in clip.passive_joint_names))
  merged: dict = {"passive_joint_names": names}
  for field in ("passive_joint_pos", "passive_joint_vel"):
    parts = []
    for clip in clips:
      part = clip.joint_pos.new_zeros((clip.cycle_frames, len(names)))
      if clip.passive_joint_names:
        columns = [names.index(name) for name in clip.passive_joint_names]
        part[:, columns] = getattr(clip, field)
      parts.append(part)
    merged[field] = torch.cat(parts)
  return merged


def _merge_clips(
  clips: list[ClimbMotionClip],
) -> tuple[ClimbMotionClip, list[int]]:
  """Concatenate clips along the TIME axis into one flat table.

  The band machinery's core trick: every parent accessor indexes
  ``clip.X[time_steps]`` flat, so K native-angle clips laid end to end need
  no per-env gather — each env just lives inside its own [base, base+len)
  window (clock clamp, RSI draw and phase are windowed by the subclass).
  Returns the merged clip plus each input's base frame. Scalar semantics of
  ``cycle_frames`` become "total frames"; the subclass overrides every
  scalar consumer (phase, cycle_time_s, clamp, RSI).
  """
  first = clips[0]
  for clip in clips[1:]:
    if (
      clip.joint_names != first.joint_names
      or clip.ee_names != first.ee_names
      or clip.fps != first.fps
      or clip.step_height != first.step_height
    ):
      raise ValueError(
        "Family clips disagree on joints/EEs/fps/step_height — regenerate "
        "the band grid with one generator version."
      )
  bases: list[int] = []
  total = 0
  for clip in clips:
    bases.append(total)
    total += clip.cycle_frames
  fields = (
    "joint_pos",
    "joint_vel",
    "root_pos",
    "root_quat",
    "root_lin_vel",
    "root_ang_vel",
    "ee_pos_r",
    "ee_engaged",
    "ee_edge_times",
    "ee_edge_distance",
    "ee_run_time",
    "toe_braced",
    "ee_anchor_pos",
  )
  merged = {
    name: torch.cat([getattr(clip, name) for clip in clips], dim=0)
    for name in fields
  }
  return (
    ClimbMotionClip(
      joint_names=first.joint_names,
      ee_names=first.ee_names,
      fps=first.fps,
      step_height=first.step_height,
      cycle_frames=total,
      **_merge_passive_state(clips),
      **merged,
    ),
    bases,
  )


def stitch_mount_into_cycle(
  mount: ClimbMotionClip,
  cycle: ClimbMotionClip,
  joint_tol: float = 1e-4,
) -> tuple[ClimbMotionClip, int]:
  """Concatenate the mount prologue with the regrip cycle, seam-aware.

  The unified-episode table (docs/plans/ L-D): frames [0, Tm) replay the
  acyclic prologue, frames [Tm, Tm+Tc) the cycle period — the unified
  clock rolls INTO the cycle region and wraps there forever. Three seam
  obligations, each a lesson from the 08-09 phantom-edge hunt:

  * CONTINUITY — the prologue's terminal stance is the cycle's frame-0
    stance at a lower height, so the cycle segment (root positions and the
    anchor ladder) is rigidly shifted to end-match; joints/orientation are
    asserted continuous rather than trusted.
  * CROSS-SEAM LOOKAHEAD — the prologue tail's countdown lanes must see
    the CYCLE's upcoming edges (the acyclic rebuild's "never" sentinels
    would tell the policy no release is ever coming right before the first
    cycle regrip). Computed on [mount ‖ cycle ‖ cycle], rows [0, Tm),
    normalized by Tm and capped at the 1.0 sentinel.
  * SEGMENT-LOCAL NORMALIZATION — mount rows stay /Tm, cycle rows keep
    their circular tables VERBATIM (/Tc): each region's obs lanes match
    what that region's frozen teacher trained on.

  Returns the stitched clip plus the handoff frame Tm.
  """
  if (
    mount.joint_names != cycle.joint_names
    or mount.ee_names != cycle.ee_names
    or mount.fps != cycle.fps
  ):
    raise ValueError("Mount and cycle clips disagree on the clip contract.")
  t_mount = mount.cycle_frames
  t_cycle = cycle.cycle_frames

  # -- continuity: end-match the cycle segment onto the prologue terminal.
  shift = mount.root_pos[-1] - cycle.root_pos[0]
  if float(shift[:2].abs().max()) > 5e-3:
    raise ValueError(
      f"Handoff xy mismatch {shift[:2].tolist()} — the prologue does not "
      "end on the cycle's frame-0 stance."
    )
  joint_gap = float((mount.joint_pos[-1] - cycle.joint_pos[0]).abs().max())
  if joint_gap > joint_tol:
    raise ValueError(f"Handoff joint gap {joint_gap} exceeds {joint_tol}.")
  if not bool(mount.ee_engaged[-1].all() and cycle.ee_engaged[0].all()):
    raise ValueError("Handoff must be an all-four stance on both sides.")

  # -- cross-seam mount tables from the raw engage bools.
  m_eng = mount.ee_engaged.cpu().numpy().astype(bool)
  c_eng = cycle.ee_engaged.cpu().numpy().astype(bool)
  ext = np.concatenate([m_eng, c_eng, c_eng], axis=0)
  future = _acyclic_edge_frame_countdowns(ext)[:t_mount]
  future = np.minimum(future, t_mount)
  past = _acyclic_edge_frame_countdowns(m_eng[::-1])[::-1]
  past = np.minimum(past, t_mount)
  edge_frames = np.minimum(future.min(axis=-1), past.min(axis=-1) - 1)
  edge_distance = np.minimum(edge_frames, t_mount) / t_mount
  runs = _acyclic_run_lengths(np.concatenate([m_eng, c_eng], axis=0))[:t_mount]
  run_time = np.minimum(runs, t_mount) / t_mount

  device = mount.ee_edge_times.device

  def _t(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(
      np.ascontiguousarray(arr.astype(np.float32)), device=device
    )

  cat = torch.cat
  passive = _merge_passive_state([mount, cycle])
  if passive["passive_joint_names"] and not torch.allclose(
    passive["passive_joint_pos"][t_mount - 1],
    passive["passive_joint_pos"][t_mount], atol=joint_tol, rtol=0.0,
  ):
    raise ValueError("Mount/cycle passive joint poses do not match at the handoff")
  return (
    ClimbMotionClip(
      joint_names=mount.joint_names,
      ee_names=mount.ee_names,
      fps=mount.fps,
      # The stitched table's step bump is the CYCLE's (the wrap lives in
      # the cycle region; the prologue never tiles).
      step_height=cycle.step_height,
      cycle_frames=t_mount + t_cycle,
      joint_pos=cat([mount.joint_pos, cycle.joint_pos]),
      joint_vel=cat([mount.joint_vel, cycle.joint_vel]),
      root_pos=cat([mount.root_pos, cycle.root_pos + shift]),
      root_quat=cat([mount.root_quat, cycle.root_quat]),
      root_lin_vel=cat([mount.root_lin_vel, cycle.root_lin_vel]),
      root_ang_vel=cat([mount.root_ang_vel, cycle.root_ang_vel]),
      ee_pos_r=cat([mount.ee_pos_r, cycle.ee_pos_r]),
      ee_engaged=cat([mount.ee_engaged, cycle.ee_engaged]),
      ee_edge_times=cat([_t(future / t_mount), cycle.ee_edge_times]),
      ee_edge_distance=cat([_t(edge_distance), cycle.ee_edge_distance]),
      ee_run_time=cat([_t(run_time), cycle.ee_run_time]),
      toe_braced=cat([mount.toe_braced, cycle.toe_braced]),
      ee_anchor_pos=cat([mount.ee_anchor_pos, cycle.ee_anchor_pos + shift]),
      **passive,
    ),
    t_mount,
  )


def _apply_spawn_pose_noise(
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  disengaged: torch.Tensor,
  xy_noise: tuple[float, float],
  yaw_noise: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Uniform base x/y/yaw noise on rows where ``disengaged`` is True.

  Root velocities are left untouched: the disengaged prologue frames carry
  at most ~0.03 m/s, so rotating them by a few degrees changes nothing
  measurable.
  """
  if xy_noise[0] == 0.0 and xy_noise[1] == 0.0 and yaw_noise == 0.0:
    return root_pos, root_quat
  count = root_pos.shape[0]
  device = root_pos.device
  gate = disengaged.to(root_pos.dtype)
  pos = root_pos.clone()
  pos[:, 0] += gate * sample_uniform(
    -xy_noise[0], xy_noise[0], (count,), device=device
  )
  pos[:, 1] += gate * sample_uniform(
    -xy_noise[1], xy_noise[1], (count,), device=device
  )
  yaw = gate * sample_uniform(-yaw_noise, yaw_noise, (count,), device=device)
  axis = torch.zeros(count, 3, device=device, dtype=root_pos.dtype)
  axis[:, 2] = 1.0
  quat = quat_mul(quat_from_angle_axis(yaw, axis), root_quat)
  return pos, quat


class MountMotionCommand(ClimbMotionCommand):
  """One-shot prologue playback (see the module docstring).

  Band mode (``cfg.angle_clip_files`` non-empty): K native-angle clips are
  time-concatenated into one flat table (:func:`_merge_clips`); at reset
  the env's wall angle (written by the band-pinned wall_angle curriculum)
  selects the NEAREST native clip and the env lives inside that clip's
  frame window — clock clamp, RSI draw, bias window and phase are all
  windowed per env. The residual |sampled - native| <= 2 deg rotation
  flows through the parent's wall-frame machinery unchanged (the clips are
  stored de-rotated to the canonical frame). Single-clip mode keeps the v4
  code paths bit-for-bit.
  """

  cfg: MountMotionCommandCfg

  # Class-level defaults so stubbed unit tests (and the single-clip mode)
  # need none of the band machinery.
  _family: bool = False
  _env_end: torch.Tensor | None = None
  _uniform_frames: torch.Tensor | None = None
  _clamp_frame: int | None = None

  def __init__(self, cfg: MountMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    if cfg.angle_clip_files:
      self._init_clip_family(cfg, env)
      return
    rebuild_acyclic_schedule(self.clip)
    # First frame with any scheduled engagement: the boundary of the
    # "standing skills" stretch (stand -> raise -> place).
    self._first_engage_frame = int(
      self.clip.ee_engaged.any(dim=-1).int().argmax().item()
    )
    # Biased-spawn window (frames): the named clip segments when given,
    # else everything before the first engagement. v4 aims it at the
    # place -> set stretch — spawn AT the latch moment (arms already at the
    # anchors, one press away) so the press-to-latch skill trains directly
    # instead of waiting for the reach-up to be discovered first.
    window = self._bias_window_for(
      cfg, self.clip, self._first_engage_frame, cfg.motion_file
    )
    self._bias_window = window
    if cfg.spawn_disengaged_only:
      pool = (~self.clip.ee_engaged.any(dim=-1)).nonzero(as_tuple=False)
      pool = pool.squeeze(-1)
      if pool.numel() == 0:
        raise ValueError(
          "spawn_disengaged_only: the clip schedules engagement on every "
          "frame — there is no standing stretch to spawn into."
        )
      self._uniform_frames = pool
    if cfg.clamp_at_first_engage:
      # Stand-first clock: hold the last PRE-ENGAGE frame forever instead
      # of marching into the hanging stances. The episode then stays a
      # standing lesson end to end — the prefix-gated fence, executioner
      # and root-xy income never expire (the un-clamped clock walked past
      # first_engage at 5.6 s and turned all three off for the remaining
      # 24 s: the v20-v21 motionless wall-lean lived exactly there), the
      # reference stays reachable, and full pose income requires HOLDING
      # the raised place pose — the raise cannot be waited out.
      if self._first_engage_frame <= 0:
        raise ValueError(
          "clamp_at_first_engage: the clip schedules engagement at frame "
          "0 — there is no standing prefix to hold."
        )
      self._clamp_frame = self._first_engage_frame - 1

  @staticmethod
  def _bias_window_for(
    cfg: MountMotionCommandCfg,
    clip: ClimbMotionClip,
    first_engage_frame: int,
    motion_file: str,
  ) -> tuple[int, int]:
    """Biased-spawn frame window for ONE clip (local frames)."""
    window = None
    if cfg.spawn_bias_segments:
      window = resolve_segment_window(motion_file, cfg.spawn_bias_segments)
    if window is None:
      return (0, max(first_engage_frame, 1))
    cycle = clip.cycle_frames
    lo = int(window[0] * cycle)
    hi = max(int(window[1] * cycle), lo + 1)
    return (lo, min(hi, cycle))

  def _init_clip_family(
    self, cfg: MountMotionCommandCfg, env: ManagerBasedRlEnv
  ) -> None:
    """Load the native-angle grid and switch to windowed playback."""
    if cfg.spawn_disengaged_only or cfg.clamp_at_first_engage:
      raise ValueError(
        "spawn_disengaged_only/clamp_at_first_engage are single-clip only "
        "(no band stand task)."
      )
    angles: list[float] = []
    clips: list[ClimbMotionClip] = []
    bias_lo: list[int] = []
    bias_hi: list[int] = []
    for angle, path in cfg.angle_clip_files:
      clip = ClimbMotionClip.from_file(
        path, step_dt=env.step_dt, device=str(self.device)
      )
      if (
        clip.joint_names != self.clip.joint_names
        or clip.ee_names != self.clip.ee_names
      ):
        raise ValueError(
          f"Band clip {path} disagrees with {cfg.motion_file} on the "
          "joint/EE contract."
        )
      rebuild_acyclic_schedule(clip)
      angles.append(float(angle))
      clips.append(clip)
    merged, bases = _merge_clips(clips)
    self.clip = merged
    self._resolve_passive_joint_ids()
    for base, clip, (_, path) in zip(bases, clips, cfg.angle_clip_files):
      first_engage = int(clip.ee_engaged.any(dim=-1).int().argmax().item())
      lo, hi = self._bias_window_for(cfg, clip, first_engage, path)
      bias_lo.append(base + lo)
      bias_hi.append(base + hi)

    def _lt(values: list[int]) -> torch.Tensor:
      return torch.tensor(values, dtype=torch.long, device=self.device)

    self._family = True
    self._grid_angles = torch.tensor(angles, device=self.device)
    self._slot_base = _lt(bases)
    self._slot_len = _lt([clip.cycle_frames for clip in clips])
    self._slot_end = self._slot_base + self._slot_len - 1
    self._slot_bias_lo = _lt(bias_lo)
    self._slot_bias_hi = _lt(bias_hi)
    fill = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._env_base = fill + self._slot_base[0]
    self._env_len = fill + self._slot_len[0]
    self._env_end = fill + self._slot_end[0]

  def _resolve_passive_joint_ids(self) -> None:
    passive_ids, passive_names = self.robot.find_joints(
      list(self.clip.passive_joint_names), preserve_order=True
    ) if self.clip.passive_joint_names else ([], [])
    if tuple(passive_names) != self.clip.passive_joint_names:
      raise ValueError("Reference passive joints do not match the robot")
    self._passive_joint_ids = torch.tensor(passive_ids, dtype=torch.long, device=self.device)

  def _draw_rsi_frames(
    self, count: int, slots: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Uniform RSI draw with a biased-window overlay.

    ``spawn_early_fraction`` of resets land uniformly inside the biased
    window; the rest stay uniform over the whole clip. History: v2 aimed
    the window at [0, first_engage) to unstick the stand/raise skills
    (v1's uniform draw starved them at ~7% share); v4 re-aims it at the
    place -> set latch stretch via ``spawn_bias_segments``.

    ``slots`` (band mode) restricts every draw to each env's own clip
    window and its per-slot bias stretch.
    """
    if slots is None:
      pool = self._uniform_frames
      if pool is None:
        frames = torch.randint(
          0, self.clip.cycle_frames, (count,), device=self.device
        )
      else:
        # <DELTA> stand-first: the uniform half draws standing frames only.
        # With the engage economy offline a wall-hanging spawn has no
        # income channel to hold itself — it can only rip and die (v17:
        # 1.55 Hz forced releases from half the batch being unlearnable).
        frames = pool[
          torch.randint(0, pool.numel(), (count,), device=self.device)
        ]
      lo, hi = self._bias_window
      if self.cfg.spawn_early_fraction > 0.0 and hi > lo:
        early = (
          torch.rand(count, device=self.device) < self.cfg.spawn_early_fraction
        )
        frames[early] = torch.randint(
          lo, hi, (int(early.sum().item()),), device=self.device
        )
      edge_frac = getattr(self.cfg, "spawn_edge_fraction", 0.0)
      if edge_frac > 0.0:
        # <DELTA> v37 micro-window (the v12-era card (c), finally played):
        # a slice of births lands in the final moments BEFORE the first
        # engage edge — arms on the anchors, the reference about to press
        # — so the press->debounce->weld conversion is one small action
        # away from birth instead of eleven seconds of choreography away.
        fe = self._first_engage_frame
        k = max(
          1,
          int(
            round(
              getattr(self.cfg, "spawn_edge_window_s", 0.3) * self.clip.fps
            )
          ),
        )
        lo_e = max(0, fe - k)
        edge_draw = torch.randint(
          lo_e, max(fe, lo_e + 1), (count,), device=self.device
        )
        sel = torch.rand(count, device=self.device) < edge_frac
        frames = torch.where(sel, edge_draw, frames)
      return frames

    base = self._slot_base[slots]
    length = self._slot_len[slots]
    u = torch.rand(count, device=self.device)
    frames = base + torch.minimum((u * length.float()).long(), length - 1)
    if self.cfg.spawn_early_fraction > 0.0:
      lo = self._slot_bias_lo[slots]
      width = (self._slot_bias_hi[slots] - lo).clamp(min=1)
      v = torch.rand(count, device=self.device)
      biased = lo + torch.minimum((v * width.float()).long(), width - 1)
      early = (
        torch.rand(count, device=self.device) < self.cfg.spawn_early_fraction
      )
      frames = torch.where(early, biased, frames)
    return frames

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Mirror of ClimbMotionCommand._resample_command with two transition
    # deltas, marked <DELTA>: the biased RSI frame draw and the
    # blind-delivery spawn noise on disengaged frames. Kept as a full copy
    # because the parent offers no seam around its frame draw — sync
    # consciously if the parent's RSI ever changes.
    self._check_term_order()
    count = len(env_ids)

    # Wall frame cache (parent parity), read BEFORE the frame draw: band
    # variants derive the per-env clip slot from the sampled wall angle
    # (the band-pinned wall_angle curriculum wrote the pose earlier in
    # this same reset pass). The 90 task installs no wall_angle
    # curriculum, so there this stays the identity. The <DELTA> spawn
    # noise below keeps plain world x/y/yaw axes at every angle: the
    # scatter models the WALK delivery error of a robot standing on the
    # horizontal floor — floor-plane semantics, not wall-frame ones.
    wall = self._env.scene[self.cfg.wall_entity_name]
    self.wall_quat[env_ids] = wall.data.default_root_state[
      env_ids, 3:7
    ].clone()

    if self._family:
      angles = _wall_angle_from_wall_quat(self.wall_quat[env_ids])
      slots = torch.argmin(
        (angles.unsqueeze(1) - self._grid_angles.unsqueeze(0)).abs(), dim=1
      )
      self._env_base[env_ids] = self._slot_base[slots]
      self._env_len[env_ids] = self._slot_len[slots]
      self._env_end[env_ids] = self._slot_end[slots]
      if self.cfg.sampling_mode == "start":
        frames = self._slot_base[slots]
      else:
        frames = self._draw_rsi_frames(count, slots)  # <DELTA> biased draw
    elif self.cfg.sampling_mode == "start":
      frames = torch.zeros(count, dtype=torch.long, device=self.device)
    else:
      frames = self._draw_rsi_frames(count)  # <DELTA> biased draw
    self.time_steps[env_ids] = frames
    offset = torch.zeros(count, 3, device=self.device)
    offset[:, 2] = sample_uniform(
      self.cfg.height_offset_range[0],
      self.cfg.height_offset_range[1],
      (count,),
      device=self.device,
    )
    if self.cfg.spawn_match_vertical_height:
      # Parent parity; exactly zero on the transition's identity wall.
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
    # <DELTA> v62e2 (user call, 08-19): the seeded-frame exemption xy/yaw
    # noise always had, extended to JOINT noise. A seeded birth welds the
    # pad AT the anchor while noised joints put it mm-cm away — the weld
    # constraint bites the mismatch back within the first steps, a
    # stiffness impulse that gravity scaling cannot shrink (the 0.6 g
    # package saw rips RISE when the 2-step guard exposed it). Standing
    # spawns keep full joint noise; welded pads cannot move with the chain.
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
    # <DELTA> standing spawns carry the blind-delivery pose scatter; frames
    # with seeded latches keep the exact clip pose. World x/y/yaw axes are
    # valid because the transition wall is always vertical (see above).
    root_pos, root_quat = _apply_spawn_pose_noise(
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
    # The R4 reseat freeze is preserved; past the end the clock CLAMPS on
    # the last playable frame (the all-four handoff stance). No root_offset
    # step bump: the prologue never tiles. Band mode clamps on each env's
    # OWN clip end.
    hold = self._affordance().reseat_hold
    next_steps = torch.where(hold, self.time_steps, self.time_steps + 1)
    if self._env_end is None:
      end = (
        self.clip.cycle_frames - 1
        if self._clamp_frame is None
        else self._clamp_frame
      )
      self.time_steps = next_steps.clamp(max=end)
    else:
      self.time_steps = torch.minimum(next_steps, self._env_end)

  @property
  def at_terminal_hold(self) -> torch.Tensor:
    """(E,) bool: the clock sits on the env's last playable frame."""
    if self._env_end is None:
      return self.time_steps >= (self.clip.cycle_frames - 1)
    return self.time_steps >= self._env_end

  @property
  def command(self) -> torch.Tensor:
    # <DELTA> vs parent: band mode reads a LOCAL phase — each env's clock
    # normalized by its own clip window, not by the merged table length.
    if not self._family:
      return ClimbMotionCommand.command.fget(self)
    local = (self.time_steps - self._env_base).float()
    phase = local / self._env_len.float() * (2.0 * torch.pi)
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
  def cycle_time_s(self) -> torch.Tensor | float:
    # <DELTA> vs parent: band clips may differ in length, so the duration
    # is per-env, (E, 1) — every reward consumer multiplies/divides a
    # (E, H) tensor by it and broadcasts cleanly.
    if not self._family:
      return ClimbMotionCommand.cycle_time_s.fget(self)
    return (self._env_len.float() / self.clip.fps).unsqueeze(-1)


@dataclass(kw_only=True)
class MountMotionCommandCfg(ClimbMotionCommandCfg):
  """Configuration for :class:`MountMotionCommand`."""

  # Blind-delivery spawn scatter, applied ONLY to RSI frames with no
  # engaged pads (standing starts): uniform base x/y offset (m) and yaw
  # (rad). The reference and its wall anchors do NOT follow the offset.
  spawn_xy_noise: tuple[float, float] = (0.0, 0.0)
  spawn_yaw_noise: float = 0.0
  # Fraction of RSI draws forced into the biased window; the rest stay
  # uniform over the whole clip. 0.0 = plain DeepMimic uniform.
  spawn_early_fraction: float = 0.0
  # Clip segments defining the biased window (resolved from the raw clip's
  # labels, robust to retiming). Empty = everything before the first
  # scheduled engagement (the v2 stand/raise window).
  spawn_bias_segments: tuple[str, ...] = ()
  # Restrict the UNIFORM half of the RSI draw to frames with no scheduled
  # engagement (the biased window is unaffected — its segments are chosen
  # by name). Stand-first sets this: with the engage economy offline a
  # wall-hanging spawn has nothing paying it to hold on and can only rip
  # and die. Single-clip mode only.
  spawn_disengaged_only: bool = False
  # v37 micro-window: fraction of RSI births pinned to the final
  # ``spawn_edge_window_s`` seconds before the first engage edge (arms on
  # the anchors, one press from the weld). Single-clip draws only; the
  # band draw ignores it.
  spawn_edge_fraction: float = 0.0
  spawn_edge_window_s: float = 0.3
  # Clamp the clock on the last pre-engage frame instead of the clip end:
  # the whole episode stays a standing lesson (fence/executioner/root-xy
  # gates never expire, the reference never leaves reach, and full pose
  # income requires holding the raised place pose). Stand-first only;
  # single-clip mode only.
  clamp_at_first_engage: bool = False
  # Band mode: ((native_angle_deg, clip_path), ...) — the discrete grid of
  # native-angle mount clips covering one deploy band. Empty = single-clip
  # playback of ``motion_file`` (the v4 code paths, bit-for-bit). When
  # non-empty, ``motion_file`` should point at the band's center clip (it
  # is loaded for the contract check and as the tools' default), and the
  # env cfg must install a band-pinned wall_angle curriculum — the sampled
  # wall angle picks the nearest native clip per env at reset.
  angle_clip_files: tuple[tuple[float, str], ...] = ()

  def build(self, env: ManagerBasedRlEnv) -> MountMotionCommand:
    return MountMotionCommand(self, env)


class UnifiedMountClimbCommand(MountMotionCommand):
  """Full-chain playback: prologue rolls INTO the cycle and wraps there.

  The unified-episode command (docs/plans/ L-D): the stitched table from
  :func:`stitch_mount_into_cycle` replaces the terminal clamp with a
  roll-in — past the handoff frame the clock behaves exactly like the
  cycle parent (wrap + one step-height anchor-ladder bump per loop), so a
  policy trained here has no "switch": the mount is just the episode's
  first Tm frames. Phase and cycle_time_s are SEGMENT-LOCAL (mount rows
  /Tm, cycle rows /Tc), matching what each region's frozen teacher
  trained on; the KL router reads :attr:`in_cycle_region`.
  """

  cfg: UnifiedMountClimbCommandCfg

  def __init__(
    self, cfg: UnifiedMountClimbCommandCfg, env: ManagerBasedRlEnv
  ):
    if cfg.angle_clip_files:
      raise ValueError(
        "Band-unified is a follow-up: stitch per native clip before "
        "enabling angle_clip_files here."
      )
    super().__init__(cfg, env)
    cycle = ClimbMotionClip.from_file(
      cfg.cycle_motion_file, step_dt=env.step_dt, device=str(self.device)
    )
    self.clip, handoff = stitch_mount_into_cycle(self.clip, cycle)
    self._resolve_passive_joint_ids()
    self._handoff_frame = handoff
    self._cycle_frames = cycle.cycle_frames
    seam = int(round(cfg.seam_window_s * self.clip.fps))
    self._seam_window = (
      max(0, handoff - seam),
      min(self.clip.cycle_frames, handoff + seam),
    )

  # -- segment machinery ------------------------------------------------------

  @property
  def in_cycle_region(self) -> torch.Tensor:
    """(E,) bool: past the handoff frame — the cycle teacher's territory."""
    return self.time_steps >= self._handoff_frame

  @property
  def at_terminal_hold(self) -> torch.Tensor:
    # For the unified table "handoff-ready" means the chain crossed the
    # seam (there is no terminal clamp to sit on).
    return self.time_steps >= (self._handoff_frame - 1)

  def _draw_rsi_frames(
    self, count: int, slots: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Three-way episode mix: full chain / on-cycle RSI / seam RSI."""
    del slots
    p_full, p_cycle, p_seam = self.cfg.episode_mix
    total = max(p_full + p_cycle + p_seam, 1e-6)
    u = torch.rand(count, device=self.device)
    frames = torch.zeros(count, dtype=torch.long, device=self.device)
    cycle_draw = torch.randint(
      self._handoff_frame,
      self.clip.cycle_frames,
      (count,),
      device=self.device,
    )
    lo, hi = self._seam_window
    seam_draw = torch.randint(lo, max(hi, lo + 1), (count,), device=self.device)
    in_cycle = u >= (p_full / total)
    in_seam = u >= ((p_full + p_cycle) / total)
    frames = torch.where(in_cycle, cycle_draw, frames)
    frames = torch.where(in_seam, seam_draw, frames)
    return frames

  def _update_command(self) -> None:
    # Mount region: plain advance (rolls into the cycle at Tm). Cycle
    # region: parent-parity wrap back to Tm with the step-height bump.
    hold = self._affordance().reseat_hold
    next_steps = torch.where(hold, self.time_steps, self.time_steps + 1)
    wrapped = next_steps >= self.clip.cycle_frames
    next_steps = torch.where(
      wrapped, next_steps - self._cycle_frames, next_steps
    )
    self.root_offset[:, 2] += wrapped.float() * self.clip.step_height
    self.time_steps = next_steps

  @property
  def command(self) -> torch.Tensor:
    # Segment-local phase (see class docstring).
    in_cycle = self.in_cycle_region
    local = torch.where(
      in_cycle, self.time_steps - self._handoff_frame, self.time_steps
    ).float()
    denom = torch.where(
      in_cycle,
      torch.full_like(self.time_steps, self._cycle_frames),
      torch.full_like(self.time_steps, self._handoff_frame),
    ).float()
    phase = local / denom * (2.0 * torch.pi)
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
  def cycle_time_s(self) -> torch.Tensor:
    denom = torch.where(
      self.in_cycle_region,
      torch.full_like(self.time_steps, self._cycle_frames),
      torch.full_like(self.time_steps, self._handoff_frame),
    ).float()
    return (denom / self.clip.fps).unsqueeze(-1)


@dataclass(kw_only=True)
class UnifiedMountClimbCommandCfg(MountMotionCommandCfg):
  """Configuration for :class:`UnifiedMountClimbCommand`.

  ``motion_file`` is the mount prologue, ``cycle_motion_file`` the regrip
  cycle it rolls into.
  """

  cycle_motion_file: str = ""
  # Episode-start mix (full chain from standing / on-cycle RSI / seam RSI),
  # normalized internally. Plan initial values 0.3/0.5/0.2: the cycle keeps
  # its experience share, the seam gets deliberate coverage.
  episode_mix: tuple[float, float, float] = (0.3, 0.5, 0.2)
  # Half-width (s) of the seam RSI window around the handoff frame.
  seam_window_s: float = 0.5

  def build(self, env: ManagerBasedRlEnv) -> UnifiedMountClimbCommand:
    return UnifiedMountClimbCommand(self, env)
