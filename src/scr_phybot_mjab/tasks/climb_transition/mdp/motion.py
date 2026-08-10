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
  """One-shot prologue playback (see the module docstring)."""

  cfg: MountMotionCommandCfg

  def __init__(self, cfg: MountMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
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
    window = None
    if cfg.spawn_bias_segments:
      window = resolve_segment_window(cfg.motion_file, cfg.spawn_bias_segments)
    if window is None:
      self._bias_window = (0, max(self._first_engage_frame, 1))
    else:
      cycle = self.clip.cycle_frames
      lo = int(window[0] * cycle)
      hi = max(int(window[1] * cycle), lo + 1)
      self._bias_window = (lo, min(hi, cycle))

  def _draw_rsi_frames(self, count: int) -> torch.Tensor:
    """Uniform RSI draw with a biased-window overlay.

    ``spawn_early_fraction`` of resets land uniformly inside
    ``self._bias_window``; the rest stay uniform over the whole clip.
    History: v2 aimed the window at [0, first_engage) to unstick the
    stand/raise skills (v1's uniform draw starved them at ~7% share); v4
    re-aims it at the place -> set latch stretch via
    ``spawn_bias_segments``.
    """
    frames = torch.randint(
      0, self.clip.cycle_frames, (count,), device=self.device
    )
    lo, hi = self._bias_window
    if self.cfg.spawn_early_fraction > 0.0 and hi > lo:
      early = (
        torch.rand(count, device=self.device) < self.cfg.spawn_early_fraction
      )
      frames[early] = torch.randint(
        lo, hi, (int(early.sum().item()),), device=self.device
      )
    return frames

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Mirror of ClimbMotionCommand._resample_command with two transition
    # deltas, marked <DELTA>: the biased RSI frame draw and the
    # blind-delivery spawn noise on disengaged frames. Kept as a full copy
    # because the parent offers no seam around its frame draw — sync
    # consciously if the parent's RSI ever changes.
    self._check_term_order()
    count = len(env_ids)
    if self.cfg.sampling_mode == "start":
      frames = torch.zeros(count, dtype=torch.long, device=self.device)
    else:
      frames = self._draw_rsi_frames(count)  # <DELTA> biased draw
    self.time_steps[env_ids] = frames

    # Wall frame cache (parent parity). The transition installs no
    # wall_angle curriculum, so this is always the identity: the mount runs
    # on the physically vertical wall — which is also why the <DELTA> spawn
    # noise below can stay in plain world x/y/yaw axes.
    wall = self._env.scene[self.cfg.wall_entity_name]
    self.wall_quat[env_ids] = wall.data.default_root_state[
      env_ids, 3:7
    ].clone()
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
    # <DELTA> standing spawns carry the blind-delivery pose scatter; frames
    # with seeded latches keep the exact clip pose. World x/y/yaw axes are
    # valid because the transition wall is always vertical (see above).
    root_pos, root_quat = _apply_spawn_pose_noise(
      root_pos,
      self._to_world_quat(self.clip.root_quat[frames], env_ids),
      disengaged=~self.clip.ee_engaged[frames].any(dim=-1),
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
    # step bump: the prologue never tiles.
    hold = self._affordance().reseat_hold
    next_steps = torch.where(hold, self.time_steps, self.time_steps + 1)
    self.time_steps = next_steps.clamp(max=self.clip.cycle_frames - 1)


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

  def build(self, env: ManagerBasedRlEnv) -> MountMotionCommand:
    return MountMotionCommand(self, env)
