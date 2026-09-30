"""Unit tests for the transition task's MountMotionCommand machinery.

The subclass lives in ``tasks/climb_transition`` and must (a) rebuild the
engage tables acyclically, (b) clamp the clock instead of wrapping, and
(c) scatter only disengaged spawns — while the climb parent keeps its
circular/wrapping behavior byte-for-byte (pinned by test_climb_motion.py).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scr_phybot_mjab.tasks.climb_transition.mdp.motion import (
  MountMotionCommand,
  UnifiedMountClimbCommand,
  _apply_spawn_pose_noise,
  _merge_clips,
  _wall_angle_from_wall_quat,
  rebuild_acyclic_schedule,
  resolve_segment_window,
  stitch_mount_into_cycle,
)
from scr_phybot_mjab.tasks.climb_wall.mdp.curriculums import (
  wall_pose_for_angle,
)
from scr_phybot_mjab.tasks.climb_wall.mdp.motion import (
  ClimbMotionClip,
  ClimbMotionCommand,
)

_YAW90_XYZW = (0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0))


def _synthetic_clip(fps: float = 50.0) -> dict:
  """Mirror of test_climb_motion._synthetic_clip: 4-frame cycle + 2-frame
  end_hold tail, feet scheduled OFF at frames 1-2."""
  frames = 6
  t = np.arange(frames, dtype=np.float32)
  joint_positions = np.stack([0.1 * t, np.full(frames, -0.5)], axis=1)
  root_position = np.zeros((frames, 3), dtype=np.float32)
  root_position[:, 2] = 1.0 + 0.02 * t
  root_quaternion = np.tile(np.array(_YAW90_XYZW, dtype=np.float32), (frames, 1))
  ee_positions = root_position[:, None, :] + np.array([1.0, 0.0, 0.0])
  ee_positions = np.repeat(ee_positions, 4, axis=1)
  ee_engaged = np.ones((frames, 4), dtype=np.uint8)
  ee_engaged[1:3, 2:] = 0
  return {
    "joints_list": ["j0", "j1"],
    "joint_positions": joint_positions,
    "root_position": root_position,
    "root_quaternion": root_quaternion,
    "fps": np.float32(fps),
    "ee_list": ["lh", "rh", "lf", "rf"],
    "ee_positions": ee_positions.astype(np.float32),
    "ee_anchor_positions": ee_positions.astype(np.float32),
    "ee_engaged": ee_engaged,
    "phase_segment": np.array([0, 0, 0, 0, 1, 1], dtype=np.uint8),
    "segment_names": ["hold", "end_hold"],
    "cycle_index": np.zeros(frames, dtype=np.uint16),
    "step_height": np.float32(0.08),
  }


def test_rebuild_replaces_circular_tables_with_acyclic_ones():
  """No phantom wrap edges at either end of a one-shot clip.

  The circular numbers on this exact clip are pinned by
  ``test_climb_motion.test_loader_derives_schedule_edge_countdowns`` — the
  values here differ exactly where the wrap used to manufacture edges.
  """
  clip = ClimbMotionClip.from_dict(_synthetic_clip(), step_dt=0.02, device="cpu")
  rebuild_acyclic_schedule(clip)

  # Hands never flip: both lanes read the "never" sentinel everywhere.
  assert torch.allclose(clip.ee_edge_times[:, 0], torch.ones(4, 2))
  assert torch.allclose(clip.ee_edge_times[:, 1], torch.ones(4, 2))
  # Feet: after the LAST real edge every lane reads 1.0 instead of
  # counting down to a wrap phantom.
  for foot in (2, 3):
    assert torch.allclose(
      clip.ee_edge_times[:, foot, 0], torch.tensor([0.25, 1.0, 1.0, 1.0])
    )
    assert torch.allclose(
      clip.ee_edge_times[:, foot, 1], torch.tensor([0.75, 0.5, 0.25, 1.0])
    )
  # Edge-free hand rows read exactly 1.0 (the both-sentinel case must not
  # leak (cycle-1)/cycle = 0.75).
  assert torch.allclose(clip.ee_edge_distance[:, 0], torch.ones(4))
  assert torch.allclose(clip.ee_edge_distance[:, 1], torch.ones(4))
  # Interior edges are identical under both computations.
  for foot in (2, 3):
    assert torch.allclose(
      clip.ee_edge_distance[:, foot], torch.tensor([0.25, 0.0, 0.25, 0.0])
    )
  # Runs do not fuse across the seam: the feet's frame-0 ON run is a single
  # frame (circular reads 0.5 by joining frames 3 and 0).
  for foot in (2, 3):
    assert torch.allclose(
      clip.ee_run_time[:, foot], torch.tensor([0.25, 0.5, 0.5, 0.25])
    )


def _make_update_stub(cls, reseat_hold: torch.Tensor):
  cmd = object.__new__(cls)
  cmd.clip = SimpleNamespace(cycle_frames=4, step_height=0.08)
  cmd.time_steps = torch.tensor([2, 3], dtype=torch.long)
  cmd.root_offset = torch.zeros(2, 3)
  cmd._affordance = lambda: SimpleNamespace(reseat_hold=reseat_hold)
  cmd.cfg = SimpleNamespace(synchronize_reset_phase=False, initialize_seeded_anchors=False)
  cmd._fresh = torch.zeros(2, dtype=torch.bool)
  cmd.phase_sampler = None
  return cmd


def test_mount_command_clamps_where_the_parent_wraps():
  cmd = _make_update_stub(MountMotionCommand, torch.tensor([False, False]))
  cmd._update_command()
  assert cmd.time_steps.tolist() == [3, 3]
  cmd._update_command()
  assert cmd.time_steps.tolist() == [3, 3]  # clamped, no wrap
  assert torch.allclose(cmd.root_offset, torch.zeros(2, 3))  # no step bump

  # The R4 reseat freeze still wins over the clamp advance.
  held = _make_update_stub(MountMotionCommand, torch.tensor([True, False]))
  held._update_command()
  assert held.time_steps.tolist() == [2, 3]

  # Parent regression: the cycle command still wraps and tiles upward.
  parent = _make_update_stub(ClimbMotionCommand, torch.tensor([False, False]))
  parent._update_command()
  assert parent.time_steps.tolist() == [3, 0]
  assert parent.root_offset[1, 2].item() == pytest.approx(0.08)

  # v23 stand clock: with _clamp_frame set the clock holds the last
  # PRE-ENGAGE frame instead of the clip end — the standing-prefix gates
  # (fence, executioner, root-xy) never expire.
  held_early = _make_update_stub(MountMotionCommand, torch.tensor([False, False]))
  held_early._clamp_frame = 2
  held_early._update_command()
  assert held_early.time_steps.tolist() == [2, 2]
  held_early._update_command()
  assert held_early.time_steps.tolist() == [2, 2]


def test_spawn_pose_noise_only_touches_disengaged_rows():
  pos = torch.tensor([[1.27, 0.0, 0.726], [1.327, 0.0, 1.0]])
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
  disengaged = torch.tensor([True, False])

  out_pos, out_quat = _apply_spawn_pose_noise(
    pos, quat, disengaged, xy_noise=(0.05, 0.03), yaw_noise=0.1
  )
  # Engaged row: exact clip pose (welded pads cannot move with the base).
  assert torch.equal(out_pos[1], pos[1])
  assert torch.equal(out_quat[1], quat[1])
  # Disengaged row: bounded offsets, pure z rotation, unit quaternion.
  assert abs(out_pos[0, 0] - pos[0, 0]) <= 0.05 + 1e-6
  assert abs(out_pos[0, 1] - pos[0, 1]) <= 0.03 + 1e-6
  assert out_pos[0, 2] == pos[0, 2]
  assert abs(out_quat[0, 1]) < 1e-6 and abs(out_quat[0, 2]) < 1e-6
  assert out_quat[0].norm().item() == pytest.approx(1.0, abs=1e-5)
  assert abs(out_quat[0, 3]) <= np.sin(0.05) + 1e-6

  # Zero ranges: identity fast path, inputs returned untouched.
  same_pos, same_quat = _apply_spawn_pose_noise(
    pos, quat, disengaged, xy_noise=(0.0, 0.0), yaw_noise=0.0
  )
  assert same_pos is pos and same_quat is quat


def test_rsi_bias_window_draw():
  cmd = object.__new__(MountMotionCommand)
  cmd.cfg = SimpleNamespace(spawn_early_fraction=0.5)
  cmd.clip = SimpleNamespace(cycle_frames=710)
  cmd._env = SimpleNamespace(device="cpu")  # .device property reads this
  cmd._bias_window = (280, 380)  # e.g. the place -> set latch stretch
  torch.manual_seed(0)
  frames = cmd._draw_rsi_frames(4096)
  inside = ((frames >= 280) & (frames < 380)).float().mean().item()
  # 50% forced into the window + the uniform tail's own ~14% share.
  assert 0.5 < inside < 0.68, inside
  assert int(frames.max()) < 710 and int(frames.min()) >= 0

  # Bias off: plain uniform.
  cmd.cfg = SimpleNamespace(spawn_early_fraction=0.0)
  frames = cmd._draw_rsi_frames(4096)
  assert ((frames >= 280) & (frames < 380)).float().mean().item() < 0.25


def test_wall_angle_quat_roundtrip():
  """The band slot picker decodes the SAME quat the curriculum writes."""
  angles = torch.tensor([76.0, 80.0, 84.0, 88.0, 90.0])
  _, quat = wall_pose_for_angle(angles)
  decoded = _wall_angle_from_wall_quat(quat)
  assert torch.allclose(decoded, angles, atol=1e-4)


def _clip_of_frames(cycle: int) -> ClimbMotionClip:
  """Synthetic clip with ``cycle`` playable frames (+2 end_hold tail)."""
  frames = cycle + 2
  data = _synthetic_clip()
  t = np.arange(frames, dtype=np.float32)
  data["joint_positions"] = np.stack([0.1 * t, np.full(frames, -0.5)], axis=1)
  root = np.zeros((frames, 3), dtype=np.float32)
  root[:, 2] = 1.0 + 0.02 * t
  data["root_position"] = root
  data["root_quaternion"] = np.tile(
    np.array(_YAW90_XYZW, dtype=np.float32), (frames, 1)
  )
  ee = root[:, None, :] + np.array([1.0, 0.0, 0.0])
  ee = np.repeat(ee, 4, axis=1).astype(np.float32)
  data["ee_positions"] = ee
  data["ee_anchor_positions"] = ee
  engaged = np.ones((frames, 4), dtype=np.uint8)
  engaged[1:3, 2:] = 0
  data["ee_engaged"] = engaged
  labels = np.zeros(frames, dtype=np.uint8)
  labels[-2:] = 1
  data["phase_segment"] = labels
  data["cycle_index"] = np.zeros(frames, dtype=np.uint16)
  return ClimbMotionClip.from_dict(data, step_dt=0.02, device="cpu")


def test_merge_clips_concatenates_along_time():
  a = _clip_of_frames(4)
  b = _clip_of_frames(8)
  rebuild_acyclic_schedule(a)
  rebuild_acyclic_schedule(b)
  merged, bases = _merge_clips([a, b])
  assert bases == [0, 4]
  assert merged.cycle_frames == 12
  for name in ("joint_pos", "ee_engaged", "ee_edge_times", "ee_anchor_pos"):
    field = getattr(merged, name)
    assert field.shape[0] == 12
    assert torch.equal(field[:4], getattr(a, name))
    assert torch.equal(field[4:], getattr(b, name))
  # Family contract violations refuse to merge.
  bad = _clip_of_frames(4)
  bad.ee_names = ("x",) * 4
  with pytest.raises(ValueError):
    _merge_clips([a, bad])


def test_merge_family_preserves_regenerated_passive_state():
  legacy, cad = _clip_of_frames(4), _clip_of_frames(8)
  cad.passive_joint_names = ("left_hand_spine_gimbal_x",)
  cad.passive_joint_pos = torch.full((8, 1), -0.0523598776)
  cad.passive_joint_vel = torch.full((8, 1), 0.03)
  merged, bases = _merge_clips([legacy, cad])
  assert bases == [0, 4]
  assert merged.passive_joint_names == cad.passive_joint_names
  assert not merged.passive_joint_pos[:4].any()
  assert not merged.passive_joint_vel[:4].any()
  assert torch.equal(merged.passive_joint_pos[4:], cad.passive_joint_pos)
  assert torch.equal(merged.passive_joint_vel[4:], cad.passive_joint_vel)


def _banded_stub(spawn_early_fraction: float = 0.0):
  cmd = object.__new__(MountMotionCommand)
  cmd.cfg = SimpleNamespace(spawn_early_fraction=spawn_early_fraction)
  cmd._env = SimpleNamespace(device="cpu")
  cmd._family = True
  cmd._slot_base = torch.tensor([0, 4], dtype=torch.long)
  cmd._slot_len = torch.tensor([4, 8], dtype=torch.long)
  cmd._slot_end = torch.tensor([3, 11], dtype=torch.long)
  cmd._slot_bias_lo = torch.tensor([1, 6], dtype=torch.long)
  cmd._slot_bias_hi = torch.tensor([3, 10], dtype=torch.long)
  return cmd


def test_banded_rsi_draw_stays_inside_each_slot_window():
  cmd = _banded_stub(spawn_early_fraction=0.0)
  torch.manual_seed(0)
  slots = torch.tensor([0, 1]).repeat(2048)
  frames = cmd._draw_rsi_frames(len(slots), slots)
  s0 = frames[slots == 0]
  s1 = frames[slots == 1]
  assert int(s0.min()) >= 0 and int(s0.max()) <= 3
  assert int(s1.min()) >= 4 and int(s1.max()) <= 11
  # Bias overlay lands inside each slot's own window.
  cmd = _banded_stub(spawn_early_fraction=1.0)
  frames = cmd._draw_rsi_frames(len(slots), slots)
  assert int(frames[slots == 0].min()) >= 1
  assert int(frames[slots == 0].max()) <= 2
  assert int(frames[slots == 1].min()) >= 6
  assert int(frames[slots == 1].max()) <= 9


def test_banded_clock_clamps_on_each_envs_own_end():
  cmd = _make_update_stub(MountMotionCommand, torch.tensor([False, False]))
  cmd._env_end = torch.tensor([3, 11], dtype=torch.long)
  cmd.time_steps = torch.tensor([2, 10], dtype=torch.long)
  cmd._update_command()
  assert cmd.time_steps.tolist() == [3, 11]
  cmd._update_command()
  assert cmd.time_steps.tolist() == [3, 11]  # per-env clamp, no wrap
  assert cmd.at_terminal_hold.tolist() == [True, True]


def test_banded_command_reads_local_phase():
  cmd = _banded_stub()
  cmd.time_steps = torch.tensor([0, 8], dtype=torch.long)
  cmd._env_base = torch.tensor([0, 4], dtype=torch.long)
  cmd._env_len = torch.tensor([4, 8], dtype=torch.long)
  cmd.clip = SimpleNamespace(
    ee_edge_times=torch.ones(12, 4, 2),
    ee_engaged=torch.ones(12, 4, dtype=torch.bool),
    toe_braced=torch.zeros(12, 2, dtype=torch.bool),
    fps=50.0,
  )
  out = cmd.command
  # env0 at local 0/4 -> cos=1; env1 at local 4/8 -> half cycle, cos=-1.
  assert out[0, 0].item() == pytest.approx(1.0)
  assert out[1, 0].item() == pytest.approx(-1.0)
  # Per-env clip duration, (E, 1): 4/50 s and 8/50 s.
  assert torch.allclose(
    cmd.cycle_time_s, torch.tensor([[0.08], [0.16]]), atol=1e-6
  )


def _continuous_pair() -> tuple[ClimbMotionClip, ClimbMotionClip]:
  """Mount (4 frames) + cycle (8 frames) with a continuous handoff: the
  cycle's frame 0 matches the mount's terminal joints/quat; root z differs
  (the stitcher's shift path)."""
  mount = _clip_of_frames(4)
  data = _synthetic_clip()
  frames = 10
  t = np.arange(frames, dtype=np.float32)
  joints = np.stack([0.1 * t, np.full(frames, -0.5)], axis=1)
  joints[:, 0] += float(mount.joint_pos[-1, 0])  # continuity in j0
  data["joint_positions"] = joints
  root = np.zeros((frames, 3), dtype=np.float32)
  root[:, 2] = 1.0 + 0.02 * t
  data["root_position"] = root
  data["root_quaternion"] = np.tile(
    np.array(_YAW90_XYZW, dtype=np.float32), (frames, 1)
  )
  ee = root[:, None, :] + np.array([1.0, 0.0, 0.0])
  ee = np.repeat(ee, 4, axis=1).astype(np.float32)
  data["ee_positions"] = ee
  data["ee_anchor_positions"] = ee
  engaged = np.ones((frames, 4), dtype=np.uint8)
  engaged[1:3, 2:] = 0  # feet release early in the cycle
  data["ee_engaged"] = engaged
  labels = np.zeros(frames, dtype=np.uint8)
  labels[-2:] = 1
  data["phase_segment"] = labels
  data["cycle_index"] = np.zeros(frames, dtype=np.uint16)
  cycle = ClimbMotionClip.from_dict(data, step_dt=0.02, device="cpu")
  rebuild_acyclic_schedule(mount)
  return mount, cycle


def test_stitch_mount_into_cycle_rewrites_the_seam():
  mount, cycle = _continuous_pair()
  stitched, handoff = stitch_mount_into_cycle(mount, cycle)
  assert handoff == 4 and stitched.cycle_frames == 12
  # Cycle rows: circular tables and engage bools VERBATIM.
  assert torch.equal(stitched.ee_edge_times[4:], cycle.ee_edge_times)
  assert torch.equal(stitched.ee_run_time[4:], cycle.ee_run_time)
  assert torch.equal(stitched.ee_engaged[4:], cycle.ee_engaged)
  # Continuity shift: the cycle segment is end-matched onto the prologue.
  shift = mount.root_pos[-1] - cycle.root_pos[0]
  assert torch.allclose(stitched.root_pos[4], mount.root_pos[-1])
  assert torch.allclose(
    stitched.ee_anchor_pos[4:], cycle.ee_anchor_pos + shift
  )
  # Cross-seam lookahead: at the prologue terminal the feet's next release
  # is the CYCLE's frame-1 falling edge — 2 frames out, /Tm = 0.5. The
  # plain acyclic rebuild reads the 1.0 "never" sentinel here.
  assert mount.ee_edge_times[3, 2, 0].item() == pytest.approx(1.0)
  assert stitched.ee_edge_times[3, 2, 0].item() == pytest.approx(0.5)
  # Runs fuse across the seam: the terminal all-four hold extends into the
  # cycle until its first release (2 frames total, /Tm = 0.5).
  assert stitched.ee_run_time[3, 2].item() == pytest.approx(0.5)
  # Discontinuous handoff refuses to stitch.
  bad_mount = _clip_of_frames(4)
  bad_mount.joint_pos[-1, 0] += 0.5
  with pytest.raises(ValueError):
    stitch_mount_into_cycle(bad_mount, cycle)


def test_stitch_preserves_and_checks_passive_handoff():
  mount, cycle = _continuous_pair()
  for clip in (mount, cycle):
    clip.passive_joint_names = ("left_hand_spine_gimbal_x",)
    clip.passive_joint_pos = torch.full((clip.cycle_frames, 1), -0.05)
    clip.passive_joint_vel = torch.zeros((clip.cycle_frames, 1))
  stitched, handoff = stitch_mount_into_cycle(mount, cycle)
  assert stitched.passive_joint_names == mount.passive_joint_names
  assert torch.equal(stitched.passive_joint_pos[:handoff], mount.passive_joint_pos)
  assert torch.equal(stitched.passive_joint_pos[handoff:], cycle.passive_joint_pos)
  cycle.passive_joint_pos[0, 0] = 0.0
  with pytest.raises(ValueError, match="passive joint poses"):
    stitch_mount_into_cycle(mount, cycle)


def _unified_stub(episode_mix=(0.3, 0.5, 0.2)):
  cmd = object.__new__(UnifiedMountClimbCommand)
  cmd.cfg = SimpleNamespace(episode_mix=episode_mix, spawn_early_fraction=0.0)
  cmd._env = SimpleNamespace(device="cpu")
  cmd.clip = SimpleNamespace(cycle_frames=12, step_height=0.08, fps=50.0)
  cmd._handoff_frame = 4
  cmd._cycle_frames = 8
  cmd._seam_window = (2, 6)
  return cmd


def test_unified_clock_rolls_in_and_wraps_with_step_bump():
  cmd = _unified_stub()
  cmd.time_steps = torch.tensor([3, 11], dtype=torch.long)
  cmd.root_offset = torch.zeros(2, 3)
  cmd._affordance = lambda: SimpleNamespace(
    reseat_hold=torch.tensor([False, False])
  )
  cmd._update_command()
  # Mount region rolls INTO the cycle (no clamp); the cycle end wraps back
  # to the handoff frame with one anchor-ladder bump.
  assert cmd.time_steps.tolist() == [4, 4]
  assert cmd.root_offset[0, 2].item() == pytest.approx(0.0)
  assert cmd.root_offset[1, 2].item() == pytest.approx(0.08)
  # The reseat freeze still wins.
  cmd._affordance = lambda: SimpleNamespace(
    reseat_hold=torch.tensor([True, False])
  )
  cmd._update_command()
  assert cmd.time_steps.tolist() == [4, 5]


def test_unified_command_segment_local_phase_and_duration():
  cmd = _unified_stub()
  cmd.time_steps = torch.tensor([2, 4, 8], dtype=torch.long)
  cmd.clip = SimpleNamespace(
    cycle_frames=12,
    step_height=0.08,
    fps=50.0,
    ee_edge_times=torch.ones(12, 4, 2),
    ee_engaged=torch.ones(12, 4, dtype=torch.bool),
    toe_braced=torch.zeros(12, 2, dtype=torch.bool),
  )
  out = cmd.command
  # local 2/4 -> cos(pi) = -1; cycle start 0/8 -> 1; cycle mid 4/8 -> -1.
  assert out[0, 0].item() == pytest.approx(-1.0)
  assert out[1, 0].item() == pytest.approx(1.0)
  assert out[2, 0].item() == pytest.approx(-1.0)
  assert cmd.in_cycle_region.tolist() == [False, True, True]
  assert torch.allclose(
    cmd.cycle_time_s, torch.tensor([[4 / 50], [8 / 50], [8 / 50]])
  )


def test_unified_episode_mix_draw():
  torch.manual_seed(0)
  cmd = _unified_stub(episode_mix=(0.0, 1.0, 0.0))
  frames = cmd._draw_rsi_frames(2048)
  assert int(frames.min()) >= 4 and int(frames.max()) <= 11
  cmd = _unified_stub(episode_mix=(0.0, 0.0, 1.0))
  frames = cmd._draw_rsi_frames(2048)
  assert int(frames.min()) >= 2 and int(frames.max()) <= 5
  cmd = _unified_stub(episode_mix=(1.0, 0.0, 0.0))
  frames = cmd._draw_rsi_frames(2048)
  assert int(frames.max()) == 0


def test_spawn_bias_window_curriculum(tmp_path):
  """Reverse-curriculum RSI: stage lookup + window rewrite on stage change."""
  from scr_phybot_mjab.tasks.climb_transition.mdp.curriculums import (
    spawn_bias_window,
  )

  labels = np.array([0, 0, 1, 1, 1, 2, 3, 3], dtype=np.uint8)
  data = {
    "segment_names": ["stand", "place", "set", "end_hold"],
    "phase_segment": labels,
    "cycle_index": np.zeros(8, dtype=np.uint16),
  }
  clip_path = tmp_path / "clip.npy"
  np.save(clip_path, data)

  motion = SimpleNamespace(
    cfg=SimpleNamespace(motion_file=str(clip_path)),
    clip=SimpleNamespace(cycle_frames=6),
    _bias_window=(0, 1),
  )
  stages = [
    {"iter": 0, "segments": ("set",)},
    {"iter": 100, "segments": ("place", "set")},
  ]

  def _env(iteration):
    return SimpleNamespace(
      common_step_counter=iteration * 96,
      command_manager=SimpleNamespace(get_term=lambda name: motion),
      device="cpu",
    )

  out = spawn_bias_window(_env(0), torch.arange(2), stages)
  assert out.item() == 0.0
  assert motion._bias_window == (5, 6)  # "set" = frame 5 of the 6-cycle
  # Stage 2 fires at the ceiling and widens the window backward.
  out = spawn_bias_window(_env(150), torch.arange(2), stages)
  assert out.item() == 100.0
  assert motion._bias_window == (2, 6)
  # Unresolvable segments degrade to the whole clip, never a stale window.
  spawn_bias_window(
    _env(150), torch.arange(2), [{"iter": 0, "segments": ("nope",)}]
  )
  assert motion._bias_window == (0, 6)


def test_motion_approach_progress_pays_derivative_not_rent():
  """PBRS tow: approach pays, hovering pays zero, retreat refunds; fresh
  episodes and newly-active pads re-baseline with no windfall."""
  from scr_phybot_mjab.tasks.climb_transition.mdp.rewards import (
    motion_approach_progress,
  )

  term = motion_approach_progress()
  anchor = torch.zeros(1, 2, 3)
  pad_pos = {"d": torch.tensor([[[0.50, 0.0, 0.0], [0.50, 0.0, 0.0]]])}

  motion = SimpleNamespace(
    ref_engaged=torch.zeros(1, 4, dtype=torch.bool),
    robot_engaged=torch.zeros(1, 4, dtype=torch.bool),
    ref_anchor_pos_w=torch.zeros(1, 4, 3),
    _affordance=lambda: SimpleNamespace(
      _get_executor=lambda: SimpleNamespace(
        pad_center_world=lambda: torch.cat(
          [pad_pos["d"], torch.full((1, 2, 3), 9.0)], dim=1
        )
      )
    ),
  )
  motion.ref_anchor_pos_w[:, :2] = anchor
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: motion),
    episode_length_buf=torch.tensor([10]),
  )

  def step(x: float) -> float:
    pad_pos["d"] = torch.tensor([[[x, 0.0, 0.0], [x, 0.0, 0.0]]])
    return float(term(env, command_name="climb_motion").item())

  assert step(0.50) == pytest.approx(0.0)  # first active step: baseline
  assert step(0.47) == pytest.approx(2 * 4.0 * 0.03)  # approach pays
  assert step(0.47) == pytest.approx(0.0)  # hovering pays NOTHING
  assert step(0.49) == pytest.approx(-2 * 4.0 * 0.02)  # retreat refunds
  # Teleport guard: a 30 cm jump clamps at the per-step limit.
  assert step(0.19) == pytest.approx(2 * 4.0 * 0.05)
  # Fresh episode: re-baseline, no windfall from the RSI teleport.
  env.episode_length_buf = torch.tensor([0])
  assert step(0.50) == pytest.approx(0.0)
  env.episode_length_buf = torch.tensor([10])
  assert step(0.48) == pytest.approx(2 * 4.0 * 0.02)


def test_motion_scheduled_support_loss_follows_the_schedule():
  """v19 rent abolition: the shortfall requirement is the SCHEDULED engage
  count capped at minimum_supports — zero while the clip stands (the
  inherited flat-2 bill charged the standing prefix -8/s and made dying
  optimal), exact climb parity from any wall-borne stance."""
  from scr_phybot_mjab.tasks.climb_transition.mdp.rewards import (
    motion_scheduled_support_loss,
  )

  ref = torch.tensor(
    [
      [0, 0, 0, 0],  # standing prefix: nothing scheduled
      [1, 0, 0, 0],  # first hand placed
      [1, 1, 0, 0],  # both hands scheduled
      [1, 1, 1, 1],  # terminal hold (climb parity)
      [0, 0, 0, 0],  # standing but robot latched anyway
      [1, 1, 1, 1],  # hanging, robot dropped to one anchor
    ],
    dtype=torch.bool,
  )
  eng = torch.tensor(
    [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [1, 1, 1, 1],
      [1, 1, 0, 0],
      [1, 0, 0, 0],
    ],
    dtype=torch.bool,
  )
  motion = SimpleNamespace(ref_engaged=ref)
  grip = SimpleNamespace(func=SimpleNamespace(engaged=eng))
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: motion),
    event_manager=SimpleNamespace(get_term_cfg=lambda name: grip),
  )
  out = motion_scheduled_support_loss(env, grace_s=0.0)
  # Standing free = NO rent; behind schedule bills the scheduled count
  # (capped at 2); extra engagement is never rewarded, only un-billed.
  assert out.tolist() == [0.0, 1.0, 2.0, 0.0, 0.0, 1.0]
  # v51 grace ramp: a NEW shortfall bills 0 at first and ramps linearly
  # over grace_s — the deadline cliff funded the slam-weld (stopping the
  # instant -8/s beat attempting a gentle weld while bleeding).
  env.episode_length_buf = torch.full((6,), 10)
  env.step_dt = 0.02
  first = motion_scheduled_support_loss(env, grace_s=1.0)
  assert first.tolist() == pytest.approx([0.0] * 6)  # ramp starts at 0
  ramped = motion_scheduled_support_loss(env, grace_s=1.0)
  # One step later: ramp = 0.02/1.0 of the raw shortfall.
  assert ramped[2].item() == pytest.approx(2.0 * 0.02)
  assert ramped[0].item() == pytest.approx(0.0)


def test_unscheduled_press_bill_all_press_flag():
  """v24: at the clamped stand clock the regrip-imminent exemption is
  permanently true — ``bill_all_press`` closes it so every un-latched
  press bills; latched pads stay exempt on both paths."""
  from scr_phybot_mjab.tasks.climb_transition.mdp.rewards import (
    motion_unscheduled_press_penalty,
  )

  motion = SimpleNamespace(
    ref_engaged=torch.zeros(1, 4, dtype=torch.bool),
    robot_engaged=torch.tensor([[False, True, False, False]]),
    ref_edge_countdowns=torch.full((1, 4, 2), 0.001),  # edge "imminent"
    cycle_time_s=15.6,
    _affordance=lambda: SimpleNamespace(
      pad_press_force=torch.tensor([[2.0, 2.0, 8.0, 0.0]])
    ),
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: motion)
  )
  # Window semantics: the imminent rising edge exempts every press.
  assert motion_unscheduled_press_penalty(env).item() == pytest.approx(0.0)
  # bill_all_press: LH 2/4 + LF capped 1.0 bill; latched RH stays free.
  out = motion_unscheduled_press_penalty(env, bill_all_press=True)
  assert out.item() == pytest.approx(1.5)


def test_force_target_scale_pays_by_closeness():
  """v45 doctrine: force rewards pay by closeness to the target, never
  "more is better"; the floor keeps a far-side learning signal."""
  from scr_phybot_mjab.tasks.climb_transition.mdp.rewards import (
    force_target_scale,
  )

  f = torch.tensor([0.0, 2.5, 5.0, 27.5, 50.0, 80.0])
  s = force_target_scale(f, target_n=5.0, zero_at_n=50.0, floor=0.25)
  assert s.tolist() == pytest.approx([0.25, 0.5, 1.0, 0.5, 0.25, 0.25])
  s0 = force_target_scale(f, target_n=5.0, zero_at_n=50.0, floor=0.0)
  assert s0[0].item() == pytest.approx(0.0)
  assert s0[-1].item() == pytest.approx(0.0)


def test_gentle_press_pay_and_tow_edge_gate():
  """v28: the wage pays [0, cutoff] only (slams earn zero), and the masked
  tow pays nothing outside the scheduled-edge lead window even at zero
  distance."""
  from scr_phybot_mjab.tasks.climb_transition.mdp.rewards import (
    motion_scheduled_press_gentle,
    motion_swing_approach_masked,
  )

  motion = SimpleNamespace(
    ref_engaged=torch.zeros(1, 4, dtype=torch.bool),
    robot_engaged=torch.zeros(1, 4, dtype=torch.bool),
    ref_edge_countdowns=torch.full((1, 4, 2), 0.001),  # edges imminent
    cycle_time_s=15.0,
    ref_anchor_pos_w=torch.zeros(1, 4, 3),
    _affordance=lambda: SimpleNamespace(
      pad_press_force=torch.tensor([[3.0, 70.0, 0.0, 0.0]]),
      _get_executor=lambda: SimpleNamespace(
        pad_center_world=lambda: torch.zeros(1, 4, 3)  # AT the anchors
      ),
    ),
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: motion)
  )
  # Triangle wage (v34; v38 moves the peak to the WELD-capable 5 N — a
  # flat 3 N split four ways can never cross the 3 N per-sphere gate).
  pay = motion_scheduled_press_gentle(env, press_peak_n=5.0)
  assert pay.item() == pytest.approx(0.6)  # 3 N = 3/5 up the left slope
  motion._affordance = lambda: SimpleNamespace(
    pad_press_force=torch.tensor([[5.0, 0.0, 0.0, 0.0]])
  )
  assert motion_scheduled_press_gentle(
    env, press_peak_n=5.0
  ).item() == pytest.approx(1.0)
  motion._affordance = lambda: SimpleNamespace(
    pad_press_force=torch.tensor([[3.0, 70.0, 0.0, 0.0]]),
    _get_executor=lambda: SimpleNamespace(
      pad_center_world=lambda: torch.zeros(1, 4, 3)
    ),
  )
  # v35 spatial anchor: the same press pays ZERO away from the anchor
  # (the hip-height wall-slap wage farm); at the anchor it pays by the
  # v47 default peak (10 N weldable band), so 3 N = 0.3.
  pay = motion_scheduled_press_gentle(env, anchor_radius_m=0.15)
  assert pay.item() == pytest.approx(0.375)  # pads AT anchors: 3/8 rise
  motion.ref_anchor_pos_w = torch.full((1, 4, 3), 0.5)
  pay = motion_scheduled_press_gentle(env, anchor_radius_m=0.15)
  assert pay.item() == pytest.approx(0.0)
  motion.ref_anchor_pos_w = torch.zeros(1, 4, 3)
  # Tow at zero distance, edges imminent, presses below cutoff -> pays
  # for the un-pressed EEs; the 70 N EE is cut off by the press gate.
  tow = motion_swing_approach_masked(
    env, ee_indices=(0, 1, 2, 3), edge_lead_time_s=2.0
  )
  assert tow.item() == pytest.approx(3.0)  # EEs 0,2,3 (EE1 pressing 70 N)
  # Edges far away -> the same zero-distance hover earns NOTHING.
  motion.ref_edge_countdowns = torch.full((1, 4, 2), 0.5)
  tow = motion_swing_approach_masked(
    env, ee_indices=(0, 1, 2, 3), edge_lead_time_s=2.0
  )
  assert tow.item() == pytest.approx(0.0)


def test_rsi_uniform_half_restricted_to_disengaged_frames():
  """spawn_disengaged_only: the uniform half draws from the standing pool
  only; the bias-window overlay is unaffected."""
  cmd = object.__new__(MountMotionCommand)
  cmd.cfg = SimpleNamespace(spawn_early_fraction=0.0)
  cmd._env = SimpleNamespace(device="cpu")
  cmd.clip = SimpleNamespace(cycle_frames=10)
  cmd._bias_window = (0, 3)
  cmd._uniform_frames = torch.tensor([0, 1, 2, 3, 7], dtype=torch.long)
  torch.manual_seed(0)
  frames = cmd._draw_rsi_frames(2048)
  assert set(frames.tolist()) <= {0, 1, 2, 3, 7}
  # The class default keeps the plain full-clip uniform path.
  assert MountMotionCommand._uniform_frames is None


def test_resolve_segment_window(tmp_path):
  labels = np.array([0, 0, 1, 1, 1, 2, 3, 3], dtype=np.uint8)
  data = {
    "segment_names": ["stand", "place", "set", "end_hold"],
    "phase_segment": labels,
    "cycle_index": np.zeros(8, dtype=np.uint16),
  }
  path = tmp_path / "clip.npy"
  np.save(path, data)
  # Cycle = 6 frames (end_hold tail excluded). place+set span frames 2-5.
  lo, hi = resolve_segment_window(str(path), ("place", "set"))
  assert lo == pytest.approx(2 / 6) and hi == pytest.approx(6 / 6)
  # Unknown segment or missing file degrade to None (caller falls back).
  assert resolve_segment_window(str(path), ("nope",)) is None
  assert resolve_segment_window(str(tmp_path / "absent.npy"), ("place",)) is None
