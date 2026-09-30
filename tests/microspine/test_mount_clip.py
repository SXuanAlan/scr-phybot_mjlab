"""Handoff contract of the a80 mount prologue clip (gen_mount_clip.py).

The mount ends where the climb cycle begins: the committed prologue must land
on the climb clip's frame 0 modulo a rigid vertical shift with all four pads
engaged, and the motion loader must accept its acyclic schema.
"""

from pathlib import Path

import numpy as np
import pytest

from scr_phybot_mjab.tasks.climb_transition80.mdp.motion import rebuild_acyclic_schedule
from scr_phybot_mjab.tasks.climb_wall.mdp.motion import ClimbMotionClip

_ASSETS = Path(__file__).resolve().parents[2] / "src/scr_phybot_mjab/assets/motions"
_MOUNT = "climb_mount_prologue_a80.npy"


@pytest.fixture(scope="module")
def mount() -> dict:
  return np.load(_ASSETS / _MOUNT, allow_pickle=True).item()


@pytest.fixture(scope="module")
def cycle(mount: dict) -> dict:
  return np.load(_ASSETS / mount["handoff_motion"], allow_pickle=True).item()


def test_mount_wall_angle_metadata(mount: dict) -> None:
  """The clip is stored in the canonical frame; its native wall angle is metadata."""
  assert float(mount["wall_angle"]) == 80.0


def test_handoff_frame_matches_climb_state1(mount: dict, cycle: dict) -> None:
  assert list(mount["joints_list"]) == list(cycle["joints_list"])
  assert list(mount["ee_list"]) == list(cycle["ee_list"])

  # Joint-space pose: the final mount frame IS state 1.
  dq = np.abs(
    np.asarray(mount["joint_positions"][-1])
    - np.asarray(cycle["joint_positions"][0])
  )
  assert float(dq.max()) < 1.0e-4

  if mount.get("passive_joint_names"):
    assert tuple(mount["passive_joint_names"]) == tuple(cycle["passive_joint_names"])
    assert np.allclose(
      mount["passive_joint_positions"][-1],
      cycle["passive_joint_positions"][0],
      atol=1.0e-5,
    )

  # Base orientation identical; base position identical in x/y and rigidly
  # shifted in z (the mount ends dz below the canonical stance so the final
  # stance clears the floor from standing reach).
  assert np.allclose(
    mount["root_quaternion"][-1], cycle["root_quaternion"][0], atol=1.0e-5
  )
  dp = np.asarray(mount["root_position"][-1]) - np.asarray(
    cycle["root_position"][0]
  )
  assert abs(dp[0]) < 1.0e-4 and abs(dp[1]) < 1.0e-4
  dz = float(dp[2])
  assert dz < 0.0  # standing reach puts the mount stance below canonical

  # Base-relative pad positions match to the millimeter; world anchors are
  # the cycle's shifted by exactly dz.
  rel_mount = np.asarray(mount["pad_positions"][-1]) - np.asarray(
    mount["root_position"][-1]
  )
  rel_cycle = np.asarray(cycle["pad_positions"][0]) - np.asarray(
    cycle["root_position"][0]
  )
  assert float(np.abs(rel_mount - rel_cycle).max()) < 1.5e-3
  anchor_shift = np.asarray(mount["ee_anchor_positions"][-1]) - np.asarray(
    cycle["ee_anchor_positions"][0]
  )
  assert float(np.abs(anchor_shift[:, :2]).max()) < 1.5e-3
  assert float(np.abs(anchor_shift[:, 2] - dz).max()) < 1.5e-3

  # All four pads engaged on both sides of the seam.
  assert np.asarray(mount["ee_engaged"][-1]).all()
  assert np.asarray(cycle["ee_engaged"][0]).all()


def test_mount_schedule_shape(mount: dict) -> None:
  engaged = np.asarray(mount["ee_engaged"], dtype=bool)
  ground = np.asarray(mount["foot_on_ground"], dtype=bool)
  assert ground.shape == (engaged.shape[0], 2)

  # Starts standing (nothing engaged, both feet on the floor), ends attached
  # (all engaged, nothing on the floor).
  assert not engaged[0].any() and ground[0].all()
  assert engaged[-1].all() and not ground[-1].any()

  # Engagement is monotone per EE: the mount never schedules a release
  # (hand walk-up regrips excepted — those show as OFF runs BETWEEN two ON
  # runs for the hands, so allow at most one release per hand and none for
  # the feet).
  for i, ee in enumerate(mount["ee_list"]):
    col = engaged[:, i].astype(np.int8)
    drops = int(np.count_nonzero(np.diff(col) < 0))
    # v6g: the left hand's TWO-step walk-up regrips twice.
    assert drops <= (2 if "hand" in str(ee) else 0), (ee, drops)

  # The floor schedule only ever loses contact.
  for i in range(2):
    assert int(np.count_nonzero(np.diff(ground[:, i].astype(np.int8)) > 0)) == 0

  # Acyclic schema markers.
  assert float(mount["step_height"]) == 0.0
  # The clip ENDS in end_hold (the loaders look the segment up by name; the
  # name table itself may list later-added segments after it).
  names = list(mount["segment_names"])
  assert names[int(np.asarray(mount["phase_segment"])[-1])] == "end_hold"
  assert not np.asarray(mount["toe_braced"]).any()


def test_loader_accepts_mount_clip(mount: dict) -> None:
  clip = ClimbMotionClip.from_dict(mount, step_dt=0.02, device="cpu")
  # Everything before end_hold reads as the "cycle"; the tail feeds the
  # finite-difference velocities.
  labels = np.asarray(mount["phase_segment"])
  end_id = list(mount["segment_names"]).index("end_hold")
  expected_cycle = int(np.count_nonzero(labels != end_id))
  assert clip.cycle_frames == expected_cycle
  assert bool(clip.ee_engaged[-1].all())
  assert clip.step_height == 0.0


def test_terminal_hold_billing_never_rides_the_edge_grace(mount: dict) -> None:
  """Anti-regression for the terminal-clamp exploit.

  The transition task clamps the clock on the LAST playable frame. If any
  EE's final engage edge sits inside the grace band there (edge closer than
  0.3 x its run), letting go at the handoff stance is billed near zero and
  the command's release countdown reads "imminent" — the handoff_hold
  segment exists to push every edge a full run away from the clamp.
  """
  # The playable region must END with a long all-engaged run.
  labels = np.asarray(mount["phase_segment"])
  end_id = list(mount["segment_names"]).index("end_hold")
  engaged = np.asarray(mount["ee_engaged"], dtype=bool)[labels != end_id]
  trailing = 0
  while trailing < len(engaged) and engaged[-1 - trailing].all():
    trailing += 1
  assert trailing >= 15, trailing

  clip = ClimbMotionClip.from_dict(mount, step_dt=0.02, device="cpu")
  rebuild_acyclic_schedule(clip)
  # Acyclic tables: no future release exists at the clamp — every EE's
  # release lane reads the "never" sentinel, not a phantom wrap edge.
  assert bool((clip.ee_edge_times[-1][:, 0] >= 1.0).all())
  # Mismatch grace fully expired at the clamp: edge distance exceeds
  # 0.3 x run for all EEs, so the bill multiplier is 1.0.
  bill = (clip.ee_edge_distance[-1] / (clip.ee_run_time[-1] * 0.3)).clamp(max=1.0)
  assert bool((bill >= 1.0).all())
  # And no phantom edge at frame 0 either (standing start, nothing near).
  assert float(clip.ee_edge_distance[0].min()) > 0.0
