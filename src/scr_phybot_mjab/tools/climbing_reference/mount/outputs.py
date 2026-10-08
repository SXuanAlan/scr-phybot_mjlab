"""The finished mount clip: smoothing, canonical-frame arrays and provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_POLICY_JOINT_NAMES, PHYBOT_C2_XML
from scr_phybot_mjab.tools.climbing_reference import kinematics as kin

from .builder import SEGMENT_NAMES, MountBuilder


def smooth(b: MountBuilder) -> None:
  """Zero-phase Gaussian smoothing of the solved frames up to the handoff.

  The per-frame solve leaves one-frame steps where tasks hand over. The
  handoff and end frames are exact copies of the climb cycle's first frame
  and pad the right edge, so the closure still lands on them; the audit runs
  on the smoothed frames, so any drift the blur costs is measured.
  """
  recipe = b.recipe
  if recipe.smooth_sigma <= 0.0:
    return
  n_play = len(b.qpos_frames) - b.seg_frames["handoff_hold"] - b.seg_frames["end_hold"]
  radius = max(int(round(3.0 * recipe.smooth_sigma)), 1)
  kern = np.exp(-0.5 * (np.arange(-radius, radius + 1) / recipe.smooth_sigma) ** 2)
  kern /= kern.sum()
  arr = np.stack(b.qpos_frames)
  left = np.repeat(arr[:1], radius, axis=0)
  right = np.repeat(arr[n_play : n_play + 1], radius, axis=0)
  padded = np.concatenate([left, arr[:n_play], right], axis=0)
  sm = np.empty_like(arr[:n_play])
  for col in range(arr.shape[1]):
    sm[:, col] = np.convolve(padded[:, col], kern, mode="valid")
  sm[:, 3:7] /= np.linalg.norm(sm[:, 3:7], axis=1, keepdims=True)
  for i in range(n_play):
    b.qpos_frames[i] = sm[i]


def clip_data(b: MountBuilder, handoff_motion: Path) -> dict:
  """The clip dictionary: the cycle clip's schema plus ``foot_on_ground``.

  Every world-frame array is stored in the canonical vertical-wall frame
  (de-rotated about the base line), so the runtime re-poses the clip at
  the sampled wall angle through its wall-frame machinery.
  """
  ik, model, data, wall = b.ik, b.model, b.data, b.wall
  frames = len(b.qpos_frames)
  joint_positions = np.zeros((frames, len(PHYBOT_C2_POLICY_JOINT_NAMES)), np.float32)
  root_position = np.zeros((frames, 3), np.float32)
  root_quaternion = np.zeros((frames, 4), np.float32)
  ee_positions = np.zeros((frames, 4, 3), np.float32)
  ee_quaternions = np.zeros((frames, 4, 4), np.float32)
  pad_positions = np.zeros((frames, 4, 3), np.float32)
  foot_positions = np.zeros((frames, 2, 3), np.float32)
  for t, q in enumerate(b.qpos_frames):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    joint_positions[t] = [data.qpos[ik.qpos_addresses[nm]] for nm in PHYBOT_C2_POLICY_JOINT_NAMES]
    root_position[t] = q[0:3]
    root_quaternion[t] = (q[4], q[5], q[6], q[3])
    for i, ee in enumerate(ik.ee_names):
      bid = ik.ee_bid[ee]
      ee_positions[t, i] = data.xpos[bid]
      xq = data.xquat[bid]
      ee_quaternions[t, i] = (xq[1], xq[2], xq[3], xq[0])
      pad_positions[t, i] = data.site_xpos[ik.pad_sid[ee]]
    for i, sid in enumerate(ik.foot_sid):
      foot_positions[t, i] = data.site_xpos[sid]

  out = {
    "joints_list": list(PHYBOT_C2_POLICY_JOINT_NAMES),
    "joint_positions": joint_positions,
    "root_position": wall.derotate_points(root_position),
    "root_quaternion": wall.derotate_quats_xyzw(root_quaternion),
    "foot_sites_list": list(kin.FOOT_SITE_NAMES),
    "foot_positions": wall.derotate_points(foot_positions),
    "fps": np.float32(b.recipe.fps),
    "ee_list": list(ik.ee_names),
    "ee_positions": wall.derotate_points(ee_positions),
    "ee_quaternions": wall.derotate_quats_xyzw(ee_quaternions),
    "pad_positions": wall.derotate_points(pad_positions),
    "ee_engaged": np.stack(b.engaged_frames),
    "ee_anchor_positions": wall.derotate_points(np.stack(b.anchor_frames).astype(np.float32)),
    "toe_braced": np.zeros((frames, 2), dtype=np.uint8),
    "toe_list": list(ik.toes.keys()),
    "foot_on_ground": np.stack(b.ground_frames),
    "phase_segment": np.asarray(b.labels, dtype=np.uint8),
    "segment_names": list(SEGMENT_NAMES),
    "cycle_index": np.zeros(frames, dtype=np.uint16),
    "step_height": np.float32(0.0),
    "wall_surface_x": np.float32(kin.WALL_SURFACE_X),
    "pad_touch_x": np.float32(kin.PAD_TOUCH_X),
    # The native wall angle; the frame data above is canonical.
    "wall_angle": np.float32(wall.angle_deg),
  }
  # The passive gimbal coordinates are reference geometry and RSI state,
  # never policy actuator targets.
  passive_names = tuple(name for name in ik.qpos_addresses if "_gimbal_" in name)
  out["passive_joint_names"] = list(passive_names)
  out["passive_joint_positions"] = np.stack(b.qpos_frames)[
    :, [ik.qpos_addresses[name] for name in passive_names]
  ].astype(np.float32)
  out["pad_reference_point"] = "C"
  out["generator"] = "scripts/gen_mount_clip.py"
  out["handoff_motion"] = handoff_motion.name
  out["handoff_motion_sha256"] = hashlib.sha256(handoff_motion.read_bytes()).hexdigest()
  out["asset_xml_sha256"] = hashlib.sha256(PHYBOT_C2_XML.read_bytes()).hexdigest()
  return out
