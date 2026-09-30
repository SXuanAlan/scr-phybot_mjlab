"""Validate and replay AMP motion clips on the C2 MuJoCo model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))

from amp_rsl_rl.utils import AMPLoader, VelocityRepresentation  # noqa: E402
import scr_phybot_mjab.tasks  # noqa: E402,F401
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (  # noqa: E402
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_XML,
)

_EXPECTED_AMP_DIM = 48


def _fps_value(data: dict) -> float:
  fps = data["fps"]
  if isinstance(fps, np.ndarray):
    return float(fps.reshape(-1)[0])
  return float(fps)


def _load_clip(path: Path) -> dict:
  data = np.load(path, allow_pickle=True).item()
  required = {"joints_list", "joint_positions", "root_position", "root_quaternion", "fps"}
  missing = required - set(data)
  if missing:
    raise ValueError(f"{path.name}: missing keys {sorted(missing)}")
  return data


def _validate_clip(path: Path, expected_joints: list[str]) -> tuple[str, int]:
  data = _load_clip(path)
  name = path.stem
  joints = list(data["joints_list"])
  if joints != expected_joints:
    raise ValueError(f"{path.name}: joint order mismatch\nexpected={expected_joints}\nactual={joints}")

  joint_pos = np.asarray(data["joint_positions"], dtype=np.float32)
  root_pos = np.asarray(data["root_position"], dtype=np.float32)
  root_quat = np.asarray(data["root_quaternion"], dtype=np.float32)
  fps = _fps_value(data)

  frames = joint_pos.shape[0]
  if joint_pos.shape != (frames, len(expected_joints)):
    raise ValueError(f"{path.name}: joint_positions shape {joint_pos.shape} is invalid")
  if root_pos.shape != (frames, 3):
    raise ValueError(f"{path.name}: root_position shape {root_pos.shape} is invalid")
  if root_quat.shape != (frames, 4):
    raise ValueError(f"{path.name}: root_quaternion shape {root_quat.shape} is invalid")
  if frames < 2:
    raise ValueError(f"{path.name}: needs at least two frames")
  if not np.isfinite(joint_pos).all() or not np.isfinite(root_pos).all() or not np.isfinite(root_quat).all():
    raise ValueError(f"{path.name}: contains NaN or Inf")

  quat_norm_err = float(np.max(np.abs(np.linalg.norm(root_quat, axis=1) - 1.0)))
  if quat_norm_err > 1.0e-3:
    raise ValueError(f"{path.name}: quaternion norm error too large ({quat_norm_err:.3e})")

  dt = 1.0 / fps
  xy_speed = np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1) / dt
  disp_xy = root_pos[-1, :2] - root_pos[0, :2]
  print(
    f"{name:22s} frames={frames:4d} fps={fps:5.1f} dur={frames / fps:6.2f}s "
    f"disp_xy=({disp_xy[0]: .3f},{disp_xy[1]: .3f}) "
    f"speed_mean={xy_speed.mean():.3f} speed_p95={np.quantile(xy_speed, 0.95):.3f} "
    f"root_z=({root_pos[:, 2].min():.3f},{root_pos[:, 2].max():.3f}) "
    f"joint_abs_max={np.abs(joint_pos).max():.3f}"
  )
  return name, frames


def _set_model_frame(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  clip: dict,
  frame_id: int,
  joint_qpos_addr: dict[str, int],
) -> None:
  mujoco.mj_resetData(model, data)
  data.qpos[:] = model.qpos0
  data.qvel[:] = 0.0

  root_pos = np.asarray(clip["root_position"], dtype=np.float32)[frame_id]
  quat_xyzw = np.asarray(clip["root_quaternion"], dtype=np.float32)[frame_id]
  data.qpos[:3] = root_pos
  data.qpos[3:7] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

  joint_positions = np.asarray(clip["joint_positions"], dtype=np.float32)[frame_id]
  for joint_name, value in zip(clip["joints_list"], joint_positions):
    data.qpos[joint_qpos_addr[joint_name]] = float(value)

  mujoco.mj_forward(model, data)


def _render_clip(
  path: Path,
  output_dir: Path,
  width: int,
  height: int,
  stride: int,
  max_render_frames: int,
) -> Path:
  clip = _load_clip(path)
  model = mujoco.MjModel.from_xml_path(str(PHYBOT_C2_XML))
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=height, width=width)
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.distance = 3.0
  camera.azimuth = 90.0
  camera.elevation = -12.0

  joint_qpos_addr = {}
  for joint_name in PHYBOT_C2_POLICY_JOINT_NAMES:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
      raise ValueError(f"C2 XML missing joint {joint_name}")
    joint_qpos_addr[joint_name] = int(model.jnt_qposadr[joint_id])

  root_pos = np.asarray(clip["root_position"], dtype=np.float32)
  frame_ids = list(range(0, len(root_pos), max(1, stride)))
  if max_render_frames > 0:
    frame_ids = frame_ids[:max_render_frames]

  frames = []
  for frame_id in frame_ids:
    _set_model_frame(model, data, clip, frame_id, joint_qpos_addr)
    camera.lookat[:] = data.qpos[:3]
    renderer.update_scene(data, camera=camera)
    frames.append(renderer.render())

  output_dir.mkdir(parents=True, exist_ok=True)
  out = output_dir / f"{path.stem}.mp4"
  fps = max(1, int(round(_fps_value(clip) / max(1, stride))))
  imageio.mimsave(out, frames, fps=fps)
  renderer.close()
  return out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--data-dir",
    type=Path,
    default=_REPO / "src/scr_phybot_mjab/assets/motions",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=_REPO / "logs/amp_dataset_replay",
  )
  parser.add_argument("--no-render", action="store_true")
  parser.add_argument("--width", type=int, default=960)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--render-stride", type=int, default=2)
  parser.add_argument("--max-render-frames", type=int, default=500)
  args = parser.parse_args()

  expected_joints = list(PHYBOT_C2_POLICY_JOINT_NAMES)
  paths = sorted(args.data_dir.glob("*.npy"))
  if not paths:
    raise FileNotFoundError(f"No .npy clips found under {args.data_dir}")

  print(f"AMP dataset: {args.data_dir}")
  print(f"clips: {len(paths)}")
  for path in paths:
    _validate_clip(path, expected_joints)

  datasets = {path.stem: 1.0 for path in paths}
  loader = AMPLoader(
    device="cpu",
    dataset_path_root=args.data_dir,
    datasets=datasets,
    simulation_dt=0.02,
    slow_down_factor=1,
    expected_joint_names=expected_joints,
    velocity_representation=VelocityRepresentation.BODY_FIXED_REPRESENTATION,
  )
  print(f"loader.all_obs shape: {tuple(loader.all_obs.shape)}")
  if loader.all_obs.shape[1] != _EXPECTED_AMP_DIM:
    raise ValueError(f"AMP obs dim mismatch: got {loader.all_obs.shape[1]}, expected {_EXPECTED_AMP_DIM}")

  if args.no_render:
    return

  print(f"rendering replay videos to {args.output_dir}")
  for path in paths:
    out = _render_clip(
      path=path,
      output_dir=args.output_dir,
      width=args.width,
      height=args.height,
      stride=args.render_stride,
      max_render_frames=args.max_render_frames,
    )
    print(f"  wrote {out}")


if __name__ == "__main__":
  main()
