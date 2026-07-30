"""Render a climb motion clip to an MP4 with the wall in the scene.

Kinematically replays a climb clip in the AMP/motion schema (see
``gen_climb_clip.py``) on the C2 model, adds the climb wall and a ground plane
for context, and writes the frames offscreen (EGL) to a video. No physics is
stepped.

Usage:
    python scripts/render_climb_video.py
    python scripts/render_climb_video.py <clip.npy> --output <out.mp4>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mjlab  # noqa: F401  # Initialize task entry points before PHYBot imports.
import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (  # noqa: E402
  get_spec,
)
from scr_phybot_mjab.assets.wall.wall_constants import (  # noqa: E402
  WALL_HALF_SIZE,
  WALL_OFFSET,
)

_DEFAULT_CLIP = (
  _REPO
  / "src/scr_phybot_mjab/assets/motions/climb_cycle_moment_safe.npy"
)


def _build_model(width: int, height: int) -> mujoco.MjModel:
  spec = get_spec()
  spec.visual.global_.offwidth = width
  spec.visual.global_.offheight = height
  spec.worldbody.add_geom(
    name="viz_ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[10.0, 10.0, 0.1],
    rgba=[0.35, 0.4, 0.45, 1.0],
    contype=0,
    conaffinity=0,
  )
  spec.worldbody.add_geom(
    name="viz_wall",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=WALL_HALF_SIZE,
    pos=list(WALL_OFFSET),
    rgba=[0.72, 0.68, 0.6, 1.0],
    contype=0,
    conaffinity=0,
  )
  # Checkered plane just off the wall face so climb progress is visible.
  spec.add_texture(
    name="viz_wall_tex",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=[0.72, 0.68, 0.6],
    rgb2=[0.63, 0.59, 0.52],
    width=300,
    height=300,
  )
  mat = spec.add_material(
    name="viz_wall_mat", texrepeat=[8.0, 8.0], texuniform=True
  )
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "viz_wall_tex"
  spec.worldbody.add_geom(
    name="viz_wall_face",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[WALL_HALF_SIZE[2], WALL_HALF_SIZE[1], 0.1],
    pos=[WALL_OFFSET[0] - WALL_HALF_SIZE[0] - 0.001, WALL_OFFSET[1], WALL_OFFSET[2]],
    quat=[0.7071067811865476, 0.0, -0.7071067811865476, 0.0],
    material="viz_wall_mat",
    contype=0,
    conaffinity=0,
  )
  spec.worldbody.add_light(
    name="viz_light",
    pos=[-1.5, -1.0, 3.0],
    dir=[0.6, 0.3, -0.75],
    castshadow=True,
  )
  return spec.compile()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("motion_file", type=Path, nargs="?", default=_DEFAULT_CLIP)
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="output video path (default: <clip dir>/<clip stem>.mp4)",
  )
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--azimuth", type=float, default=55.0)
  parser.add_argument("--elevation", type=float, default=-12.0)
  parser.add_argument("--distance", type=float, default=2.2)
  parser.add_argument(
    "--grip-markers",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="draw green (latched) / orange (released) markers at the four pads",
  )
  parser.add_argument(
    "--frame",
    type=int,
    default=None,
    help="render this single frame to a PNG instead of the full video",
  )
  args = parser.parse_args()

  clip = np.load(args.motion_file, allow_pickle=True).item()
  joints = list(clip["joints_list"])
  joint_pos = np.asarray(clip["joint_positions"], dtype=np.float64)
  root_pos = np.asarray(clip["root_position"], dtype=np.float64)
  root_quat_xyzw = np.asarray(clip["root_quaternion"], dtype=np.float64)
  pad_positions = np.asarray(clip.get("pad_positions"), dtype=np.float64)
  ee_engaged = np.asarray(clip.get("ee_engaged"), dtype=bool)
  ee_names = list(clip.get("ee_list", ()))
  fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
  if args.grip_markers and (
    pad_positions.shape[:2] != ee_engaged.shape
    or pad_positions.shape[-1] != 3
  ):
    raise ValueError(
      "--grip-markers requires pad_positions (T,H,3) and ee_engaged (T,H)"
    )

  model = _build_model(args.width, args.height)
  data = mujoco.MjData(model)
  grip_visual_geoms: list[int] = []
  if args.grip_markers:
    for ee_name in ee_names:
      body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, ee_name
      )
      geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
      visual = [
        int(geom_id)
        for geom_id in geom_ids
        if (mujoco.mj_id2name(
          model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
        ) or "").endswith("_visual")
      ]
      if len(visual) != 1:
        raise ValueError(
          f"Expected one visual geom on pad body '{ee_name}', got {visual}."
        )
      grip_visual_geoms.append(visual[0])
  qadr = {
    n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    for n in joints
  }

  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.distance = args.distance
  camera.azimuth = args.azimuth
  camera.elevation = args.elevation

  def set_frame(t: int) -> None:
    data.qpos[:] = 0.0
    data.qpos[0:3] = root_pos[t]
    data.qpos[3:7] = root_quat_xyzw[t, [3, 0, 1, 2]]  # xyzw -> wxyz
    for i, name in enumerate(joints):
      data.qpos[qadr[name]] = joint_pos[t, i]
    if args.grip_markers:
      for geom_id, latched in zip(
        grip_visual_geoms, ee_engaged[t], strict=True
      ):
        model.geom_rgba[geom_id] = (
          (0.1, 0.9, 0.25, 1.0)
          if latched
          else (1.0, 0.25, 0.05, 1.0)
        )
    mujoco.mj_forward(model, data)

  renderer = mujoco.Renderer(model, height=args.height, width=args.width)
  # Fixed x/y viewpoint; only track the climb height so the wall stays put.
  lookat_xy = root_pos[:, :2].mean(axis=0)

  def render(t: int) -> np.ndarray:
    set_frame(t)
    camera.lookat[:] = (*lookat_xy, root_pos[t, 2])
    renderer.update_scene(data, camera=camera)
    if args.grip_markers:
      scene = renderer.scene
      identity = np.eye(3).reshape(-1)
      for pad_index, (pad_pos, latched) in enumerate(
        zip(pad_positions[t], ee_engaged[t], strict=True)
      ):
        if scene.ngeom >= scene.maxgeom:
          raise RuntimeError("MuJoCo render scene has no room for grip markers.")
        rgba = (
          np.array([0.1, 0.9, 0.25, 0.95])
          if latched
          else np.array([1.0, 0.25, 0.05, 0.95])
        )
        # Pull the marker toward the robot/camera and just outside the limb so
        # the pad mesh cannot hide it against the wall.
        lateral = 0.045 if pad_index in (0, 2) else -0.045
        marker_pos = pad_pos + np.array([-0.045, lateral, 0.0])
        mujoco.mjv_initGeom(
          scene.geoms[scene.ngeom],
          mujoco.mjtGeom.mjGEOM_SPHERE,
          np.full(3, 0.020),
          marker_pos,
          identity,
          rgba,
        )
        scene.ngeom += 1
    return renderer.render()

  if args.frame is not None:
    out = args.output or args.motion_file.with_suffix(".png")
    imageio.imwrite(out, render(args.frame))
    print(f"wrote {out} (frame {args.frame})")
  else:
    out = args.output or args.motion_file.with_suffix(".mp4")
    frames = [render(t) for t in range(len(root_pos))]
    imageio.mimsave(out, frames, fps=int(round(fps)))
    print(
      f"wrote {out} ({len(frames)} frames, {len(frames) / fps:.2f}s @ {fps:.0f} fps)"
    )
  renderer.close()


if __name__ == "__main__":
  main()
