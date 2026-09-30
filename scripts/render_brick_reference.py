"""Render a PNG preview or MP4 of the kinematic reference, not dynamic policy replay."""

import argparse
from pathlib import Path

import imageio.v2 as iio
import mjlab  # noqa: F401
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from scr_phybot_mjab.assets.brick import BrickCfg
from scr_phybot_mjab.tasks.brick.reference import PHASE_NAMES, REFERENCE_PATH, BrickIK, load_reference


def render_frame(ik, renderer, camera, ref, phase, progress):
  knot = np.clip(progress, 0.0, 1.0) * (ref["joint_pos"].shape[1] - 1)
  lower = int(np.floor(knot))
  upper = min(lower + 1, ref["joint_pos"].shape[1] - 1)
  alpha = knot - lower

  def blend(key):
    return (1.0 - alpha) * ref[key][phase, lower] + alpha * ref[key][phase, upper]

  root = blend("root_pose")
  # Quaternions q and -q describe the same pose; interpolate on the short arc.
  q0, q1 = ref["root_pose"][phase, [lower, upper], 3:]
  quat = (1.0 - alpha) * q0 + alpha * (q1 if np.dot(q0, q1) >= 0 else -q1)
  quat /= np.linalg.norm(quat)
  rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz")
  x = np.r_[root[:3], rotation, blend("joint_pos"), blend("gimbal_pos")]
  ik.set_pose(x, blend("brick_pos"))
  mujoco.mj_forward(ik.model, ik.data)
  renderer.update_scene(ik.data, camera=camera)
  return renderer.render().copy()


def caption_frame(frame, phase, time_s, duration_s, font):
  canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 64), (18, 27, 38))
  canvas.paste(Image.fromarray(frame), (0, 64))
  draw = ImageDraw.Draw(canvas)
  draw.text((14, 7), "Phybot C2 | Kinematic reference", font=font, fill=(240, 244, 248))
  draw.text(
    (14, 35),
    f"{phase + 1}/{len(PHASE_NAMES)}  {PHASE_NAMES[phase]}    {time_s:04.1f} / {duration_s:.1f} s",
    font=font,
    fill=(149, 208, 229),
  )
  return np.asarray(canvas)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
  parser.add_argument("--output", type=Path, default=Path("/tmp/brick_reference_preview.png"))
  parser.add_argument("--fps", type=int, default=25, help="MP4 frames per second")
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=640, help="Scene height, excluding the MP4 caption")
  args = parser.parse_args()
  if args.output.suffix.lower() not in (".png", ".mp4"):
    parser.error("--output must end in .png or .mp4")
  if min(args.fps, args.width, args.height) <= 0:
    parser.error("--fps, --width and --height must be positive")
  ref = load_reference(args.reference)
  brick = BrickCfg(size=tuple(ref["brick_size"]), mass=float(ref["brick_mass"]))
  ik = BrickIK(brick, str(ref["unit_layout"]))
  ik.model.vis.global_.offwidth = max(args.width, ik.model.vis.global_.offwidth)
  ik.model.vis.global_.offheight = max(args.height, ik.model.vis.global_.offheight)
  renderer = mujoco.Renderer(ik.model, width=args.width, height=args.height)
  camera = mujoco.MjvCamera()
  camera.lookat[:] = [0.05, 0.07, 0.6]
  camera.distance, camera.azimuth, camera.elevation = 2.4, 125.0, -15.0
  args.output.parent.mkdir(parents=True, exist_ok=True)
  try:
    if args.output.suffix.lower() == ".png":
      frames = [render_frame(ik, renderer, camera, ref, phase, 1.0) for phase in range(len(PHASE_NAMES))]
      iio.imwrite(args.output, np.concatenate(frames, axis=1))
      print(f"Kinematic phase endpoints ({', '.join(PHASE_NAMES)}): {args.output}")
    else:
      try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
      except OSError:
        font = ImageFont.load_default()
      phase_ends = np.cumsum(ref["durations"])
      duration_s = float(phase_ends[-1])
      frame_count = int(np.ceil(duration_s * args.fps)) + 1
      last_phase = -1
      with iio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8, pixelformat="yuv420p") as writer:
        for index in range(frame_count):
          time_s = min(index / args.fps, duration_s)
          phase = min(int(np.searchsorted(phase_ends, time_s, side="right")), len(PHASE_NAMES) - 1)
          start_s = float(phase_ends[phase - 1]) if phase else 0.0
          progress = (time_s - start_s) / float(ref["durations"][phase])
          frame = render_frame(ik, renderer, camera, ref, phase, progress)
          writer.append_data(caption_frame(frame, phase, time_s, duration_s, font))
          if phase != last_phase:
            print(f"Rendering {phase + 1}/{len(PHASE_NAMES)}: {PHASE_NAMES[phase]}", flush=True)
            last_phase = phase
      print(f"Kinematic reference ({duration_s:.1f} s, {frame_count} frames): {args.output}")
  finally:
    renderer.close()


if __name__ == "__main__":
  main()
