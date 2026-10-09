"""Generate a C2 zero-command standing AMP clip.

The output format matches amp-rsl-rl's `.npy` motion dataset schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import scr_phybot_mjab.tasks  # noqa: E402,F401
from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (  # noqa: E402
  PHYBOT_C2_POLICY_JOINT_NAMES,
  PHYBOT_C2_STANDING_JOINT_POS,
)
from scr_phybot_mjab.tasks.walk.motions import STAND_ZERO_MOTION


def _smooth_jitter(
  frames: int,
  joints: int,
  fps: float,
  amplitude: float,
  seed: int,
) -> np.ndarray:
  if amplitude <= 0.0:
    return np.zeros((frames, joints), dtype=np.float32)

  rng = np.random.default_rng(seed)
  t = np.arange(frames, dtype=np.float32)[:, None] / np.float32(fps)
  freqs = rng.uniform(0.08, 0.35, size=(1, joints)).astype(np.float32)
  phases = rng.uniform(0.0, 2.0 * np.pi, size=(1, joints)).astype(np.float32)
  amps = rng.uniform(0.35, 1.0, size=(1, joints)).astype(np.float32) * np.float32(amplitude)
  jitter = amps * np.sin(2.0 * np.pi * freqs * t + phases)

  # Fade endpoints so the synthetic clip loops without a sharp discriminator cue.
  ramp_len = max(1, min(frames // 5, int(fps)))
  ramp = np.ones((frames, 1), dtype=np.float32)
  fade = np.sin(np.linspace(0.0, np.pi / 2.0, ramp_len, dtype=np.float32)) ** 2
  ramp[:ramp_len, 0] = fade
  ramp[-ramp_len:, 0] = fade[::-1]
  return (jitter * ramp).astype(np.float32)


def generate_clip(
  output: Path,
  duration_s: float,
  fps: float,
  jitter_rad: float,
  seed: int,
) -> None:
  frames = int(round(duration_s * fps))
  if frames < 2:
    raise ValueError("duration_s * fps must produce at least two frames")

  joint_names = list(PHYBOT_C2_POLICY_JOINT_NAMES)
  base_pose = np.array(
    [PHYBOT_C2_STANDING_JOINT_POS[name] for name in joint_names],
    dtype=np.float32,
  )
  joint_positions = base_pose[None, :] + _smooth_jitter(
    frames=frames,
    joints=len(joint_names),
    fps=fps,
    amplitude=jitter_rad,
    seed=seed,
  )

  data = {
    "joints_list": joint_names,
    "joint_positions": joint_positions.astype(np.float32),
    "root_position": np.tile(np.array([[0.0, 0.0, 0.726]], dtype=np.float32), (frames, 1)),
    # amp-rsl-rl datasets use SciPy's xyzw quaternion convention.
    "root_quaternion": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (frames, 1)),
    "fps": np.float32(fps),
  }

  output.parent.mkdir(parents=True, exist_ok=True)
  np.save(output, data)
  print(f"wrote {output} ({frames} frames, {duration_s:.2f}s @ {fps:.1f} fps, jitter={jitter_rad:.4f} rad)")


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--output",
    type=Path,
    default=STAND_ZERO_MOTION,
  )
  parser.add_argument("--duration-s", type=float, default=15.0)
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument("--jitter-rad", type=float, default=0.003)
  parser.add_argument("--seed", type=int, default=7)
  args = parser.parse_args(argv)

  generate_clip(
    output=args.output,
    duration_s=args.duration_s,
    fps=args.fps,
    jitter_rad=args.jitter_rad,
    seed=args.seed,
  )


if __name__ == "__main__":
  main()
