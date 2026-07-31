#!/usr/bin/env python3
"""Measure the engaged-pad rotational stiffness the weld actually provides.

For each candidate ``anchor_torquescale`` this builds the climb env at zero
gravity, latches all four pads at the spawn stance, settles, then ramps a
pure external moment on one EE body while recording the pad's rotation from
its engage-capture orientation. The slope of the moment-tilt line is the
chassis-to-wall rotational stiffness [N*m/rad] — the quantity to match to
the physical spine-suspension estimate (~0.1 N*m at ~4 deg of edge-pivot
travel => ~1.4 N*m/rad; see docs/microspine_tilt_calibration.md).

The limit surface is disabled during the probe (no releases; pure stiffness
measurement). Usage:

  python scripts/calibrate_tilt_stiffness.py --torquescales 0.005 0.01 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import torch

import validate_microspine_constraint as vmc  # noqa: E402  (script import)


def probe(torquescale: float, ee_index: int, axis: str, args) -> float:
  """Return the fitted rotational stiffness [N*m/rad] for one EE and axis."""
  env = vmc.build_env(
    num_envs=1,
    limit_surface_enable=False,  # pure stiffness probe: no releases
    anchor_torquescale=torquescale,
  )
  try:
    executor, affordance = vmc.get_terms(env.unwrapped)
    u = env.unwrapped
    robot = u.scene["robot"]
    env.reset()
    vmc.set_gravity(u, 0.0, magnitude=0.0)
    actions = vmc.hold_pose_actions(u)
    zero_force = torch.zeros(1, 1, 3, device=u.device)
    axis_vec = {"y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis]
    axis_t = torch.tensor([[list(axis_vec)]], device=u.device)
    body_ids = [executor._ee_body_ids[ee_index]]

    # Settle with zero moment so PD transients decay.
    for _ in range(args.settle_steps):
      env.step(actions)

    q0 = robot.data.body_link_quat_w[:, body_ids[0], :].clone()
    moments, tilts = [], []
    for k in range(args.ramp_steps):
      m_val = args.max_moment * (k + 1) / args.ramp_steps
      robot.write_external_wrench_to_sim(
        forces=zero_force, torques=m_val * axis_t, body_ids=body_ids
      )
      env.step(actions)
      q = robot.data.body_link_quat_w[:, body_ids[0], :]
      dot = (q * q0).sum(dim=-1).abs().clamp(max=1.0)
      tilt = float(2.0 * torch.acos(dot))
      moments.append(m_val)
      tilts.append(tilt)
    moments = np.asarray(moments)
    tilts = np.asarray(tilts)
    # Least-squares slope through the origin over the sampled ramp.
    denom = float((tilts * tilts).sum())
    slope = float((moments * tilts).sum()) / max(denom, 1.0e-12)
    peak_tilt = float(np.degrees(tilts.max()))
    print(
      f"  ts={torquescale:<7g} ee={ee_index} axis={axis}: "
      f"k = {slope:7.2f} N*m/rad   (peak tilt {peak_tilt:.2f} deg at "
      f"{args.max_moment:.2f} N*m)"
    )
    return slope
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--torquescales", type=float, nargs="+", default=[0.005, 0.01, 0.05]
  )
  parser.add_argument("--ee-indices", type=int, nargs="+", default=[0, 2])
  parser.add_argument("--axes", type=str, nargs="+", default=["y", "z"])
  parser.add_argument("--max-moment", type=float, default=0.3)
  parser.add_argument("--ramp-steps", type=int, default=150)
  parser.add_argument("--settle-steps", type=int, default=50)
  args = parser.parse_args()
  print(
    "engaged-pad rotational stiffness probe (target ~1.4 N*m/rad: "
    "0.1 N*m at ~4 deg suspension travel)"
  )
  for ts in args.torquescales:
    for ee in args.ee_indices:
      for axis in args.axes:
        probe(ts, ee, axis, args)


if __name__ == "__main__":
  main()
