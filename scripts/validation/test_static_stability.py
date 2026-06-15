"""Static stability of heel-down vs flat-foot standing poses, no DR, zero action.

Uses plain MuJoCo (CPU) with the compiled walk-task model pieces: robot spec +
ground plane, PD position actuators at the deploy gains, holding the target
pose. 15 s rollouts, log tilt / base z / which part of the sole touches.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path("/home/spider/projects/scr-phybot_mjlab")
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))

import mujoco
import numpy as np

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_STANDING_JOINT_POS,
)

POSES = {
  # name: (ankle_pitch, base_z)
  "heel_down(-0.295)": (-0.295, 0.726),
  "flat(-0.254)": (-0.254, 0.722),
}


def build_model(ankle_pitch: float, base_z: float) -> tuple[mujoco.MjModel, dict]:
  from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import get_spec

  spec = get_spec()
  # Ground plane.
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1],
    friction=[0.7, 0.005, 0.0001],
  )
  # PD position actuators at deploy gains for the 21 policy joints + neck.
  gains = {
    "hip_pitch": 100.0, "hip_roll": 100.0, "hip_yaw": 100.0, "knee": 100.0,
    "ankle_pitch": 50.0, "ankle_roll": 50.0, "waist_yaw": 20.0,
    "shoulder_pitch": 50.0, "shoulder_roll": 50.0, "shoulder_yaw": 50.0,
    "elbow_pitch": 50.0, "neck": 20.0,
  }
  damps = {"waist_yaw": 10.0, "neck": 2.0}
  for joint in spec.joints:
    if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
      continue
    group = joint.name.removeprefix("left_").removeprefix("right_")
    kp = gains.get(group)
    if kp is None:
      continue
    kv = damps.get(group, 5.0)
    spec.add_actuator(
      name=f"pd_{joint.name}", target=joint.name,
      trntype=mujoco.mjtTrn.mjTRN_JOINT,
      gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[kp] + [0.0] * 9,
      biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0.0, -kp, -kv] + [0.0] * 7,
    )
  model = spec.compile()
  model.opt.timestep = 0.005

  pose = dict(PHYBOT_C2_STANDING_JOINT_POS)
  pose["left_ankle_pitch"] = ankle_pitch
  pose["right_ankle_pitch"] = ankle_pitch
  return model, {"pose": pose, "base_z": base_z}


def run(name: str, ankle_pitch: float, base_z: float) -> None:
  model, info = build_model(ankle_pitch, base_z)
  data = mujoco.MjData(model)
  pose = info["pose"]

  data.qpos[:3] = [0, 0, base_z + 0.005]
  data.qpos[3:7] = [1, 0, 0, 0]
  for jname, jval in pose.items():
    j = model.joint(jname)
    data.qpos[model.jnt_qposadr[j.id]] = jval
    try:
      data.ctrl[model.actuator(f"pd_{jname}").id] = jval
    except KeyError:
      pass  # unactuated (neck) — negligible mass

  mujoco.mj_forward(model, data)
  print(f"--- {name}: ankle_pitch={ankle_pitch} base_z={base_z}")
  for step in range(3000):  # 15 s
    mujoco.mj_step(model, data)
    if (step + 1) % 400 == 0:
      quat = data.qpos[3:7]
      R = np.zeros(9)
      mujoco.mju_quat2Mat(R, quat)
      tilt = np.degrees(np.arccos(np.clip(R[8], -1, 1)))
      # Foot contact x-extent (world) to see heel-only vs full sole.
      xs = []
      for c in range(data.ncon):
        g1 = model.geom(data.contact[c].geom1).name
        g2 = model.geom(data.contact[c].geom2).name
        if "floor" in (g1, g2) and ("ankle_roll" in g1 or "ankle_roll" in g2):
          xs.append(data.contact[c].pos[0])
      span = (max(xs) - min(xs)) if len(xs) >= 2 else 0.0
      print(
        f"t={(step + 1) * 0.005:5.1f}s tilt={tilt:6.2f}deg z={data.qpos[2]:.4f} "
        f"foot_contacts={len(xs)} contact_x_span={span:.3f}m"
      )
      if tilt > 80:
        print("  -> fallen, stop")
        break


for name, (ap, z) in POSES.items():
  run(name, ap, z)
