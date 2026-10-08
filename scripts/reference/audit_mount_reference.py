#!/usr/bin/env python3
"""Check a mount clip against current CAD FK and native wall/floor geometry.

Checks collision meshes and primitives at every saved frame, including feet
already attached to the wall. This is a kinematic audit, not a force/preload
or policy rollout certificate. Collision clearance uses the actual mesh
vertices, not their conservative bounding boxes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mjlab  # noqa: F401  # Initialize task registry before robot imports.
import mujoco
import numpy as np

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import PHYBOT_C2_XML
from scr_phybot_mjab.tools.climbing_reference.model import (
  WallGeometry,
  build_reference_model,
)


def audit_motion_file(path: Path) -> dict:
  clip = np.load(path, allow_pickle=True).item()
  model, sites, bodies, _, _ = build_reference_model(tuple(clip["ee_list"]))
  data = mujoco.MjData(model)
  wall = WallGeometry(clip["wall_angle"], clip["wall_surface_x"])
  frames = len(clip["joint_positions"])
  names = list(clip["joints_list"]) + list(clip.get("passive_joint_names", ()))
  qadr = [int(model.joint(n).qposadr[0]) for n in names]
  joint_pos = np.concatenate(
    [
      clip["joint_positions"],
      clip.get("passive_joint_positions", np.zeros((frames, 0))),
    ],
    axis=1,
  )
  signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
  geoms = []
  for g in range(model.ngeom):
    if not (model.geom_contype[g] or model.geom_conaffinity[g]) or model.geom_bodyid[g] == 0:
      continue
    kind = model.geom_type[g]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
      mid = model.geom_dataid[g]
      start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
      points = model.mesh_vert[start : start + count].astype(np.float64)
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
      points = signs * model.geom_size[g]
    else:
      points = None
    geoms.append((g, kind, points))

  clearances: dict[str, dict] = {}

  def record(key: str, gap: float, frame: int, geom: int) -> None:
    if key not in clearances or gap * 1000 < clearances[key]["gap_mm"]:
      clearances[key] = {
        "gap_mm": float(gap * 1000),
        "frame": frame,
        "geom": model.geom(geom).name,
        "segment": clip["segment_names"][int(clip["phase_segment"][frame])],
      }

  fk_error = center_error = rotation_error = drift = 0.0
  for t in range(frames):
    data.qpos[:] = model.qpos0
    data.qpos[:3] = clip["root_position"][t]
    data.qpos[3:7] = clip["root_quaternion"][t, [3, 0, 1, 2]]
    data.qpos[qadr] = joint_pos[t]
    mujoco.mj_kinematics(model, data)
    for i, ee in enumerate(clip["ee_list"]):
      bid = bodies[ee]
      center = data.site_xpos[sites[ee]]
      fk_error = max(fk_error, float(np.linalg.norm(data.xpos[bid] - clip["ee_positions"][t, i])))
      center_error = max(center_error, float(np.linalg.norm(center - clip["pad_positions"][t, i])))
      ref_quat = clip["ee_quaternions"][t, i, [3, 0, 1, 2]].astype(float)
      ref_quat /= np.linalg.norm(ref_quat)
      rotation_error = max(rotation_error, 1.0 - abs(float(data.xquat[bid] @ ref_quat)))
      if clip["ee_engaged"][t, i]:
        drift = max(drift, float(np.linalg.norm(center - clip["ee_anchor_positions"][t, i])))

    data.qpos[:3] = wall.rotate_point(clip["root_position"][t])
    mujoco.mju_mulQuat(data.qpos[3:7], wall.quat, clip["root_quaternion"][t, [3, 0, 1, 2]].astype(float))
    mujoco.mj_kinematics(model, data)
    segment = clip["segment_names"][int(clip["phase_segment"][t])]
    for g, kind, points in geoms:
      rot = data.geom_xmat[g].reshape(3, 3)
      directions = np.stack([rot[2], wall.outward @ rot], axis=1)
      if points is not None:
        support = (points @ directions).min(axis=0)
      elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        support = np.full(2, -model.geom_size[g, 0])
      elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        support = -model.geom_size[g, 0] - model.geom_size[g, 1] * np.abs(directions[2])
      elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        support = -model.geom_size[g, 0] * np.linalg.norm(directions[:2], axis=0) - model.geom_size[g, 1] * np.abs(
          directions[2]
        )
      else:
        raise ValueError(f"Unsupported collision primitive: {model.geom(g).name} ({kind})")
      floor_gap = data.geom_xpos[g, 2] + support[0]
      wall_gap = wall.distance(data.geom_xpos[g]) + support[1]
      record("floor", floor_gap, t, g)
      record("wall", wall_gap, t, g)
      record(f"floor/{segment}", floor_gap, t, g)
      record(f"wall/{segment}", wall_gap, t, g)

  xml = PHYBOT_C2_XML
  xml_hash = hashlib.sha256(xml.read_bytes()).hexdigest()
  checks = {
    "asset_xml_matches": clip.get("asset_xml_sha256") == xml_hash,
    "body_fk_within_2mm": fk_error < 0.002,
    "contact_c_fk_within_2mm": center_error < 0.002,
    "body_orientation_matches": rotation_error < 1e-6,
    "engaged_anchor_drift_within_5mm": drift < 0.005,
    # Floor contacts intentionally carry a 2 mm seating bias in the IK.
    "floor_penetration_within_3mm": clearances["floor"]["gap_mm"] >= -3.0,
    "wall_penetration_within_2mm": clearances["wall"]["gap_mm"] >= -2.0,
  }
  return {
    "clip": path.name,
    "clip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "asset_xml_sha256": xml_hash,
    "frames": frames,
    "fps": float(clip["fps"]),
    "wall_angle_deg": float(clip["wall_angle"]),
    "body_fk_max_mm": fk_error * 1000,
    "contact_c_fk_max_mm": center_error * 1000,
    "quaternion_dot_error": rotation_error,
    "engaged_anchor_drift_max_mm": drift * 1000,
    "checks": checks,
    "passed": all(checks.values()),
    "scope": "Saved-frame FK, wall and floor collision geometry; excludes self-collision, force/preload feasibility and dynamic policy tracking.",
    "clearances": clearances,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("motion_file", type=Path)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  report = audit_motion_file(args.motion_file)
  output = args.output or args.motion_file.with_suffix(".audit.json")
  output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps({k: v for k, v in report.items() if k != "clearances"}, indent=2))
  print(f"Full report: {output}")
  raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
  main()
