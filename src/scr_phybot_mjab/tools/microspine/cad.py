"""Read the supplied pad assembly without scaling or deforming its parts.

Offline asset tooling only. STEP analytic cylinders locate the sleeve stop
and bearing-seat midplane; the URDF/GLTF export supplies the assembled solids.
The exported planar mates are not a model of the hardware's passive bearing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from scr_phybot_mjab.tools.paths import REPOSITORY_ROOT

CAD_DIRECTORY = REPOSITORY_ROOT / "cad_model/assembly_1"
CAD_STEP = REPOSITORY_ROOT / "cad_model/Assembly 1.step"


def origin_transform(origin: ET.Element | None) -> np.ndarray:
  result = np.eye(4)
  if origin is not None:
    result[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    result[:3, :3] = Rotation.from_euler(
      "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    ).as_matrix()
  return result


def cad_parts(directory: Path = CAD_DIRECTORY) -> list[tuple[str, str, trimesh.Trimesh]]:
  """Return every solid in the CAD connector's visual frame, in metres."""
  robot = ET.parse(directory / "urdf/assembly_1.urdf").getroot()
  poses = {"root": np.eye(4)}
  pending = list(robot.findall("joint"))
  while pending:
    ready = [j for j in pending if j.find("parent").get("link") in poses]
    if not ready:
      raise ValueError("Disconnected CAD assembly")
    for joint in ready:
      poses[joint.find("child").get("link")] = (
        poses[joint.find("parent").get("link")] @ origin_transform(joint.find("origin"))
      )
      pending.remove(joint)
  to_connector = np.linalg.inv(origin_transform(robot.find("link[@name='root']/visual/origin")))
  cache, result = {}, []
  for link in robot.findall("link"):
    for visual in link.findall("visual"):
      geometry = visual.find("geometry/mesh")
      if not np.allclose(np.fromstring(geometry.get("scale", "1 1 1"), sep=" "), 1):
        raise ValueError("CAD import expects unscaled metric meshes")
      filename = Path(geometry.get("filename")).name
      if filename not in cache:
        mesh = trimesh.load_scene(directory / "meshes" / filename).to_geometry()
        mesh.merge_vertices()
        if not mesh.is_watertight or mesh.volume <= 0:
          raise ValueError(f"Non-solid CAD part: {filename}")
        cache[filename] = mesh
      mesh = cache[filename].copy()
      mesh.apply_transform(to_connector @ poses[link.get("name")] @ origin_transform(visual.find("origin")))
      result.append((link.get("name"), filename, mesh))
  return result


def connector_features(path: Path = CAD_STEP) -> dict:
  """Read the two trimmed cylindrical bores from this Onshape STEP export.

  This intentionally small STEP reader supports analytic cylinder/circle
  features, not arbitrary CAD topology. It rejects ambiguous geometries.
  No numeric STEP entity identifiers or mounting offsets are hard-coded.
  """
  entities = {int(i): v for i, v in re.findall(r"#(\d+)\s*=\s*(.*?);", path.read_text(), re.S)}

  def refs(i):
    return [int(x) for x in re.findall(r"#(\d+)", entities[i])]

  def xyz(i):
    return np.fromstring(re.search(r"\(([^()]*)\)\s*\)$", entities[i]).group(1), sep=",")

  def circle(i):
    placement = refs(refs(i)[0])
    radius = float(re.search(r",([^,]+)\)$", entities[i]).group(1))
    return radius, xyz(placement[0]), xyz(placement[1])

  solids = []
  for i, text in entities.items():
    if text.startswith("MANIFOLD_SOLID_BREP"):
      faces = refs(refs(i)[0])
      cylinders = {refs(f)[-1]: f for f in faces if entities[refs(f)[-1]].startswith("CYLINDRICAL_SURFACE")}
      if cylinders:
        solids.append(cylinders)
  # The connector contains the largest cylindrical surface in this assembly.
  cylinders = max(solids, key=lambda values: max(circle(i)[0] for i in values))
  parallel = [i for i in cylinders if np.allclose(np.abs(circle(i)[2]), [0, 0, 1], atol=1e-8)]
  if len(parallel) != 2:
    raise ValueError("Expected concentric inner and outer sleeve cylinders")
  inner, outer = sorted(parallel, key=lambda i: circle(i)[0])
  transverse = [i for i in cylinders if np.allclose(np.abs(circle(i)[2]), [0, 1, 0], atol=1e-8)]
  bearing = max(transverse, key=lambda i: circle(i)[0])

  def bore_ends(surface):
    radius, origin, direction = circle(surface)
    seen, todo = set(), refs(cylinders[surface])[:-1]
    while todo:
      i = todo.pop()
      if i not in seen:
        seen.add(i)
        todo.extend(refs(i))
    centres = []
    for i in seen:
      if entities[i].startswith("CIRCLE"):
        r, p, axis = circle(i)
        if np.isclose(r, radius, atol=1e-10) and np.isclose(abs(axis @ direction), 1, atol=1e-8):
          centres.append(p)
    centres = np.unique(np.round(centres, 12), axis=0)
    if centres.shape != (2, 3):
      raise ValueError("Expected exactly two circular bore ends")
    return centres

  sleeve = bore_ends(inner)
  sleeve = sleeve[np.argsort(sleeve[:, 2])]
  seat = bore_ends(bearing)
  if not np.allclose(sleeve[:, :2], circle(outer)[1][:2], atol=1e-9):
    raise ValueError("Sleeve cylinders are not concentric")
  return {
    "bearing_center_m": seat.mean(axis=0).tolist(),
    "bearing_seat_ends_m": seat.tolist(),
    "bearing_seat_radius_m": circle(bearing)[0],
    "sleeve_mouth_m": sleeve[0].tolist(),
    "sleeve_shoulder_m": sleeve[1].tolist(),
    "sleeve_inner_radius_m": circle(inner)[0],
    "sleeve_outer_radius_m": circle(outer)[0],
    "reference": "B is the midpoint of the CAD cylindrical bearing seat; E is the inner sleeve stop.",
  }


@dataclass
class CadPadAssembly:
  parts: list[tuple[str, str, trimesh.Trimesh]]
  unit_centres: np.ndarray
  unit_parts: list[trimesh.Trimesh]
  features: dict
  colour: str

  @property
  def pivot(self) -> np.ndarray:
    return np.asarray(self.features["bearing_center_m"])

  @property
  def contact_center(self) -> np.ndarray:
    return self.unit_centres.mean(axis=0) - self.pivot


def load_cad_pad_assembly(directory: Path = CAD_DIRECTORY, step: Path = CAD_STEP) -> CadPadAssembly:
  parts = cad_parts(directory)
  units = [m for _, f, m in parts if f == "Part_1_1.gltf"]
  centres = np.array([[m.bounds[:, 0].mean(), m.bounds[0, 1], m.bounds[:, 2].mean()] for m in units])
  order = np.lexsort((centres[:, 0], np.round(centres[:, 2], 6)))
  centres = centres[order]
  units = [units[i] for i in order]
  if len(units) != 12:
    raise ValueError("Expected the supplied 4-column, 3-row pad")
  root = ET.parse(directory / "urdf/assembly_1.urdf").getroot()
  colours = {v.find("material/color").get("rgba") for v in root.findall("link/visual")}
  if len(colours) != 1:
    raise ValueError("Per-part CAD materials need explicit handling")
  return CadPadAssembly(parts, centres, units, connector_features(step), colours.pop())
