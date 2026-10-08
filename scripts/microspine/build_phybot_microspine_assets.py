"""Rigidly import the supplied Onshape pad assembly into the PHYBot asset.

CAD geometry and assembly transforms are authoritative. The sleeve stop
seats on the bare forearm end. No CAD vertices are warped or rescaled.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from xml.etree import ElementTree as ET

import mjlab  # noqa: F401  # Initialize plugins before project tool imports.
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from scr_phybot_mjab.tools.microspine.cad import (
  CAD_DIRECTORY,
  CAD_STEP,
  CadPadAssembly,
  load_cad_pad_assembly,
)
from scr_phybot_mjab.tools.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
ASSET = ROOT / "src/scr_phybot_mjab/assets/phybot_c2"
CAD = CAD_DIRECTORY
PLA_DENSITY = 1240.0  # kg/m^3, provisional solid-PLA estimate, no infill claim.
SPINES = tuple(f"{side}_{limb}_spine" for limb in ("foot", "hand") for side in ("left", "right"))


def numbers(values) -> str:
  return " ".join(f"{float(v):.12g}" for v in np.atleast_1d(values))


def printed_inertia(meshes: list[trimesh.Trimesh]) -> dict:
  masses = np.array([mesh.volume * PLA_DENSITY for mesh in meshes])
  centers = np.array([mesh.center_mass for mesh in meshes])
  center = np.sum(masses[:, None] * centers, axis=0) / masses.sum()
  inertia = np.zeros((3, 3))
  for mesh, mass, com in zip(meshes, masses, centers):
    delta = com - center
    inertia += mesh.moment_inertia * PLA_DENSITY
    inertia += mass * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
  if np.linalg.eigvalsh(inertia).min() <= 0:
    raise ValueError("Invalid printed inertia")
  return {"mass_kg": float(masses.sum()), "com_m": center.tolist(), "inertia_com_kg_m2": inertia.tolist()}


def forearms_without_hands(original):
  result = {}
  for side in ("left", "right"):
    source = ASSET / "new_meshes" / f"{side}_elbow_pitch.STL"
    mesh = trimesh.load_mesh(source)
    components = mesh.split(only_watertight=False)
    # The dummy gripper and its wrist shell are separate STL components,
    # entirely distal to z=-173 mm. Retain the elbow and the 42-mm-OD tube
    # (z=-68..-188 mm), not a planar cut through the overlapping wrist shell.
    kept = [part for part in components if part.bounds[1, 2] > -0.150]
    if not kept or any(not part.is_watertight for part in kept):
      raise ValueError(f"Could not isolate solid forearm components: {side}")
    clean = trimesh.util.concatenate(kept)
    clean.export(ASSET / "new_meshes" / f"{side}_forearm_no_hand.stl")
    end_z = float(clean.bounds[0, 2])
    if abs(end_z + 0.188) > 1e-5:
      raise ValueError(f"Unexpected bare forearm end: {end_z}")
    props = printed_inertia(kept)
    old_mass = float(original.find(f".//body[@name='{side}_elbow_pitch']/inertial").get("mass"))
    # The actuator/tube is not PLA. Retain its existing nominal link mass,
    # redistributing it over the retained geometry as a provisional estimate.
    props["inertia_com_kg_m2"] = (np.array(props["inertia_com_kg_m2"]) * old_mass / props["mass_kg"]).tolist()
    props["mass_kg"] = old_mass
    result[side] = {
      "forearm_end_z_m": end_z,
      "inertial": props,
      "retained_components": len(kept),
      "removed_components": len(components) - len(kept),
    }
  return result


def normalize_geometry():
  assembly = load_cad_pad_assembly()
  # A SINGLE rigid translation is shared by every assembled solid.
  parts = []
  for link, filename, raw in assembly.parts:
    mesh = raw.copy()
    mesh.apply_translation(-assembly.pivot)
    parts.append((link, filename, mesh))
  unit = assembly.unit_parts[0].copy()
  unit.apply_translation(-assembly.unit_centres[0])
  backplate = next(m for _, f, m in parts if f == "Part_1_4.gltf")
  connector = next(m for _, f, m in parts if f == "Part_1_6.gltf")
  brackets = trimesh.util.concatenate(
    [m for _, f, m in parts if f not in ("Part_1_1.gltf", "Part_1_4.gltf", "Part_1_6.gltf")]
  )
  moving_meshes = [m for _, f, m in parts if f != "Part_1_6.gltf"]
  outputs = {"unit": unit, "backplate": backplate, "brackets": brackets, "forearm_mount": connector}
  for name, mesh in outputs.items():
    mesh.fix_normals()
    if not mesh.is_watertight:
      raise ValueError(f"Generated non-solid mesh: {name}")
    mesh.export(ASSET / "new_meshes" / f"microspine_{name}.stl")
  return outputs, printed_inertia(moving_meshes), printed_inertia([connector]), assembly


def add_inertial(body: ET.Element, props: dict):
  inertia = np.array(props["inertia_com_kg_m2"])
  ET.SubElement(
    body,
    "inertial",
    pos=numbers(props["com_m"]),
    mass=numbers(props["mass_kg"]),
    fullinertia=numbers(inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]),
  )


def visual(body, name, mesh, color, pos=(0, 0, 0)):
  return ET.SubElement(
    body,
    "geom",
    name=name,
    type="mesh",
    mesh=mesh,
    pos=numbers(pos),
    contype="0",
    conaffinity="0",
    density="0",
    group="1",
    rgba=color,
  )


def collision(body, name, size, pos, kind="box"):
  return ET.SubElement(
    body,
    "geom",
    name=name,
    type=kind,
    size=numbers(size),
    pos=numbers(pos),
    mass="0",
    density="0",
    group="3",
    rgba="0.25 0.55 0.85 0.25",
  )


def pad_body(name: str, pos, quat, props: dict, meshes, assembly: CadPadAssembly) -> ET.Element:
  body = ET.Element("body", name=name, pos=numbers(pos), quat=numbers(quat))
  add_inertial(body, props)
  visual(body, f"{name}_visual", "microspine_backplate", assembly.colour)
  visual(body, f"{name}_brackets_visual", "microspine_brackets", assembly.colour)
  centres = assembly.unit_centres - assembly.pivot
  for index, center in enumerate(centres):
    row, col = divmod(index, 4)
    visual(body, f"{name}_unit_{row}_{col}_visual", "microspine_unit", assembly.colour, center)
    ET.SubElement(
      body, "site", name=f"{name}_unit_{row}_{col}", pos=numbers(center), size="0.0015", group="4", rgba="1 0.55 0.1 1"
    )
  bounds = meshes["backplate"].bounds
  collision(body, f"{name}_collision", np.diff(bounds, axis=0)[0] / 2, bounds.mean(axis=0))
  for label, point, color in (
    ("ball_center", [0, 0, 0], "0.1 0.5 1 1"),
    ("contact_center", assembly.contact_center, "1 0.25 0.1 1"),
  ):
    ET.SubElement(body, "site", name=f"{name}_{label}", pos=numbers(point), size="0.003", group="4", rgba=color)
  # Only the four corner units collide. Their boxes use the printed unit's
  # actual bounds and CAD placement. No fabricated pin/ball visual is added.
  unit_bounds = meshes["unit"].bounds
  half_size = np.diff(unit_bounds, axis=0)[0] / 2
  box_offset = np.mean(unit_bounds, axis=0)
  for row, source_row in enumerate((0, 2)):
    for col, source_col in enumerate((0, 3)):
      contact_name = f"{name}_c_{row}_{col}"
      child = ET.SubElement(
        body, "body", name=contact_name, pos=numbers(centres[source_row * 4 + source_col] + box_offset)
      )
      ET.SubElement(
        child,
        "geom",
        name=contact_name,
        type="box",
        size=numbers(half_size),
        contype="1",
        conaffinity="1",
        condim="6",
        priority="1",
        friction="30.0 0.01 0.01",
        solref="0.02 1.5",
        solimp="0.9 0.99 0.001 0.5 2",
        rgba="0.9 0.7 0.5 0",
        mass="0",
      )
  return body


def mount_body(side, pos, quat, props, meshes, assembly):
  name = f"{side}_forearm_pad_mount"
  body = ET.Element("body", name=name, pos=numbers(pos), quat=numbers(quat))
  add_inertial(body, props)
  visual(body, f"{name}_visual", "microspine_forearm_mount", assembly.colour)
  features = assembly.features
  mouth, stop = np.asarray(features["sleeve_mouth_m"]), np.asarray(features["sleeve_shoulder_m"])
  collision(
    body,
    f"{name}_sleeve_collision",
    [features["sleeve_outer_radius_m"], (stop[2] - mouth[2]) / 2],
    (mouth + stop) / 2 - assembly.pivot,
    "cylinder",
  )
  # Conservative envelope of the solid connector below the insertion sleeve.
  raw = meshes["forearm_mount"].vertices
  web = raw[raw[:, 2] >= stop[2] - assembly.pivot[2] - 1e-8]
  bounds = np.array([web.min(axis=0), web.max(axis=0)])
  collision(body, f"{name}_web_collision", np.diff(bounds, axis=0)[0] / 2, bounds.mean(axis=0))
  ET.SubElement(body, "site", name=f"{side}_forearm_end", pos=numbers(stop - assembly.pivot), size="0.002", group="4")
  return body


def element_text(element, indent=0):
  element = copy.deepcopy(element)
  # The replacement range excludes the original element's tail. Serializing
  # that tail as well doubles whitespace after every rebuild.
  element.tail = None
  ET.indent(element, space="  ", level=indent)
  return ET.tostring(element, encoding="unicode")


def replace_named(text: str, tag: str, name: str, replacement: str, required=True) -> str:
  start = re.search(rf'<{tag}\s+name="{re.escape(name)}"(?:\s|>)', text)
  if start is None:
    if required:
      raise ValueError(f"Missing {tag}: {name}")
    return text
  depth = 0
  for token in re.finditer(rf"</?{tag}\b[^>]*>", text[start.start() :]):
    raw = token.group()
    depth += -1 if raw.startswith("</") else (0 if raw.endswith("/>") else 1)
    if depth == 0:
      stop = start.start() + token.end()
      return text[: start.start()] + replacement + text[stop:]
  raise ValueError(f"Unclosed {tag}: {name}")


def sync_urdf(pads, mounts, forearms):
  path = ASSET / "urdf/phybot_c2.urdf"
  text = path.read_text()
  package = ET.parse(ASSET / "package.xml").findtext("name")
  for side in ("left", "right"):
    text = replace_named(text, "link", f"{side}_wrist_yaw", "", required=False)
    text = replace_named(text, "joint", f"{side}_wrist_yaw", "", required=False)
  for body in [*pads, *mounts]:
    name = body.get("name")
    link = ET.Element("link", name=name)
    inert = body.find("inertial")
    new_inert = ET.SubElement(link, "inertial")
    ET.SubElement(new_inert, "origin", xyz=inert.get("pos"), rpy="0 0 0")
    ET.SubElement(new_inert, "mass", value=inert.get("mass"))
    vals = inert.get("fullinertia").split()
    ET.SubElement(new_inert, "inertia", **dict(zip(("ixx", "iyy", "izz", "ixy", "ixz", "iyz"), vals)))
    for geom in body.findall("geom"):
      is_collision = "_collision" in geom.get("name", "")
      vis = ET.SubElement(link, "collision" if is_collision else "visual")
      q = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")
      rpy = Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")
      ET.SubElement(vis, "origin", xyz=geom.get("pos", "0 0 0"), rpy=numbers(rpy))
      geometry = ET.SubElement(vis, "geometry")
      kind = geom.get("type")
      size = np.fromstring(geom.get("size", ""), sep=" ")
      if kind == "mesh":
        ET.SubElement(geometry, "mesh", filename=f"package://{package}/new_meshes/{geom.get('mesh')}.stl")
      elif kind == "box":
        ET.SubElement(geometry, "box", size=numbers(2 * size))
      elif kind == "cylinder":
        ET.SubElement(geometry, "cylinder", radius=numbers(size[0]), length=numbers(2 * size[1]))
      elif kind == "sphere":
        ET.SubElement(geometry, "sphere", radius=numbers(size[0]))
      if not is_collision:
        material = ET.SubElement(vis, "material", name=geom.get("name"))
        ET.SubElement(material, "color", rgba=geom.get("rgba"))
    # Flatten the four MJCF corner blocks into the fixed URDF pad link.
    for child in body.findall("body"):
      block = child.find("geom[@type='box']")
      if block is None:
        continue
      col = ET.SubElement(link, "collision", name=block.get("name"))
      ET.SubElement(col, "origin", xyz=child.get("pos"), rpy="0 0 0")
      geometry = ET.SubElement(col, "geometry")
      ET.SubElement(geometry, "box", size=numbers(2 * np.fromstring(block.get("size"), sep=" ")))
    joint = ET.Element("joint", name=name, type="fixed")
    side = name.split("_")[0]
    parent = f"{side}_knee" if "foot" in name else f"{side}_elbow_pitch"
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=name)
    q = np.fromstring(body.get("quat"), sep=" ")
    rpy = Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")
    ET.SubElement(joint, "origin", xyz=body.get("pos"), rpy=numbers(rpy))
    for tag, node in (("link", link), ("joint", joint)):
      replacement = element_text(node, 1)
      if re.search(rf'<{tag}\s+name="{name}"', text):
        text = replace_named(text, tag, name, replacement)
      else:
        text = text.replace("</robot>", "  " + replacement + "\n</robot>")
  for side, info in forearms.items():
    name = f"{side}_elbow_pitch"
    link = ET.fromstring(text).find(f"link[@name='{name}']")
    link.find("visual/geometry/mesh").set("filename", f"package://{package}/new_meshes/{side}_forearm_no_hand.stl")
    inertial = link.find("inertial")
    inertial.find("origin").set("xyz", numbers(info["inertial"]["com_m"]))
    matrix = np.array(info["inertial"]["inertia_com_kg_m2"])
    vals = matrix[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    inertial.find("inertia").attrib.update(dict(zip(("ixx", "iyy", "izz", "ixy", "ixz", "iyz"), numbers(vals).split())))
    col = link.find("collision")
    col.find("origin").set("xyz", "0 0 -0.128")
    cylinder = col.find("geometry/cylinder")
    cylinder.set("radius", "0.021")
    cylinder.set("length", "0.120")
    text = replace_named(text, "link", name, element_text(link, 1))
  text = re.sub(r"(?m)^[ \t]+$", "", text)
  text = re.sub(r"\n{3,}", "\n\n", text)
  path.write_text(text)


def main(argv: list[str] | None = None):
  argparse.ArgumentParser(description=__doc__).parse_args(argv)
  outputs, pad_inertia, mount_inertia, assembly = normalize_geometry()
  path = ASSET / "xml/phybot_c2.xml"
  text = path.read_text()
  original = ET.fromstring(text)
  forearms = forearms_without_hands(original)
  pads, mounts, origins = [], [], {}
  for name in SPINES:
    old = original.find(f".//body[@name='{name}']")
    quat = np.fromstring(old.get("quat"), sep=" ")
    if "hand" in name:
      side = name.split("_")[0]
      end_z = forearms[side]["forearm_end_z_m"]
      rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
      # Mate the CAD sleeve stop to the physical end of the retained tube.
      # Its axis is concentric with that tube; B need not lie on that axis.
      end = np.array([0, 0, end_z])
      stop = np.asarray(assembly.features["sleeve_shoulder_m"])
      pos = end + rotation.apply(assembly.pivot - stop)
      origins[side] = {
        "forearm_end_z_m": end_z,
        "ball_center_in_elbow_m": pos.tolist(),
        "contact_center_in_elbow_m": (pos + rotation.apply(assembly.contact_center)).tolist(),
      }
      mounts.append(mount_body(side, pos, quat, mount_inertia, outputs, assembly))
    else:
      # Preserve the OLD leg pivot, including its original normal offset.
      # Once migrated, the body origin and the explicit B site coincide.
      pos = np.fromstring(old.get("pos"), sep=" ")
      pivot_site = old.find(f"site[@name='{name}_ball_center']")
      if pivot_site is None:
        sphere_pos = np.array([np.fromstring(b.get("pos"), sep=" ") for b in old.findall("body")])
        pivot = sphere_pos.mean(axis=0) + [0, -0.012, 0]
        pos += Rotation.from_quat(quat[[1, 2, 3, 0]]).apply(pivot)
      origins[name] = {"ball_center_in_knee_m": pos.tolist()}
    body = pad_body(name, pos, quat, pad_inertia, outputs, assembly)
    pads.append(body)
    text = replace_named(text, "body", name, element_text(body, 7 if "foot" in name else 9))
  for body in mounts:
    name = body.get("name")
    block = element_text(body, 9)
    if re.search(rf'<body\s+name="{name}"', text):
      text = replace_named(text, "body", name, block)
    else:
      hand = f"{name.split('_')[0]}_hand_spine"
      text = text.replace(f'<body name="{hand}"', block + f'\n                  <body name="{hand}"', 1)
  for name in outputs:
    mesh_name = f"microspine_{name}"
    text = re.sub(rf'^\s*<mesh name="{mesh_name}"[^>]*/>\s*$', "", text, flags=re.M)
    text = text.replace("  </asset>", f'    <mesh name="{mesh_name}" file="{mesh_name}.stl"/>\n  </asset>', 1)
  # Legacy long stalk meshes are no longer referenced by pad visuals.
  for name in SPINES:
    text = re.sub(rf'^\s*<mesh name="{name}"[^>]*/>\s*$', "", text, flags=re.M)
  for side, info in forearms.items():
    name = f"{side}_elbow_pitch"
    body = ET.fromstring(text).find(f".//body[@name='{name}']")
    body.remove(body.find("inertial"))
    add_inertial(body, info["inertial"])
    col = body.find(f"geom[@name='{name}_collision']")
    col.set("pos", "0 0 -0.128")
    col.set("size", "0.021 0.060")
    if body.find(f"geom[@name='{name}_motor_collision']") is None:
      motor = collision(body, f"{name}_motor_collision", [0.040], [0, 0, 0], "capsule")
      motor.attrib.pop("pos")
      motor.set("fromto", "0 0 -0.040 0 0 0")
    text = replace_named(text, "body", name, element_text(body, 8))
    text = re.sub(rf'<mesh name="{name}"[^>]*/>', f'<mesh name="{name}" file="{side}_forearm_no_hand.stl"/>', text)
  ET.fromstring(text)
  text = re.sub(r"(?m)^[ \t]+$", "", text)
  text = re.sub(r"\n{3,}", "\n\n", text)
  path.write_text(text)
  sync_urdf(pads, mounts, forearms)
  manifest = {
    "source": "data/cad_model/assembly_1/urdf/assembly_1.urdf",
    "source_sha256": hashlib.sha256((CAD / "urdf/assembly_1.urdf").read_bytes()).hexdigest(),
    "source_step": "data/cad_model/Assembly 1.step",
    "source_step_sha256": hashlib.sha256(CAD_STEP.read_bytes()).hexdigest(),
    "geometry_mode": "rigid_cad_import",
    "cad_features_in_connector_frame": assembly.features,
    "pla_density_kg_m3": PLA_DENSITY,
    "density_source": "https://www.prusa3d.com/file/370474/prusament-pla-technical-data-sheet.pdf",
    "inertia_assumption": "Sum of the unmodified CAD solids at solid-PLA density; hardware absent from the CAD is not added or calibrated.",
    "contact_center_in_pad_m": assembly.contact_center.tolist(),
    "unit_centers_in_pad_m": (assembly.unit_centres - assembly.pivot).tolist(),
    "unit_reference": "Frontmost printed unit plane, centered laterally and vertically; not a measured spine-tip offset.",
    "contact_proxy": {"type": "box", "count_per_pad": 4, "unit_local_bounds_m": outputs["unit"].bounds.tolist()},
    "forearm_extension_m": float(assembly.pivot[2] - assembly.features["sleeve_shoulder_m"][2]),
    "cad_adjustment": "None: all CAD solids retain their original shape and relative pose. The assembly is rigidly seated on the forearm tube at the CAD sleeve stop.",
    "source_solid_count": len(assembly.parts),
    "part_bounds_in_pad_m": {name: mesh.bounds.tolist() for name, mesh in outputs.items()},
    "origins": origins,
    "bare_forearms": forearms,
    "forearm_inertia_assumption": "Retain original actuator/tube link mass; reestimate COM and tensor over retained solid geometry, excluding the dummy hand shape.",
    "moving_pad": pad_inertia,
    "forearm_mount": mount_inertia,
  }
  (ASSET / "microspine_cad_geometry.json").write_text(json.dumps(manifest, indent=2) + "\n")
  print(json.dumps({"moving_pad": pad_inertia, "forearm_mount": mount_inertia, "origins": origins}, indent=2))


if __name__ == "__main__":
  main()
