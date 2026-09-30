"""Compare the compiled robot's geometry with the original CAD solids."""

from __future__ import annotations

import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import cKDTree
import trimesh

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  get_spec,
  PHYBOT_C2_XML,
  PHYBOT_C2_PAD_GEOMETRY,
  SPINE_BODY_NAMES,
)
from scr_phybot_mjab.tools.microspine.cad import load_cad_pad_assembly, CAD_DIRECTORY, CAD_STEP


def audit_rigid_cad_asset() -> dict:
  cad = load_cad_pad_assembly()
  model = get_spec().compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  raw_plate = next(m for _, f, m in cad.parts if f == "Part_1_4.gltf")
  raw_mount = next(m for _, f, m in cad.parts if f == "Part_1_6.gltf")
  raw_brackets = trimesh.util.concatenate(
    [m for _, f, m in cad.parts if f not in ("Part_1_4.gltf", "Part_1_6.gltf", "Part_1_1.gltf")]
  )
  errors = {}

  def compare(name, raw, body):
    gid = model.geom(name).id
    mesh = model.geom_dataid[gid]
    begin, length = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[begin : begin + length] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
    bid = model.body(body).id
    expected = (raw.vertices - cad.pivot) @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
    # Both directions catch missing pieces as well as shape/placement changes.
    errors[name] = max(
      float(cKDTree(vertices).query(expected)[0].max()), float(cKDTree(expected).query(vertices)[0].max())
    )

  for pad in SPINE_BODY_NAMES:
    compare(pad + "_visual", raw_plate, pad)
    compare(pad + "_brackets_visual", raw_brackets, pad)
    for i, raw in enumerate(cad.unit_parts):
      row, col = divmod(i, 4)
      compare(f"{pad}_unit_{row}_{col}_visual", raw, pad)
  mounting_errors = {}
  for side in ("left", "right"):
    body = f"{side}_forearm_pad_mount"
    compare(body + "_visual", raw_mount, body)
    bid = model.body(body).id
    arm = model.body(f"{side}_elbow_pitch").id
    stop = np.asarray(cad.features["sleeve_shoulder_m"])
    actual = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ (stop - cad.pivot)
    in_arm = data.xmat[arm].reshape(3, 3).T @ (actual - data.xpos[arm])
    end = PHYBOT_C2_PAD_GEOMETRY["bare_forearms"][side]["forearm_end_z_m"]
    mounting_errors[side] = float(np.linalg.norm(in_arm - [0, 0, end]))
  sources = [CAD_STEP, CAD_DIRECTORY / "urdf/assembly_1.urdf", *sorted((CAD_DIRECTORY / "meshes").glob("*.gltf"))]
  max_error = max(errors.values())
  return {
    "geometry_mode": "rigid_cad_import",
    "passed": max_error < 1e-7 and max(mounting_errors.values()) < 1e-8,
    "tolerance_m": 1e-7,
    "max_compiled_vertex_error_m": max_error,
    "compiled_vertex_errors_m": errors,
    "sleeve_stop_alignment_errors_m": mounting_errors,
    "source_solid_count": len(cad.parts),
    "cad_features": cad.features,
    "B_to_C_m": cad.contact_center.tolist(),
    "source_backplate_extents_xyz_m": raw_plate.extents.tolist(),
    "source_connector_extents_xyz_m": raw_mount.extents.tolist(),
    "robot_mass_kg": float(model.body_mass.sum()),
    "provisional_pla_moving_pad_mass_kg": PHYBOT_C2_PAD_GEOMETRY["moving_pad"]["mass_kg"],
    "provisional_pla_mount_mass_kg": PHYBOT_C2_PAD_GEOMETRY["forearm_mount"]["mass_kg"],
    "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    "asset_sha256": hashlib.sha256(PHYBOT_C2_XML.read_bytes()).hexdigest(),
  }


def render_cad_comparison(output: Path):
  """Render source and imported meshes at identical orthographic scale."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  cad = load_cad_pad_assembly()
  root = ET.parse(PHYBOT_C2_XML).getroot()
  meshes = {m.get("name"): m.get("file") for m in root.findall("asset/mesh")}
  current = []
  for name in ("left_hand_spine", "left_forearm_pad_mount"):
    body = root.find(f".//body[@name='{name}']")
    for geom in body.findall("geom[@type='mesh']"):
      mesh = trimesh.load_mesh(PHYBOT_C2_XML.parent.parent / "new_meshes" / meshes[geom.get("mesh")])
      mesh.apply_translation(np.fromstring(geom.get("pos", "0 0 0"), sep=" "))
      current.append(mesh)

  def render(parts, center):
    xml = ET.Element("mujoco")
    vis = ET.SubElement(xml, "visual")
    ET.SubElement(vis, "global", orthographic="true", fovy=".255", ipd="0", offwidth="700", offheight="1050")
    ET.SubElement(vis, "headlight", ambient=".4 .4 .4", diffuse=".55 .55 .55", specular="0 0 0")
    asset, world = ET.SubElement(xml, "asset"), ET.SubElement(xml, "worldbody")
    ET.SubElement(
      asset,
      "texture",
      type="skybox",
      builtin="flat",
      rgb1=".975 .98 .985",
      rgb2=".975 .98 .985",
      width="16",
      height="16",
    )
    files = {}
    for i, raw in enumerate(parts):
      mesh = raw.copy()
      mesh.apply_translation(-center)
      mesh.vertices *= [1, -1, -1]
      name = f"part{i}"
      files[name + ".stl"] = mesh.export(file_type="stl")
      ET.SubElement(asset, "mesh", name=name, file=name + ".stl")
      ET.SubElement(world, "geom", type="mesh", mesh=name, rgba=cad.colour, contype="0", conaffinity="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"), assets=files)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    images = []
    with mujoco.Renderer(model, width=700, height=1050) as renderer:
      for azimuth in (270, 90):
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation, camera.distance = azimuth, 0, 0.4
        camera.lookat[:] = [0, 0, 0.060]
        renderer.update_scene(data, camera=camera)
        assert renderer.scene.camera[0].orthographic == 1
        images.append(renderer.render().copy())
    return images

  raw_images = render([m for _, _, m in cad.parts], cad.unit_centres.mean(axis=0))
  current_images = render(current, cad.contact_center)
  fig, axes = plt.subplots(1, 4, figsize=(16, 7.4), facecolor="#f8fafb")
  for ax, im, title in zip(
    axes,
    [raw_images[0], current_images[0], raw_images[1], current_images[1]],
    ["Original CAD | front", "Imported asset | front", "Original CAD | back", "Imported asset | back"],
  ):
    ax.imshow(im)
    ax.axis("off")
    ax.set_title(title, fontsize=12)
  fig.suptitle("Rigid CAD import: same shape, assembly pose, scale and material", fontsize=16)
  fig.text(
    0.5,
    0.028,
    "Backplate: 126 x 167.50 mm   |   Connector: 110 mm   |   Unit pitch: 28 x 31.5 mm",
    ha="center",
    fontsize=12,
  )
  fig.subplots_adjust(left=0.008, right=0.992, top=0.91, bottom=0.065, wspace=0.02)
  fig.savefig(output, dpi=175, facecolor=fig.get_facecolor())
  plt.close(fig)
