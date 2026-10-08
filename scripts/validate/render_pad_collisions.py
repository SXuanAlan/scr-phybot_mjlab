"""Render the current MJCF pad and its four block contact proxies."""

from __future__ import annotations

import copy
import os
from xml.etree import ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from PIL import Image, ImageDraw, ImageFont

from scr_phybot_mjab.tools.paths import REPOSITORY_ROOT


def main():
  root = REPOSITORY_ROOT
  asset = root / "src/scr_phybot_mjab/assets/phybot_c2"
  original = ET.parse(asset / "xml/phybot_c2.xml")
  pad = copy.deepcopy(original.find(".//body[@name='left_hand_spine']"))
  pad.set("pos", "0 0 0")
  pad.set("quat", "1 0 0 0")
  xml = ET.Element("mujoco")
  assets = ET.SubElement(xml, "asset")
  for name in sorted({g.get("mesh") for g in pad.iter("geom") if g.get("mesh")}):
    ET.SubElement(assets, "mesh", name=name, file=str(asset / "new_meshes" / (name + ".stl")))
  ET.SubElement(xml, "worldbody").append(pad)
  spec = mujoco.MjSpec.from_string(ET.tostring(xml, encoding="unicode"))
  spec.visual.global_.offwidth = 1000
  spec.visual.global_.offheight = 1000
  spec.visual.global_.fovy = 32
  spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
  spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
  spec.worldbody.add_light(pos=[-0.2, -0.4, 0.5], dir=[0.2, 0.4, -0.5], diffuse=[0.35, 0.35, 0.35])
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  option = mujoco.MjvOption()
  option.sitegroup[:] = 0
  for i in range(model.ngeom):
    if "_collision" in model.geom(i).name:
      model.geom_rgba[i, 3] = 0
  camera = mujoco.MjvCamera()
  camera.azimuth, camera.elevation, camera.distance = 65, -8, 0.40
  camera.lookat[:] = [0, -0.020, 0.050]
  with mujoco.Renderer(model, width=1000, height=1000) as renderer:
    renderer.update_scene(data, camera=camera, scene_option=option)
    cad = Image.fromarray(renderer.render())
    for i in range(model.ngeom):
      name = model.geom(i).name
      if "_spine_c_" in name:
        model.geom_rgba[i] = [0.15, 0.65, 0.95, 0.85]
      elif "_unit_" in name and any(f"_unit_{r}_{c}_" in name for r in (0, 2) for c in (0, 3)):
        model.geom_rgba[i, 3] = 0.0
    renderer.update_scene(data, camera=camera, scene_option=option)
    collision = Image.fromarray(renderer.render())
  canvas = Image.new("RGB", (2000, 1130), "#f3f5f7")
  canvas.paste(cad, (0, 85))
  canvas.paste(collision, (1000, 85))
  draw = ImageDraw.Draw(canvas)
  font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
  font = ImageFont.truetype(font_path, 25)
  draw.text((24, 20), "CAD unit geometry", font=font, fill="#25313d")
  draw.text((1024, 20), "4 corner box contacts (blue)", font=font, fill="#25313d")
  box = model.geom("left_hand_spine_c_0_0")
  c = model.site("left_hand_spine_contact_center").pos
  dimensions = " x ".join(f"{v:.2f}" for v in 2000 * box.size)
  draw.text(
    (24, 1091),
    f"Box: {dimensions} mm | Contact plane y = {1000 * c[1]:.2f} mm | Array: 4 x 3 units",
    font=font,
    fill="#25313d",
  )
  output = root / "docs/figures/phybot_c2_pad_box_collisions.png"
  canvas.save(output)
  print(output)


if __name__ == "__main__":
  main()
