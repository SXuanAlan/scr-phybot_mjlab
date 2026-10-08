"""Render the actual get_spec() robot in its nominal standing pose."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants import (
  PHYBOT_C2_STANDING_JOINT_POS,
  get_spec,
)
from scr_phybot_mjab.tools.paths import FIGURES_DIR


def main(argv: list[str] | None = None):
  argparse.ArgumentParser(description=__doc__).parse_args(argv)
  output = FIGURES_DIR
  output.mkdir(parents=True, exist_ok=True)
  width, height = 1100, 1400
  spec = get_spec()
  spec.visual.global_.offwidth = width
  spec.visual.global_.offheight = height
  spec.visual.global_.fovy = 38
  spec.visual.headlight.ambient = [0.20, 0.20, 0.20]
  spec.visual.headlight.diffuse = [0.40, 0.40, 0.40]
  for texture in spec.textures:
    if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
      texture.rgb1 = [0.88, 0.91, 0.94]
      texture.rgb2 = [0.96, 0.97, 0.98]
  for geom in spec.geoms:
    if "_collision" in (geom.name or ""):
      if geom.name in ("left_ankle_roll_collision", "right_ankle_roll_collision"):
        geom.group = 1  # These mesh geoms also supply the original shoe visual.
      else:
        geom.group = 3
  for light in spec.lights:
    light.diffuse = [0.25, 0.25, 0.25]
    light.specular = [0.05, 0.05, 0.05]
  spec.worldbody.add_geom(
    name="asset_display_floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[4, 4, 0.1],
    rgba=[0.86, 0.88, 0.90, 1],
    contype=0,
    conaffinity=0,
  )
  spec.worldbody.add_light(
    name="asset_key_light", pos=[-2, -3, 4], dir=[2, 3, -4], diffuse=[0.4, 0.4, 0.4], castshadow=True
  )
  model = spec.compile()
  data = mujoco.MjData(model)
  data.qpos[:7] = [0, 0, 0.726, 1, 0, 0, 0]
  for name, value in PHYBOT_C2_STANDING_JOINT_POS.items():
    data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
  mujoco.mj_forward(model, data)
  option = mujoco.MjvOption()
  option.geomgroup[3:] = 0
  option.sitegroup[:] = 0
  renderer = mujoco.Renderer(model, width=width, height=height)
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE

  def render(azimuth, elevation, distance, lookat):
    camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
    camera.lookat[:] = lookat
    renderer.update_scene(data, camera=camera, scene_option=option)
    return Image.fromarray(renderer.render())

  front = render(35, -12, 2.65, [0, 0, 0.69])
  back = render(145, -12, 2.65, [0, 0, 0.69])
  front.save(output / "phybot_c2_cad_front.png")
  back.save(output / "phybot_c2_cad_back.png")
  b = data.site_xpos[model.site("left_hand_spine_ball_center").id]
  c = data.site_xpos[model.site("left_hand_spine_contact_center").id]
  lookat = (b + c) / 2 + [0, 0, 0.035]
  pad_rotation = data.xmat[model.body("left_hand_spine").id].reshape(3, 3)
  face = pad_rotation @ np.array([0, -1, 0])
  # Pick the actual contact side from the live pad frame, rather than a
  # fixed world azimuth that previously displayed the back of the assembly.
  forward = -(face + [0, 0.18, 0.08])
  azimuth = np.degrees(np.arctan2(forward[1], forward[0]))
  elevation = np.degrees(np.arctan2(forward[2], np.linalg.norm(forward[:2])))
  detail = render(azimuth, elevation, 0.72, lookat)
  detail.save(output / "phybot_c2_cad_arm_detail.png")
  back_detail = render(160, -5, 0.72, lookat)
  back_detail.save(output / "phybot_c2_cad_arm_back_detail.png")
  canvas = Image.new("RGB", (2 * width, height + 100), "#f3f5f7")
  canvas.paste(front, (0, 90))
  canvas.paste(back, (width, 90))
  draw = ImageDraw.Draw(canvas)
  font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
  font = ImageFont.truetype(font_path, 28)
  small = ImageFont.truetype(font_path, 20)
  draw.text((32, 16), "PHYBot C2 | Updated microspine assets", fill="#25313d", font=font)
  draw.text((32, 54), "Front three-quarter view", fill="#4c5a68", font=small)
  draw.text((width + 32, 54), "Rear three-quarter view", fill="#4c5a68", font=small)
  canvas.save(output / "phybot_c2_cad_robot.png")
  renderer.close()
  print("Rendered", output / "phybot_c2_cad_robot.png")
  print("Detail", output / "phybot_c2_cad_arm_detail.png")
  print("Total robot mass (kg)", model.body_mass.sum())
  for side in ("left", "right"):
    name = f"{side}_hand_spine"
    print(
      name,
      "B world",
      data.site_xpos[model.site(name + "_ball_center").id],
      "C world",
      data.site_xpos[model.site(name + "_contact_center").id],
    )


if __name__ == "__main__":
  main()
