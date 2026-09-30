"""Render a synchronized pair of PHYBot-style and original ToddlerBot views.

Both panels replay the same kinematic reference without text overlays. Audit
results and the distinction from physics replay are kept in the task README.
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from scr_phybot_mjab.assets.toddlerbot.toddlerbot_constants import UNIT_NAMES, get_spec
from scr_phybot_mjab.tools.climbing_reference.toddlerbot import PIVOT, REFERENCE_PATH, load_reference, wall_rotation


def build_scene(angle, width, height):
    """Use the wall, backdrop and lights from render_climb_video.py."""
    spec = get_spec()
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.quality.shadowsize = 4096
    spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1]
    spec.add_texture(
        name="render_sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0, 0, 0],
        width=32,
        height=512,
    )
    r = wall_rotation(angle)
    wall = spec.worldbody.add_body(name="render_wall", pos=PIVOT, quat=r.as_quat(scalar_first=True))
    wall.add_geom(
        name="render_wall_box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.04, 0, 0.55],
        size=[0.04, 0.40, 0.55],
        rgba=[0.72, 0.68, 0.60, 1],
        contype=0,
        conaffinity=0,
    )
    spec.add_texture(
        name="render_wall_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.72, 0.68, 0.60],
        rgb2=[0.63, 0.59, 0.52],
        width=300,
        height=300,
    )
    material = spec.add_material(name="render_wall_mat", texrepeat=[8, 8], texuniform=True)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "render_wall_tex"
    wall.add_geom(
        name="render_wall_face",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[-0.0005, 0, 0.55],
        quat=[2**-0.5, 0, -(2**-0.5), 0],
        size=[0.55, 0.40, 0.1],
        material="render_wall_mat",
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[10, 10, 0.1],
        rgba=[0.35, 0.40, 0.45, 1],
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_light(
        name="render_overhead",
        pos=[0, 0, 1.3],
        dir=[0, 0, -1.3],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[1, 1, 1],
        specular=[0.1, 0.1, 0.1],
        castshadow=False,
    )
    spec.worldbody.add_light(name="render_key", pos=[-1.5, -1.0, 3.0], dir=[0.6, 0.3, -0.75], castshadow=True)
    return spec.compile()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--output", type=Path, default=Path("docs/toddlerbot_climb/reference.mp4"))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--azimuth", type=float, default=25.0, help="Right panel's original camera azimuth")
    parser.add_argument("--width", type=int, default=960, help="Width of each view; video width is twice this")
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    ref = load_reference(args.reference)
    angle = float(ref["wall_angle_deg"])
    r = wall_rotation(angle)
    m = build_scene(angle, args.width, args.height)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, width=args.width, height=args.height)
    phybot_camera = mujoco.MjvCamera()
    phybot_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    phybot_camera.azimuth = 55.0
    phybot_camera.elevation = -12.0
    # Scale the viewing distance to the smaller robot while keeping C2's angles.
    phybot_camera.distance = 1.25
    lookat_xy = ref["qpos"][:, :2].mean(axis=0)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = r.apply([-0.10, 0, 0.43]) + PIVOT
    camera.distance = 1.25
    camera.azimuth = args.azimuth
    camera.elevation = -10
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    option.sitegroup[:] = 0

    def draw(i):
        d.qpos[:] = ref["qpos"][i]
        mujoco.mj_forward(m, d)
        for name, on in zip(UNIT_NAMES, ref["engaged"][i]):
            m.geom_rgba[m.geom(name + "_visual").id] = [0.1, 0.9, 0.25, 1] if on else [1, 0.25, 0.05, 1]
        # Match C2's height-following camera on the left; preserve the original
        # fixed ToddlerBot camera on the right. Both use this same FK state.
        phybot_camera.lookat[:] = (*lookat_xy, ref["qpos"][i, 2])
        renderer.update_scene(d, camera=phybot_camera, scene_option=option)
        left = renderer.render().copy()
        renderer.update_scene(d, camera=camera, scene_option=option)
        right = renderer.render().copy()
        return np.concatenate((left, right), axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".png":
        imageio.imwrite(args.output, draw(min(args.frame, len(ref["qpos"]) - 1)))
    else:
        with imageio.get_writer(args.output, fps=float(ref["fps"]), codec="libx264", quality=7) as writer:
            for i in range(len(ref["qpos"])):
                writer.append_data(draw(i))
    renderer.close()
    print(args.output)


if __name__ == "__main__":
    main()
