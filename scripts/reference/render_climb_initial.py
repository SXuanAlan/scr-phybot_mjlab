"""Render a real, unstepped climbing reset and audit its contact geometry.

Uses play=True: vertical wall, reference frame zero, no pose jitter. Training
also samples other reference frames and wall angles. Rendering changes only
the copied host model's appearance; it does not move the robot to fit the wall.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer.model_sync import (
  VIEWER_MODEL_FIELDS,
  disable_model_sameframe_shortcuts,
  sync_model_fields,
)
from PIL import Image, ImageDraw, ImageFont

from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)
from scr_phybot_mjab.tools.paths import FIGURES_DIR

OUTPUT = FIGURES_DIR


def capture():
  cfg = phybot_c2_climb_wall_env_cfg(play=True)
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
  try:
    env.reset(seed=0)
    model = copy.deepcopy(env.sim.mj_model)
    disable_model_sameframe_shortcuts(model)
    fields = env.sim.expanded_fields & VIEWER_MODEL_FIELDS
    sync_model_fields(model, env.sim.model, fields, 0)
    data = mujoco.MjData(model)
    for name in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "eq_active"):
      getattr(data, name)[:] = getattr(env.sim.data, name)[0].cpu().numpy()
    mujoco.mj_forward(model, data)

    motion = env.command_manager.get_term("climb_motion")
    affordance = env.command_manager.get_term("microspine_affordance")
    face_centers = affordance.spine_field.corners_world()[0].mean(1).cpu().numpy()
    anchor_offsets = affordance._get_executor()._pad_local.cpu().numpy()
    microspine = cfg.commands["microspine_affordance"].microspine_cfg
    wall_id = model.geom(microspine.wall_geom_name).id
    wall_point = data.geom_xpos[wall_id]
    wall_normal = data.geom_xmat[wall_id].reshape(3, 3)[:, 2]
    report = {
      "task": "climb_wall",
      "configuration": "play=True; real reset; no simulation steps",
      "motion_file": cfg.commands["climb_motion"].motion_file,
      "reset_reference_frame": 0,
      "wall_point_world_m": wall_point.tolist(),
      "wall_normal_world": wall_normal.tolist(),
      "gap_sign": "positive = outside wall; negative = inside wall",
      "pads": {},
      "collision_geoms": [],
      "explicit_wall_pairs": [],
      "contacts_at_reset": [],
    }
    # Cross-check the host snapshot against the actual GPU reset kinematics.
    report["host_gpu_site_max_error_m"] = float(
      np.max(np.abs(data.site_xpos - env.sim.data.site_xpos[0].cpu().numpy()))
    )
    report["empirical_runtime"] = {
      "table": str(affordance.hold_judge.surface.path),
      "strength_draw_shape": list(affordance.unit_strength_quantile.shape),
      "loading_draw_shape": list(affordance.unit_loading_quantile.shape),
      "distinct_strength_draws_at_reset": int(affordance.unit_strength_quantile[0].unique().numel()),
      "unit_order_max_error_m": 0.0,
    }
    for i, ee in enumerate(microspine.ee_body_names):
      prefix = f"robot/{ee}"
      center = data.site_xpos[model.site(prefix + "_contact_center").id]
      unit_ids = [j for j in range(model.nsite) if (model.site(j).name or "").startswith(prefix + "_unit_")]
      unit_gaps = (data.site_xpos[unit_ids] - wall_point) @ wall_normal
      # The affordance's unit order is its first pad's site order.
      for k, unit_name in enumerate(affordance.unit_site_names):
        site = model.site(f"{prefix}_unit_{unit_name.rsplit('_unit_', 1)[1]}")
        offset = site.pos - anchor_offsets[i]
        expected = np.array([-offset[2], -offset[0]])
        actual = np.array([affordance.hold_judge.unit_x[k].item(), affordance.hold_judge.unit_y[k].item()])
        report["empirical_runtime"]["unit_order_max_error_m"] = max(
          report["empirical_runtime"]["unit_order_max_error_m"], float(np.max(np.abs(expected - actual)))
        )
      face = data.xmat[model.body(prefix).id].reshape(3, 3) @ np.asarray(microspine.face_local_axis)
      report["pads"][ee] = {
        "center_world_m": center.tolist(),
        "center_wall_gap_mm": float((center - wall_point) @ wall_normal * 1000),
        "unit_wall_gap_range_mm": [float(unit_gaps.min() * 1000), float(unit_gaps.max() * 1000)],
        "face_angle_to_wall_deg": float(np.rad2deg(np.arccos(np.clip(face @ -wall_normal, -1, 1)))),
        "reference_engaged": bool(motion.clip.ee_engaged[0, i].item()),
        "field_center_error_m": float(np.linalg.norm(face_centers[i] - center)),
        "weld_anchor_pad_local_m": anchor_offsets[i].tolist(),
      }
    for i in range(model.ngeom):
      name = model.geom(i).name or ""
      if "_collision" in name or "_spine_c_" in name:
        report["collision_geoms"].append(
          {
            "name": name,
            "contype": int(model.geom_contype[i]),
            "conaffinity": int(model.geom_conaffinity[i]),
          }
        )
    for i in range(model.npair):
      ids = (int(model.pair_geom1[i]), int(model.pair_geom2[i]))
      if wall_id in ids:
        report["explicit_wall_pairs"].append([model.geom(j).name for j in ids])
    for contact in data.contact:
      report["contacts_at_reset"].append(
        {
          "geoms": [model.geom(int(j)).name for j in contact.geom],
          "distance_mm": float(contact.dist * 1000),
        }
      )
    report["reference_collision_audit"] = audit_reference_collisions(
      model, cfg.commands["climb_motion"].motion_file, wall_id
    )
    return model, data, report
  finally:
    env.close()


def audit_reference_collisions(model, motion_file, wall_id):
  """All reference frames, actual task collision masks, native FK only."""
  clip = np.load(motion_file, allow_pickle=True).item()
  data = mujoco.MjData(model)
  root = int(model.joint("robot/floating_base_joint").qposadr[0])
  names = list(clip["joints_list"]) + list(clip.get("passive_joint_names", ()))
  ids = [int(model.joint("robot/" + name).qposadr[0]) for name in names]
  values = np.concatenate([clip["joint_positions"], clip["passive_joint_positions"]], axis=-1)
  unexpected = {}
  max_penetration = 0.0
  for frame in range(len(values)):
    data.qpos[root : root + 3] = clip["root_position"][frame]
    data.qpos[root + 3 : root + 7] = clip["root_quaternion"][frame, [3, 0, 1, 2]]
    data.qpos[ids] = values[frame]
    mujoco.mj_forward(model, data)
    for contact in data.contact:
      penetration = max(0.0, -float(contact.dist) * 1000)
      max_penetration = max(max_penetration, penetration)
      geoms = [model.geom(int(g)).name for g in contact.geom]
      intended = wall_id in contact.geom and any("_spine_c_" in name or name.endswith("_toe") for name in geoms)
      if not intended and penetration > 0.15:
        pair = " / ".join(geoms)
        previous = unexpected.get(pair)
        if previous is None or penetration > previous["penetration_mm"]:
          unexpected[pair] = dict(frame=frame, penetration_mm=penetration)
  return dict(
    frames=len(values),
    max_active_contact_penetration_mm=max_penetration,
    unexpected_penetrating_pairs=unexpected,
    collision_pass=not unexpected,
    scope="Actual task collision masks/pairs; disabled visual/chassis geometry excluded; no dynamics steps",
  )


def render(model, data, report):
  OUTPUT.mkdir(parents=True, exist_ok=True)
  width, height = 1100, 1400
  model.vis.global_.offwidth = width
  model.vis.global_.offheight = height
  model.vis.global_.fovy = 38
  model.vis.headlight.ambient[:] = [0.20, 0.20, 0.20]
  model.vis.headlight.diffuse[:] = [0.40, 0.40, 0.40]
  model.light_diffuse[:] = 0.25
  model.light_specular[:] = 0.05
  model.stat.extent = 3.0
  model.mat_reflectance[:] = 0.0
  for i in range(model.ngeom):
    name = model.geom(i).name or ""
    if "_collision" in name or "_spine_c_" in name or name.endswith("_toe"):
      model.geom_group[i] = 3
    if name in ("robot/left_ankle_roll_collision", "robot/right_ankle_roll_collision"):
      model.geom_group[i] = 1  # Original shoe meshes serve both roles.
    if name == "wall/wall_visual_geom":
      model.geom_rgba[i] = [0.70, 0.74, 0.77, 1.0]

  option = mujoco.MjvOption()
  option.geomgroup[3:] = 0
  option.sitegroup[:] = 0
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = [1.28, 0.0, 1.22]
  camera.distance = 3.25
  camera.elevation = -8
  with mujoco.Renderer(model, width=width, height=height) as renderer:

    def view(azimuth):
      camera.azimuth = azimuth
      renderer.update_scene(data, camera=camera, scene_option=option)
      return Image.fromarray(renderer.render())

    oblique = view(35)
    side = view(90)
    oblique.save(OUTPUT / "phybot_c2_climb_initial_oblique.png")
    side.save(OUTPUT / "phybot_c2_climb_initial_side.png")
    canvas = Image.new("RGB", (2 * width, height + 185), "#f3f5f7")
    canvas.paste(oblique, (0, 100))
    canvas.paste(side, (width, 100))
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 27)
    small = ImageFont.truetype(font_path, 21)
    draw.text((30, 15), "PHYBot C2 | Current climbing reset with updated assets", font=font, fill="#25313d")
    draw.text((30, 57), "Oblique view", font=small, fill="#4c5a68")
    draw.text((width + 30, 57), "Side view", font=small, fill="#4c5a68")
    draw.text(
      (30, height + 110),
      "New CAD reference frame 0 | Passive x gimbals at -3 deg stops | No physics steps",
      font=small,
      fill="#4c5a68",
    )
    gaps = report["pads"]
    hand = gaps["left_hand_spine"]["center_wall_gap_mm"]
    foot = gaps["left_foot_spine"]["center_wall_gap_mm"]
    draw.text(
      (30, height + 145),
      f"C-to-wall signed gap: hands {hand:+.1f} mm, feet {foot:+.1f} mm | Negative = wall penetration",
      font=small,
      fill="#9b412f",
    )
    canvas.save(OUTPUT / "phybot_c2_climb_initial.png")

  path = OUTPUT / "phybot_c2_climb_initial.audit.json"
  path.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report["pads"], indent=2))
  print("Host/GPU site error (m):", report["host_gpu_site_max_error_m"])
  print("Saved", OUTPUT / "phybot_c2_climb_initial.png")


def main(argv: list[str] | None = None):
  argparse.ArgumentParser(description=__doc__).parse_args(argv)
  render(*capture())


if __name__ == "__main__":
  main()
