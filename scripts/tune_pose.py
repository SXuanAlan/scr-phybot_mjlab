"""Interactive pose-tuning tool (viser-based).

Opens a viser server in the browser with a chosen robot (optionally placed
in the climb-wall scene) and exposes GUI sliders for:
  - Robot base position (x, y, z)
  - Every articulated joint

No RL, no parallel envs — just a single mj_data that gets updated from
the sliders and re-rendered. Works headlessly over SSH (unlike the native
MuJoCo viewer, which needs a local GL context).

Run:
  python scripts/tune_pose.py                    # default: phybot_c2

Then open the printed ViserServer URL in a browser.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene

from mjlab.entity import Entity

from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import (
  phybot_c2_climb_wall_env_cfg,
)

ROBOT_CHOICES = ("phybot_c2",)


def _entities_for_robot(robot_name: str) -> list[tuple[str, Entity]]:
  """Build the list of (attach_name, Entity) to include in the scene."""
  if robot_name == "phybot_c2":
    cfg = phybot_c2_climb_wall_env_cfg(play=True)
    return [
      ("robot", Entity(cfg.scene.entities["robot"])),
      ("wall", Entity(cfg.scene.entities["wall"])),
    ]
  raise ValueError(f"unknown robot {robot_name!r}; choices: {ROBOT_CHOICES}")


def _build_scene_spec(robot_name: str) -> mujoco.MjSpec:
  """Compose the selected robot (+ wall if applicable) into a single spec."""
  entities = _entities_for_robot(robot_name)

  scene = mujoco.MjSpec()
  scene.worldbody.add_geom(
    name="ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(0, 0, 0.01),
    rgba=(0.6, 0.6, 0.65, 1.0),
    contype=0,
    conaffinity=0,
  )

  key_qpos: list[np.ndarray] = []
  key_ctrl: list[np.ndarray] = []
  for name, ent in entities:
    if ent.spec.keys:
      init_key = None
      for k in ent.spec.keys:
        if k.name == "init_state":
          init_key = k
          break
      if init_key is None:
        init_key = ent.spec.keys[0]
      key_qpos.append(np.array(init_key.qpos))
      key_ctrl.append(np.array(init_key.ctrl))
      for k in list(ent.spec.keys):
        ent.spec.delete(k)
    frame = scene.worldbody.add_frame()
    scene.attach(ent.spec, prefix=f"{name}/", frame=frame)

  if key_qpos:
    scene.add_key(
      name="init_state",
      qpos=np.concatenate(key_qpos).tolist(),
      ctrl=np.concatenate(key_ctrl).tolist() if key_ctrl else [],
    )
  return scene


def _joint_limits(model: mujoco.MjModel, jnt_id: int) -> tuple[float, float]:
  """Return (low, high) for a joint — falls back to ±π for unlimited hinges."""
  if model.jnt_limited[jnt_id]:
    return float(model.jnt_range[jnt_id, 0]), float(model.jnt_range[jnt_id, 1])
  jnt_type = model.jnt_type[jnt_id]
  if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
    return (-np.pi, np.pi)
  if jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
    return (-1.0, 1.0)
  return (-1.0, 1.0)


def main() -> None:
  parser = argparse.ArgumentParser(description="Interactive pose tuner.")
  parser.add_argument(
    "--robot",
    choices=ROBOT_CHOICES,
    default="phybot_c2",
    help="Robot to display (default: phybot_c2).",
  )
  args = parser.parse_args()

  spec = _build_scene_spec(args.robot)
  model = spec.compile()
  data = mujoco.MjData(model)

  key_id = model.key("init_state").id
  mujoco.mj_resetDataKeyframe(model, data, key_id)
  mujoco.mj_forward(model, data)

  # Snapshot the initial qpos for reset.
  init_qpos = data.qpos.copy()

  # Print the values the sliders WILL be initialized to, so if the browser
  # shows something different you know the old viser tab is stale.
  print("[init] robot root pos =", init_qpos[:3])
  for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    nm = model.joint(j).name
    if not nm.startswith("robot/"):
      continue
    print(f"[init] {nm.replace('robot/', ''):26} = {init_qpos[int(model.jnt_qposadr[j])]:.4f}")
  init_mocap_pos = data.mocap_pos.copy() if model.nmocap else None
  init_mocap_quat = data.mocap_quat.copy() if model.nmocap else None

  server = viser.ViserServer(label="pose-tuner")
  scene = ViserMujocoScene(server=server, mj_model=model, num_envs=1)
  scene.env_idx = 0

  # Locate the robot's free joint (root pose) and list articulated joints.
  free_jnt_id: int | None = None
  articulated_jnt_ids: list[int] = []
  for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      free_jnt_id = j
    else:
      articulated_jnt_ids.append(j)

  def _refresh() -> None:
    mujoco.mj_forward(model, data)
    scene.update_from_mjdata(data)

  # -------------- GUI --------------
  with server.gui.add_folder("Robot base position"):
    if free_jnt_id is not None:
      qadr = int(model.jnt_qposadr[free_jnt_id])
      sliders_xyz: list[viser.GuiSliderHandle] = []
      for axis, offset in zip(("x", "y", "z"), (0, 1, 2)):
        s = server.gui.add_slider(
          axis,
          min=-5.0,
          max=10.0,
          step=0.01,
          initial_value=float(init_qpos[qadr + offset]),
        )

        def _cb(_ev, _s=s, _off=offset, _adr=qadr) -> None:
          data.qpos[_adr + _off] = _s.value
          _refresh()

        s.on_update(_cb)
        sliders_xyz.append(s)
    else:
      server.gui.add_markdown("(robot has no free joint — base is fixed)")

  joint_sliders: list[tuple[viser.GuiSliderHandle, int]] = []
  with server.gui.add_folder("Robot joints"):
    for jid in articulated_jnt_ids:
      # Robot joints live under the "robot/" prefix after scene attach.
      name = model.joint(jid).name
      if not name.startswith("robot/"):
        continue
      qadr = int(model.jnt_qposadr[jid])
      lo, hi = _joint_limits(model, jid)
      s = server.gui.add_slider(
        name.replace("robot/", ""),
        min=float(lo),
        max=float(hi),
        step=0.005,
        initial_value=float(init_qpos[qadr]),
      )

      def _cb(_ev, _s=s, _adr=qadr) -> None:
        data.qpos[_adr] = _s.value
        _refresh()

      s.on_update(_cb)
      joint_sliders.append((s, qadr))

  with server.gui.add_folder("Actions"):
    reset_btn = server.gui.add_button("Reset to init pose")
    dump_btn = server.gui.add_button("Print qpos / joint dict")

    @reset_btn.on_click
    def _(_ev) -> None:
      data.qpos[:] = init_qpos
      data.qvel[:] = 0.0
      if init_mocap_pos is not None:
        data.mocap_pos[:] = init_mocap_pos
        data.mocap_quat[:] = init_mocap_quat
      # Sync sliders.
      if free_jnt_id is not None:
        qadr = int(model.jnt_qposadr[free_jnt_id])
        for i, s in enumerate(sliders_xyz):
          s.value = float(init_qpos[qadr + i])
      for s, adr in joint_sliders:
        s.value = float(init_qpos[adr])
      _refresh()

    @dump_btn.on_click
    def _(_ev) -> None:
      print("=" * 60)
      print("Current qpos (full):")
      print(np.array2string(data.qpos, precision=4, separator=", "))
      if free_jnt_id is not None:
        qadr = int(model.jnt_qposadr[free_jnt_id])
        print(
          f"\nrobot base pos = ({data.qpos[qadr]:.4f}, "
          f"{data.qpos[qadr + 1]:.4f}, {data.qpos[qadr + 2]:.4f})"
        )
        print(
          f"robot base quat = ({data.qpos[qadr + 3]:.4f}, "
          f"{data.qpos[qadr + 4]:.4f}, {data.qpos[qadr + 5]:.4f}, "
          f"{data.qpos[qadr + 6]:.4f})"
        )
      print("\njoint_pos dict (for env_cfg):")
      print("{")
      for _, adr in joint_sliders:
        jid = next(j for j in range(model.njnt) if int(model.jnt_qposadr[j]) == adr)
        name = model.joint(jid).name.replace("robot/", "")
        print(f'  "{name}": {data.qpos[adr]:.4f},')
      print("}")
      print("=" * 60, flush=True)

  _refresh()

  print("=" * 60)
  print(f"Pose Tuner (viser) — robot={args.robot}")
  print("=" * 60)
  print("Open the URL printed above in a browser.")
  print("Adjust sliders; click 'Print qpos / joint dict' to copy back to env_cfg.")
  print("Ctrl+C to quit.")
  print("=" * 60)

  try:
    while True:
      time.sleep(1.0)
  except KeyboardInterrupt:
    print("\nshutting down")
    server.stop()


if __name__ == "__main__":
  main()
