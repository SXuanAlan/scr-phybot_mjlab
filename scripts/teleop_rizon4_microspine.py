"""3Dconnexion and keyboard Cartesian teleop for the Rizon4 microspine task.

The MJLab task consumes a 7D absolute TCP pose action:
``[x, y, z, qw, qx, qy, qz]``. Input devices produce 6D deltas, and this
script maintains the accumulated Cartesian target.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
  sys.path.insert(0, str(_REPO_SRC))

import torch
import tyro

import scr_phybot_mjab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import apply_delta_pose
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native import keys

from scr_phybot_mjab.assets.rizon4.rizon4_constants import (
  RIZON4_MICROSPINE_TCP_SITE,
  RIZON4_TELEOP_POS_STEP,
  RIZON4_TELEOP_ROT_STEP,
)


@dataclass(frozen=True)
class TeleopConfig:
  task: str = "Mjlab-Microspine-Rizon4"
  viewer: Literal["auto", "native", "viser"] = "auto"
  input: Literal["auto", "spacemouse", "keyboard", "viser"] = "auto"
  spacemouse_source: Literal["auto", "hid", "bridge"] = "auto"
  device: str | None = None
  num_envs: int = 1
  frame_rate: float = 60.0
  pos_step: float = RIZON4_TELEOP_POS_STEP
  rot_step: float = RIZON4_TELEOP_ROT_STEP
  spacemouse_deadband: float = 0.03
  spacemouse_bridge_host: str = "127.0.0.1"
  spacemouse_bridge_port: int = 8765
  viser_host: str = "0.0.0.0"
  viser_port: int = 8080
  debug: bool = False
  """Re-enable the raw per-step [Microspine] SENSOR / TRANSITION prints."""
  hud_period: float = 0.5
  """Seconds between [microspine] status lines (transitions print instantly)."""
  target_leash: float = 0.002
  """Max distance [m] the IK target may lead the measured TCP. Caps the force
  an over-pressed/stuck target can inject through the anchor (offset x
  ~20 N/mm) and stops large stuck position errors from leaking into
  tangential grinding via the damped IK solve. 0 disables."""


SPACEMOUSE_VENDOR_ID = 0x256F
SPACEMOUSE_KNOWN_PRODUCTS = {
  0xC635: "SpaceMouse Compact",
}
SPACEMOUSE_SUPPORTED_PRODUCTS = {
  "SpaceMouse Compact",
  "SpaceMouse Wireless",
  "3Dconnexion Universal Receiver",
}
SPACEMOUSE_LEFT_BUTTON = 0x01
SPACEMOUSE_RIGHT_BUTTON = 0x02
SpaceMouseControlMode = Literal["position", "orientation"]
SPACEMOUSE_UDEV_HINT = """\
3Dconnexion HID device was found but could not be opened.
On Linux, add USB and hidraw udev rules and reconnect the device:

  sudo groupadd -f plugdev
  sudo usermod -aG plugdev $USER
  printf '%s\\n' \\
    'SUBSYSTEM=="usb", ATTR{idVendor}=="256f", MODE="0660", GROUP="plugdev", TAG+="uaccess"' \\
    'KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0660", GROUP="plugdev", TAG+="uaccess"' \\
    | sudo tee /etc/udev/rules.d/99-3dconnexion.rules
  sudo udevadm control --reload-rules
  sudo udevadm trigger

Then unplug/replug the SpaceMouse and start a new login shell.
"""


@dataclass
class CartesianCommand:
  delta: torch.Tensor
  reset_target: bool = False
  reset_orientation: bool = False
  target_pos: torch.Tensor | None = None
  target_quat: torch.Tensor | None = None


class CartesianDeltaInput:
  def consume_command(self) -> CartesianCommand:
    raise NotImplementedError

  def handle_key(self, key: int, *, pos_step: float, rot_step: float) -> bool:
    del key, pos_step, rot_step
    return False

  def request_target_reset(self) -> None:
    raise NotImplementedError

  def create_gui(self, server) -> None:
    del server

  def sync_target(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> None:
    del target_pos, target_quat

  def close(self) -> None:
    pass


def _spacemouse_product_name(device_info: dict) -> str:
  product = device_info.get("product_string") or ""
  if product:
    return product
  product_id = int(device_info.get("product_id") or 0)
  return SPACEMOUSE_KNOWN_PRODUCTS.get(product_id, f"3Dconnexion HID 0x{product_id:04x}")


def _is_spacemouse_device(device_info: dict) -> bool:
  product = device_info.get("product_string") or ""
  vendor_id = int(device_info.get("vendor_id") or 0)
  if product in SPACEMOUSE_SUPPORTED_PRODUCTS or "SpaceMouse" in product:
    return True
  return vendor_id == SPACEMOUSE_VENDOR_ID


def _spacemouse_packet_size(device_name: str) -> int:
  return 13 if "Universal Receiver" in device_name else 7


def _open_hid_device(hid_device, device_info: dict) -> None:
  path = device_info.get("path")
  try:
    if path is not None:
      hid_device.open_path(path)
    else:
      hid_device.open(device_info["vendor_id"], device_info["product_id"])
  except OSError as exc:
    name = _spacemouse_product_name(device_info)
    vid = int(device_info.get("vendor_id") or 0)
    pid = int(device_info.get("product_id") or 0)
    raise OSError(
      f"Found {name} ({vid:04x}:{pid:04x}) but hidapi could not open it.\n"
      f"{SPACEMOUSE_UDEV_HINT}"
    ) from exc


class KeyboardCartesianInput(CartesianDeltaInput):
  """Accumulate key presses into one-step 6D Cartesian deltas."""

  def __init__(self, action_shape: tuple[int, ...], device: torch.device | str):
    self._delta = torch.zeros((*action_shape[:-1], 6), device=device)
    self._target_reset_requested = False
    self._lock = threading.Lock()
    self._key_map = {
      keys.KEY_W: (0, 1.0),
      keys.KEY_S: (0, -1.0),
      keys.KEY_A: (1, 1.0),
      keys.KEY_D: (1, -1.0),
      keys.KEY_Q: (2, 1.0),
      keys.KEY_E: (2, -1.0),
      keys.KEY_I: (3, 1.0),
      keys.KEY_K: (3, -1.0),
      keys.KEY_J: (4, 1.0),
      keys.KEY_L: (4, -1.0),
      keys.KEY_U: (5, 1.0),
      keys.KEY_O: (5, -1.0),
    }

  def handle_key(self, key: int, *, pos_step: float, rot_step: float) -> bool:
    if key == keys.KEY_SPACE:
      self.request_target_reset()
      return True

    mapped = self._key_map.get(key)
    if mapped is None:
      return False

    dim, sign = mapped
    scale = pos_step if dim < 3 else rot_step
    with self._lock:
      self._delta[:, dim] += sign * scale
    return True

  def consume_command(self) -> CartesianCommand:
    with self._lock:
      delta = self._delta.clone()
      reset = self._target_reset_requested
      self._delta.zero_()
      self._target_reset_requested = False
    return CartesianCommand(delta=delta, reset_target=reset)

  def request_target_reset(self) -> None:
    with self._lock:
      self._delta.zero_()
      self._target_reset_requested = True


class SpaceMouseCartesianInput(CartesianDeltaInput):
  """Read 3Dconnexion SpaceMouse HID packets into position or orientation deltas."""

  def __init__(
    self,
    action_shape: tuple[int, ...],
    device: torch.device | str,
    *,
    pos_step: float,
    rot_step: float,
    deadband: float,
  ):
    try:
      import hid
    except ImportError as exc:
      raise RuntimeError(
        "3Dconnexion input requires hidapi. Install it in env_mjlab with: "
        "python -m pip install hidapi"
      ) from exc

    self._hid = hid
    self._device = hid.device()
    self._device_name = self._find_device()
    self._delta = torch.zeros((*action_shape[:-1], 6), device=device)
    self._target_reset_requested = False
    self._orientation_reset_requested = False
    self._control_mode: SpaceMouseControlMode = "position"
    self._button_state = 0
    self._pos_step = pos_step
    self._rot_step = rot_step
    self._deadband = deadband
    self._lock = threading.Lock()
    self._stop = threading.Event()
    self._thread = threading.Thread(
      target=self._read_loop,
      name="rizon4-spacemouse",
      daemon=True,
    )
    self._thread.start()

  def _find_device(self) -> str:
    last_open_error: OSError | None = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
      for device_info in self._hid.enumerate():
        if not _is_spacemouse_device(device_info):
          continue
        product = _spacemouse_product_name(device_info)
        self._device.close()
        try:
          _open_hid_device(self._device, device_info)
          return product
        except OSError as exc:
          last_open_error = exc
      time.sleep(0.2)
    if last_open_error is not None:
      raise last_open_error
    raise OSError("No 3Dconnexion SpaceMouse HID device found.")

  def _read_loop(self) -> None:
    packet_size = _spacemouse_packet_size(self._device_name)
    while not self._stop.is_set():
      try:
        data = self._device.read(packet_size, 50)
      except OSError:
        break
      if data:
        self._update_from_packet(data)

  def _scaled(self, value: float, scale: float) -> float:
    if abs(value) < self._deadband:
      return 0.0
    return value * scale

  def _toggle_mode(self) -> None:
    self._control_mode = "orientation" if self._control_mode == "position" else "position"
    self._delta.zero_()
    print(f"SpaceMouse mode: {self._control_mode}")

  def _reset_orientation(self) -> None:
    self._delta.zero_()
    self._orientation_reset_requested = True

  def _update_from_packet(self, data: list[int]) -> None:
    packet_type = data[0]
    with self._lock:
      if packet_type == 1:
        axes = (
          -_convert_buffer(data[3], data[4]),
          -_convert_buffer(data[1], data[2]),
          -_convert_buffer(data[5], data[6]),
        )
        if self._control_mode == "position":
          self._delta[:, 0] = self._scaled(axes[0], self._pos_step)
          self._delta[:, 1] = self._scaled(axes[1], self._pos_step)
          self._delta[:, 2] = self._scaled(axes[2], self._pos_step)
          self._delta[:, 3:] = 0.0
        else:
          self._delta[:, :3] = 0.0
          self._delta[:, 3] = self._scaled(axes[0], self._rot_step)
          self._delta[:, 4] = self._scaled(axes[1], self._rot_step)
          self._delta[:, 5] = self._scaled(axes[2], self._rot_step)
      elif packet_type == 2:
        self._delta.zero_()
      elif packet_type == 3 and len(data) > 1:
        buttons = int(data[1])
        pressed = buttons & ~self._button_state
        self._button_state = buttons
        if pressed & SPACEMOUSE_LEFT_BUTTON:
          self._reset_orientation()
        if pressed & SPACEMOUSE_RIGHT_BUTTON:
          self._toggle_mode()

  def consume_command(self) -> CartesianCommand:
    with self._lock:
      delta = self._delta.clone()
      reset = self._target_reset_requested
      reset_orientation = self._orientation_reset_requested
      self._target_reset_requested = False
      self._orientation_reset_requested = False
    return CartesianCommand(delta=delta, reset_target=reset, reset_orientation=reset_orientation)

  def request_target_reset(self) -> None:
    with self._lock:
      self._delta.zero_()
      self._target_reset_requested = True

  def close(self) -> None:
    self._stop.set()
    try:
      self._device.close()
    except OSError:
      pass
    self._thread.join(timeout=0.2)


class SpaceMouseBridgeInput(CartesianDeltaInput):
  """Receive SpaceMouse commands from a local bridge over SSH port forwarding."""

  def __init__(
    self,
    action_shape: tuple[int, ...],
    device: torch.device | str,
    *,
    pos_step: float,
    rot_step: float,
    host: str,
    port: int,
  ):
    self._delta = torch.zeros((*action_shape[:-1], 6), device=device)
    self._target_reset_requested = False
    self._orientation_reset_requested = False
    self._pos_step = pos_step
    self._rot_step = rot_step
    self._lock = threading.Lock()
    self._stop = threading.Event()
    self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self._server_socket.bind((host, port))
    self._server_socket.listen(1)
    self._server_socket.settimeout(0.2)
    self._bound_host, self._bound_port = self._server_socket.getsockname()
    self._thread = threading.Thread(
      target=self._server_loop,
      name="rizon4-spacemouse-bridge",
      daemon=True,
    )
    self._thread.start()

  @property
  def bound_address(self) -> tuple[str, int]:
    return self._bound_host, self._bound_port

  def _server_loop(self) -> None:
    while not self._stop.is_set():
      try:
        conn, addr = self._server_socket.accept()
      except socket.timeout:
        continue
      except OSError:
        break

      print(f"SpaceMouse bridge connected from {addr[0]}:{addr[1]}")
      try:
        self._client_loop(conn)
      finally:
        with self._lock:
          self._delta.zero_()
        print("SpaceMouse bridge disconnected")

  def _client_loop(self, conn: socket.socket) -> None:
    buffer = ""
    with conn:
      conn.settimeout(0.2)
      while not self._stop.is_set():
        try:
          chunk = conn.recv(4096)
        except socket.timeout:
          continue
        except OSError:
          break
        if not chunk:
          break

        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
          line, buffer = buffer.split("\n", 1)
          line = line.strip()
          if not line:
            continue
          try:
            self._update_from_bridge_message(json.loads(line))
          except (TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"Ignoring invalid SpaceMouse bridge message: {exc}")

  def _update_from_bridge_message(self, message: dict) -> None:
    reset = bool(message.get("reset", False))
    reset_orientation = bool(message.get("reset_orientation", False))
    mode = message.get("mode", "position")
    axes = message.get("axes")
    delta = message.get("delta")
    with self._lock:
      if axes is not None:
        if len(axes) != 6:
          raise ValueError("axes must contain 6 values")
        axes_tensor = torch.tensor(axes[:3], device=self._delta.device, dtype=self._delta.dtype)
        if mode == "position":
          self._delta[:, :3] = axes_tensor * self._pos_step
          self._delta[:, 3:] = 0.0
        elif mode == "orientation":
          self._delta[:, :3] = 0.0
          self._delta[:, 3:] = axes_tensor * self._rot_step
        else:
          raise ValueError("mode must be 'position' or 'orientation'")
      elif delta is not None:
        if len(delta) != 6:
          raise ValueError("delta must contain 6 values")
        self._delta[:] = torch.tensor(delta, device=self._delta.device, dtype=self._delta.dtype)
      if reset:
        self._delta.zero_()
        self._target_reset_requested = True
      if reset_orientation:
        self._delta.zero_()
        self._orientation_reset_requested = True

  def consume_command(self) -> CartesianCommand:
    with self._lock:
      delta = self._delta.clone()
      reset = self._target_reset_requested
      reset_orientation = self._orientation_reset_requested
      self._target_reset_requested = False
      self._orientation_reset_requested = False
    return CartesianCommand(delta=delta, reset_target=reset, reset_orientation=reset_orientation)

  def request_target_reset(self) -> None:
    with self._lock:
      self._delta.zero_()
      self._target_reset_requested = True

  def close(self) -> None:
    self._stop.set()
    try:
      self._server_socket.close()
    except OSError:
      pass
    self._thread.join(timeout=0.5)


class ViserCartesianInput(CartesianDeltaInput):
  """Browser GUI, hold buttons, and transform gizmo for SSH-friendly control."""

  def __init__(
    self,
    action_shape: tuple[int, ...],
    device: torch.device | str,
    *,
    pos_step: float,
    rot_step: float,
  ):
    self._action_shape = action_shape
    self._device = device
    self._slider_raw = torch.zeros((*action_shape[:-1], 6), device=device)
    self._pulse_delta = torch.zeros((*action_shape[:-1], 6), device=device)
    self._scales = torch.tensor(
      (pos_step, pos_step, pos_step, rot_step, rot_step, rot_step),
      device=device,
    )
    self._target_reset_requested = False
    self._target_override_pos: torch.Tensor | None = None
    self._target_override_quat: torch.Tensor | None = None
    self._sliders_enabled = True
    self._lock = threading.Lock()
    self._sliders = []
    self._transform = None
    self._syncing_transform = False
    self._dragging_transform = False

  def consume_command(self) -> CartesianCommand:
    with self._lock:
      delta = self._pulse_delta.clone()
      self._pulse_delta.zero_()
      if self._sliders_enabled:
        delta += self._slider_raw * self._scales
      reset = self._target_reset_requested
      self._target_reset_requested = False
      target_pos = self._target_override_pos.clone() if self._target_override_pos is not None else None
      target_quat = self._target_override_quat.clone() if self._target_override_quat is not None else None
      self._target_override_pos = None
      self._target_override_quat = None
    return CartesianCommand(
      delta=delta,
      reset_target=reset,
      target_pos=target_pos,
      target_quat=target_quat,
    )

  def request_target_reset(self) -> None:
    with self._lock:
      self._pulse_delta.zero_()
      self._target_override_pos = None
      self._target_override_quat = None
      self._target_reset_requested = True

  def _set_slider_raw_dim(self, dim: int, value: float) -> None:
    with self._lock:
      self._slider_raw[..., dim] = value

  def _add_pulse_dim(self, dim: int, sign: float) -> None:
    with self._lock:
      self._pulse_delta[..., dim] += sign * self._scales[dim]

  def _batched_pose_tensor(self, values, width: int) -> torch.Tensor:
    tensor = torch.tensor(values, device=self._device, dtype=torch.float32)
    batch_shape = self._action_shape[:-1]
    if not batch_shape:
      return tensor
    return tensor.reshape(*((1,) * len(batch_shape)), width).expand(*batch_shape, width).clone()

  def _set_target_override(self, position, wxyz) -> None:
    with self._lock:
      self._target_override_pos = self._batched_pose_tensor(position, 3)
      self._target_override_quat = self._batched_pose_tensor(wxyz, 4)

  def _zero(self) -> None:
    with self._lock:
      self._slider_raw.zero_()
      self._pulse_delta.zero_()
    for slider in self._sliders:
      slider.value = 0.0

  def sync_target(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> None:
    if self._transform is None or self._dragging_transform:
      return

    position = target_pos.reshape(-1, 3)[0].detach().cpu().numpy()
    wxyz = target_quat.reshape(-1, 4)[0].detach().cpu().numpy()
    self._syncing_transform = True
    try:
      self._transform.position = position
      self._transform.wxyz = wxyz
    finally:
      self._syncing_transform = False

  def create_gui(self, server) -> None:
    import viser

    self._transform = server.scene.add_transform_controls(
      "/rizon4_tcp_target",
      scale=0.16,
      depth_test=False,
      opacity=0.85,
    )

    @self._transform.on_drag_start
    def _(_ev) -> None:
      self._dragging_transform = True

    @self._transform.on_update
    def _(_ev) -> None:
      if self._syncing_transform:
        return
      self._set_target_override(self._transform.position, self._transform.wxyz)

    @self._transform.on_drag_end
    def _(_ev) -> None:
      self._dragging_transform = False
      self._set_target_override(self._transform.position, self._transform.wxyz)

    axis_specs = (
      ("Wall axis (+toward wall)", 0),
      ("Lateral", 1),
      ("Vertical", 2),
      ("Roll", 3),
      ("Pitch", 4),
      ("Yaw", 5),
    )
    with server.gui.add_folder("Rizon4 Cartesian Teleop"):
      enabled = server.gui.add_checkbox("Enable sliders", initial_value=True)
      show_transform = server.gui.add_checkbox("Show TCP target gizmo", initial_value=True)

      @enabled.on_update
      def _(_ev) -> None:
        with self._lock:
          self._sliders_enabled = bool(enabled.value)

      @show_transform.on_update
      def _(_ev) -> None:
        if self._transform is not None:
          self._transform.visible = bool(show_transform.value)

      for label, dim in axis_specs:
        slider = server.gui.add_slider(
          label,
          min=-1.0,
          max=1.0,
          step=0.01,
          initial_value=0.0,
        )

        @slider.on_update
        def _(_ev, _slider=slider, _dim=dim) -> None:
          self._set_slider_raw_dim(_dim, float(_slider.value))

        self._sliders.append(slider)

      with server.gui.add_folder("Jog translate"):
        for label, dim, sign in (
          ("Wall +", 0, 1.0),
          ("Wall -", 0, -1.0),
          ("Left", 1, 1.0),
          ("Right", 1, -1.0),
          ("Up", 2, 1.0),
          ("Down", 2, -1.0),
        ):
          button = server.gui.add_button(label)

          @button.on_hold(callback_hz=30.0)
          def _(_ev, _dim=dim, _sign=sign) -> None:
            self._add_pulse_dim(_dim, _sign)

      with server.gui.add_folder("Jog rotate"):
        for label, dim, sign in (
          ("Roll +", 3, 1.0),
          ("Roll -", 3, -1.0),
          ("Pitch +", 4, 1.0),
          ("Pitch -", 4, -1.0),
          ("Yaw +", 5, 1.0),
          ("Yaw -", 5, -1.0),
        ):
          button = server.gui.add_button(label)

          @button.on_hold(callback_hz=30.0)
          def _(_ev, _dim=dim, _sign=sign) -> None:
            self._add_pulse_dim(_dim, _sign)

      zero_button = server.gui.add_button("Zero command", icon=viser.Icon.SQUARE_X)
      reset_target_button = server.gui.add_button(
        "Reset TCP target to current pose",
        icon=viser.Icon.TARGET,
      )

      @zero_button.on_click
      def _(_ev) -> None:
        self._zero()

      @reset_target_button.on_click
      def _(_ev) -> None:
        self.request_target_reset()


class CompositeCartesianInput(CartesianDeltaInput):
  """Combine multiple input sources into one Cartesian command stream."""

  def __init__(self, inputs: list[CartesianDeltaInput], action_shape: tuple[int, ...], device: torch.device | str):
    self._inputs = inputs
    self._action_shape = action_shape
    self._device = device

  def consume_command(self) -> CartesianCommand:
    delta = torch.zeros((*self._action_shape[:-1], 6), device=self._device)
    reset_target = False
    reset_orientation = False
    target_pos: torch.Tensor | None = None
    target_quat: torch.Tensor | None = None

    for input_device in self._inputs:
      command = input_device.consume_command()
      delta += command.delta
      reset_target = reset_target or command.reset_target
      reset_orientation = reset_orientation or command.reset_orientation
      if command.target_pos is not None and command.target_quat is not None:
        target_pos = command.target_pos
        target_quat = command.target_quat

    return CartesianCommand(
      delta=delta,
      reset_target=reset_target,
      reset_orientation=reset_orientation,
      target_pos=target_pos,
      target_quat=target_quat,
    )

  def handle_key(self, key: int, *, pos_step: float, rot_step: float) -> bool:
    return any(
      input_device.handle_key(key, pos_step=pos_step, rot_step=rot_step)
      for input_device in self._inputs
    )

  def request_target_reset(self) -> None:
    for input_device in self._inputs:
      input_device.request_target_reset()

  def create_gui(self, server) -> None:
    for input_device in self._inputs:
      input_device.create_gui(server)

  def sync_target(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> None:
    for input_device in self._inputs:
      input_device.sync_target(target_pos, target_quat)

  def close(self) -> None:
    for input_device in self._inputs:
      input_device.close()


def _convert_buffer(byte_low: int, byte_high: int) -> float:
  value = byte_low | (byte_high << 8)
  if value >= 32768:
    value = -(65536 - value)
  return max(-1.0, min(1.0, value / 350.0))


_HUD_RELEASE_REASONS = {
  0: "disengage-gate",
  1: "reverse-shear",
  2: "tension-cone",
  3: "lateral",
  4: "ultimate-shear",
}
_GREEN, _RED, _RESET = "\033[32m", "\033[31m", "\033[0m"


class MicrospineHud:
  """Throttled terminal readout of the microspine bench state (env 0).

  Event lines print on every engage/release transition — releases name the
  envelope face that fired and the violating load. A compact status line
  prints every ``period`` seconds. Reads only the affordance/executor HUD
  buffers; silently a no-op in legacy adhesion mode.
  """

  def __init__(self, env, period: float = 0.5):
    self._period = period
    self._last_status = 0.0
    self._aff = None
    self._executor = None
    try:
      aff = env.command_manager.get_term("microspine_affordance")
      executor = env.event_manager.get_term_cfg("microspine_adhesion").func
    except (KeyError, ValueError):
      return
    if not hasattr(aff, "release_reason") or not hasattr(
      executor, "anchor_pos_w"
    ):
      return
    self._aff = aff
    self._executor = executor
    self._spheres = (
      aff._mcfg.spheres_per_ee_rows * aff._mcfg.spheres_per_ee_cols
    )
    self._prev_engaged = aff.engaged.clone()

  def reset(self) -> None:
    if self._aff is not None:
      self._prev_engaged = self._aff.engaged.clone()

  def _pull_push_caps(self, h: int, f_t: float) -> tuple[float, float]:
    aff = self._aff
    ft_pos = max(f_t, 0.0)
    pull = (
      float(aff.limit_mu_adhesion[0, h]) * ft_pos + float(aff.limit_f_n0[0, h])
    )
    push = -float(aff.limit_f_rev_max[0, h])
    return pull, push

  def _print_transitions(self, engaged: torch.Tensor) -> None:
    changed = engaged != self._prev_engaged
    if not bool(changed.any().item()):
      return
    aff = self._aff
    for h in changed[0].nonzero(as_tuple=True)[0].tolist():
      if bool(engaged[0, h].item()):
        x, y, z = self._executor.anchor_pos_w[0, h].tolist()
        print(
          f"{_GREEN}[microspine] >>> ENGAGED  "
          f"anchor=({x:.3f}, {y:.3f}, {z:.3f}){_RESET}",
          flush=True,
        )
      else:
        reason = int(aff.release_reason[0, h].item())
        f_t, f_l, f_n = aff.anchor_load_tnl[0, h].tolist()
        pull_cap, push_cap = self._pull_push_caps(h, f_t)
        name = _HUD_RELEASE_REASONS.get(reason, f"reason-{reason}")
        if reason == 1:
          detail = f"f_t {f_t:.1f} < cap {push_cap:.1f} (pushed up the wall)"
        elif reason == 2:
          detail = f"f_n {f_n:.1f} > cap {pull_cap:.1f} (pulled off the wall)"
        elif reason == 3:
          cap_l = float(aff.limit_mu_lateral[0, h]) * max(f_t, 0.0) + float(
            aff.limit_f_l0[0, h]
          )
          detail = f"|f_l| {abs(f_l):.1f} > cap {cap_l:.1f}"
        elif reason == 4:
          detail = f"f_t {f_t:.1f} > cap {float(aff.limit_f_t_max[0, h]):.1f}"
        else:
          detail = "force-z disengage gate"
        print(
          f"{_RED}[microspine] >>> RELEASED ({name}): {detail}  "
          f"[f_t={f_t:.1f} f_l={f_l:.1f} f_n={f_n:.1f}]{_RESET}",
          flush=True,
        )
    self._prev_engaged = engaged.clone()

  def _print_status(self) -> None:
    aff = self._aff
    h = 0
    contacts = int(aff.contact_count[0, h].item())
    cos = float(aff.face_cos[0, h].item())
    if bool(aff.engaged[0, h].item()):
      f_t, f_l, f_n = aff.anchor_load_tnl[0, h].tolist()
      steps = int(aff._steps_since_engage[0, h].item())
      grace = aff._ls_cfg.hold_grace_steps
      state = f"GRACE {steps}/{grace}" if steps < grace else "ENGAGED"
      margin = float(aff.hold_margin[0, h].item())
      pull_cap, push_cap = self._pull_push_caps(h, f_t)
      line = (
        f"[microspine] {state:<10} f_t={f_t:+6.1f}N f_l={f_l:+6.1f}N "
        f"f_n={f_n:+6.1f}N  margin={margin:+.2f}  "
        f"pull-cap≈{pull_cap:.1f}N push-cap≈{push_cap:.1f}N  "
        f"contacts {contacts}/{self._spheres} cos={cos:.2f}"
      )
    else:
      cooldown = int(aff._cooldown_steps[0, h].item())
      state = f"OFF (cooldown {cooldown})" if cooldown > 0 else "OFF"
      line = (
        f"[microspine] {state:<18} contacts {contacts}/{self._spheres} "
        f"cos={cos:.2f}"
      )
    print(line, flush=True)

  def update(self) -> None:
    if self._aff is None:
      return
    self._print_transitions(self._aff.engaged)
    now = time.monotonic()
    if now - self._last_status >= self._period:
      self._last_status = now
      self._print_status()


class CartesianPosePolicy:
  def __init__(
    self,
    env,
    input_device: CartesianDeltaInput,
    hud_period: float = 0.5,
    target_leash: float = 0.002,
  ):
    self._target_leash = target_leash
    self._env = env.unwrapped
    self._input = input_device
    self._robot = self._env.scene["robot"]
    site_ids, _ = self._robot.find_sites(RIZON4_MICROSPINE_TCP_SITE)
    self._site_id = site_ids[0]
    self._target_pos: torch.Tensor | None = None
    self._target_quat: torch.Tensor | None = None
    self._initial_quat: torch.Tensor | None = None
    self._microspine_physics = self._get_microspine_physics()
    self._prev_engaged: torch.Tensor | None = None
    self._hud = MicrospineHud(self._env, period=hud_period)

  def _get_microspine_physics(self):
    try:
      return self._env.event_manager.get_term_cfg("microspine_adhesion").func
    except ValueError:
      return None

  def _microspine_disengaged_this_step(self) -> bool:
    if self._microspine_physics is None:
      return False
    disengaged = getattr(self._microspine_physics, "disengaged_this_step", None)
    if disengaged is None:
      return False
    return bool(disengaged.any().item())

  def _microspine_engaged_this_step(self) -> bool:
    """Rising-edge detector on the latch state.

    On engage the accumulated command offset (press-through depth plus
    whatever the hand was doing at that instant) must be cancelled: the IK is
    effectively a stiff position servo (~10^2 N/mm through the anchor), so
    even sub-mm residual motion blows through the unloaded envelope caps
    (~20 N) right after grace and the pad releases the moment it latched.
    Snapping the target to the measured TCP starts the lock load-free; any
    later load is deliberate stick deflection.
    """
    if self._microspine_physics is None:
      return False
    engaged = getattr(self._microspine_physics, "engaged", None)
    if engaged is None:
      return False
    prev = self._prev_engaged
    self._prev_engaged = engaged.clone()
    if prev is None:
      return False
    return bool((engaged & ~prev).any().item())

  def reset_target(self) -> None:
    self._target_pos = self._robot.data.site_pos_w[:, self._site_id, :].clone()
    self._target_quat = self._robot.data.site_quat_w[:, self._site_id, :].clone()
    if self._initial_quat is None:
      self._initial_quat = self._target_quat.clone()

  def reset_orientation(self) -> None:
    if self._target_quat is None:
      self.reset_target()
    assert self._target_quat is not None
    if self._initial_quat is None:
      self._initial_quat = self._target_quat.clone()
    self._target_quat = self._initial_quat.clone()

  def reset(self) -> None:
    self._target_pos = None
    self._target_quat = None
    self._initial_quat = None
    self._prev_engaged = None
    self._hud.reset()

  def __call__(self, obs) -> torch.Tensor:
    del obs
    if self._target_pos is None or self._target_quat is None:
      self.reset_target()

    command = self._input.consume_command()
    if command.reset_target:
      self.reset_target()

    assert self._target_pos is not None
    assert self._target_quat is not None
    if command.target_pos is not None and command.target_quat is not None:
      self._target_pos = command.target_pos.to(
        device=self._target_pos.device,
        dtype=self._target_pos.dtype,
      )
      self._target_quat = command.target_quat.to(
        device=self._target_quat.device,
        dtype=self._target_quat.dtype,
      )
    if command.reset_orientation:
      self.reset_orientation()
    if self._microspine_disengaged_this_step() or self._microspine_engaged_this_step():
      self.reset_target()

    delta = command.delta
    if delta.abs().max().item() > 0.0:
      self._target_pos, self._target_quat = apply_delta_pose(
        self._target_pos,
        self._target_quat,
        delta,
      )
    # Leash only while latched: free-space tracking error must not eat the
    # leash budget (it stalls the approach), but once the pad is anchored any
    # target-vs-TCP offset IS injected force, so cap it.
    latched = self._prev_engaged is not None and bool(
      self._prev_engaged.any().item()
    )
    if self._target_leash > 0.0 and latched:
      tcp = self._robot.data.site_pos_w[:, self._site_id, :]
      offset = self._target_pos - tcp
      norm = offset.norm(dim=-1, keepdim=True)
      scale = (self._target_leash / norm.clamp_min(self._target_leash)).clamp(
        max=1.0
      )
      self._target_pos = tcp + offset * scale
    self._input.sync_target(self._target_pos, self._target_quat)
    self._hud.update()
    return torch.cat((self._target_pos, self._target_quat), dim=-1)


class Rizon4TeleopViewer(NativeMujocoViewer):
  """Native viewer with task-specific keyboard bindings."""

  def __init__(
    self,
    env,
    policy: CartesianPosePolicy,
    input_device: CartesianDeltaInput,
    cfg: TeleopConfig,
  ):
    super().__init__(
      env,
      policy,
      frame_rate=cfg.frame_rate,
      enable_perturbations=True,
    )
    self._policy = policy
    self._input_device = input_device
    self._teleop_cfg = cfg

  def _safe_key_callback(self, key: int) -> None:
    if self._input_device.handle_key(
      key,
      pos_step=self._teleop_cfg.pos_step,
      rot_step=self._teleop_cfg.rot_step,
    ):
      return

    if key == keys.KEY_R:
      self._input_device.request_target_reset()
      self.request_reset()
      return
    if key == keys.KEY_ESCAPE:
      if self.viewer is not None:
        self.viewer.close()
      return

    super()._safe_key_callback(key)


class Rizon4TeleopViserViewer(ViserPlayViewer):
  """Viser viewer with browser-side Cartesian teleop controls."""

  def __init__(
    self,
    env,
    policy: CartesianPosePolicy,
    input_device: CartesianDeltaInput,
    cfg: TeleopConfig,
    viser_server,
  ):
    super().__init__(
      env,
      policy,
      frame_rate=cfg.frame_rate,
      viser_server=viser_server,
    )
    self._policy = policy
    self._input_device = input_device

  def setup(self) -> None:
    super().setup()
    self._input_device.create_gui(self._server)


def _display_available() -> bool:
  return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _resolve_viewer(cfg: TeleopConfig) -> Literal["native", "viser"]:
  if cfg.viewer != "auto":
    return cfg.viewer
  return "native" if _display_available() else "viser"


def _resolve_input(
  cfg: TeleopConfig,
  viewer: Literal["native", "viser"],
) -> Literal["spacemouse", "keyboard", "viser"]:
  if cfg.input != "auto":
    return cfg.input
  return "viser" if viewer == "viser" else "spacemouse"


def _make_spacemouse_input(
  cfg: TeleopConfig,
  action_shape: tuple[int, ...],
  device: torch.device | str,
) -> CartesianDeltaInput:
  if cfg.spacemouse_source in ("hid", "auto"):
    try:
      return SpaceMouseCartesianInput(
        action_shape,
        device,
        pos_step=cfg.pos_step,
        rot_step=cfg.rot_step,
        deadband=cfg.spacemouse_deadband,
      )
    except OSError as exc:
      if cfg.spacemouse_source == "hid":
        raise
      print(f"Local HID SpaceMouse unavailable, falling back to bridge:\n{exc}")

  bridge_input = SpaceMouseBridgeInput(
    action_shape,
    device,
    pos_step=cfg.pos_step,
    rot_step=cfg.rot_step,
    host=cfg.spacemouse_bridge_host,
    port=cfg.spacemouse_bridge_port,
  )
  host, port = bridge_input.bound_address
  print(f"SpaceMouse bridge listening on {host}:{port}")
  return bridge_input


def _make_input_device(
  cfg: TeleopConfig,
  input_kind: Literal["spacemouse", "keyboard", "viser"],
  viewer: Literal["native", "viser"],
  action_shape: tuple[int, ...],
  device: torch.device | str,
) -> CartesianDeltaInput:
  if action_shape[-1] != 7:
    raise ValueError(f"Expected 7D absolute pose action, got shape {action_shape}.")
  if viewer == "viser":
    viser_input = ViserCartesianInput(
      action_shape,
      device,
      pos_step=cfg.pos_step,
      rot_step=cfg.rot_step,
    )
    if input_kind == "viser":
      return viser_input
    if input_kind == "spacemouse":
      spacemouse_input = _make_spacemouse_input(cfg, action_shape, device)
      return CompositeCartesianInput([viser_input, spacemouse_input], action_shape, device)
    raise ValueError("Keyboard input is not supported in the Viser viewer.")

  if input_kind == "keyboard":
    return KeyboardCartesianInput(action_shape, device)
  if input_kind == "spacemouse":
    return _make_spacemouse_input(cfg, action_shape, device)
  raise ValueError("Viser input requires the Viser viewer.")


def run(cfg: TeleopConfig) -> None:
  viewer = _resolve_viewer(cfg)
  input_kind = _resolve_input(cfg, viewer)
  if viewer == "native" and not _display_available():
    raise RuntimeError("Native teleop requires DISPLAY/WAYLAND_DISPLAY. Use --viewer viser over SSH.")
  if input_kind == "viser" and viewer != "viser":
    raise RuntimeError("The Viser input backend requires --viewer viser.")
  if input_kind == "keyboard" and viewer != "native":
    raise RuntimeError("Keyboard input is only supported by the native MuJoCo viewer.")

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(cfg.task, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  if not cfg.debug:
    # The play cfg enables raw per-step microspine debug prints; the HUD
    # replaces them. The scalar `debug` field is NOT shared between the
    # command and event copies of MicrospineCfg (shallow dataclasses.replace),
    # so both must be cleared — and before env construction, because the
    # executor caches the flag in __init__.
    for owner in (
      env_cfg.commands["microspine_affordance"].microspine_cfg,
      env_cfg.events["microspine_adhesion"].params["microspine_cfg"],
    ):
      owner.debug = False
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)
  input_device: CartesianDeltaInput | None = None

  try:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    input_device = _make_input_device(cfg, input_kind, viewer, action_shape, env.unwrapped.device)
    policy = CartesianPosePolicy(
      env,
      input_device,
      hud_period=cfg.hud_period,
      target_leash=cfg.target_leash,
    )

    print(f"Viewer: {viewer}")
    input_label = "viser + spacemouse" if viewer == "viser" and input_kind == "spacemouse" else input_kind
    print(f"Input: {input_label}")
    if input_kind == "spacemouse":
      print("  3Dconnexion cap: position/orientation mode")
      print("  left button: reset orientation to initial TCP orientation")
      print("  right button: toggle position/orientation mode")
      print(f"  SpaceMouse source: {cfg.spacemouse_source}")
      if viewer == "viser":
        print("  Viser page controls and TCP target gizmo remain enabled")
    if input_kind == "keyboard":
      print("Keyboard control:")
      print("  translate: W/S wall axis, A/D lateral, Q/E vertical")
      print("  rotate:    I/K rx, J/L ry, U/O rz")
      print("  controls:  Space reset TCP target, R reset env, Esc close")
      print("  MJLab viewer controls still work for other keys.")
    if viewer == "viser":
      import viser

      server = viser.ViserServer(
        host=cfg.viser_host,
        port=cfg.viser_port,
        label="rizon4-microspine-teleop",
      )
      print("Viser remote access:")
      print(f"  ssh -L {cfg.viser_port}:localhost:{cfg.viser_port} <remote-host>")
      print(f"  then open http://localhost:{cfg.viser_port}")
      if input_kind == "spacemouse" and cfg.spacemouse_source in ("bridge", "auto"):
        print("SpaceMouse bridge over SSH:")
        print(
          f"  ssh -L {cfg.viser_port}:localhost:{cfg.viser_port} "
          f"-L {cfg.spacemouse_bridge_port}:localhost:{cfg.spacemouse_bridge_port} <remote-host>"
        )
        print(
          "  local: python scripts/spacemouse_bridge.py "
          f"--host 127.0.0.1 --port {cfg.spacemouse_bridge_port}"
        )
      Rizon4TeleopViserViewer(env, policy, input_device, cfg, server).run()
    else:
      Rizon4TeleopViewer(env, policy, input_device, cfg).run()
  finally:
    if input_device is not None:
      input_device.close()
    env.close()


def main() -> None:
  run(tyro.cli(TeleopConfig))


if __name__ == "__main__":
  main()
