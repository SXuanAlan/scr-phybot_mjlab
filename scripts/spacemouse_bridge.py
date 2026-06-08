"""Forward local 3Dconnexion SpaceMouse HID input to remote teleop.

Typical SSH usage:

  ssh -L 8765:localhost:8765 <remote-host>
  python scripts/spacemouse_bridge.py --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass


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


@dataclass(frozen=True)
class BridgeConfig:
  host: str
  port: int
  deadband: float
  reconnect_delay: float
  list_devices: bool


def _parse_args() -> BridgeConfig:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default="127.0.0.1", help="Forwarded TCP host.")
  parser.add_argument("--port", type=int, default=8765, help="Forwarded TCP port.")
  parser.add_argument("--deadband", type=float, default=0.03, help="Normalized HID axis deadband.")
  parser.add_argument("--reconnect-delay", type=float, default=1.0, help="Seconds between TCP reconnect attempts.")
  parser.add_argument("--list-devices", action="store_true", help="Print HID devices and exit.")
  args = parser.parse_args()
  return BridgeConfig(
    host=args.host,
    port=args.port,
    deadband=args.deadband,
    reconnect_delay=args.reconnect_delay,
    list_devices=args.list_devices,
  )


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


def _print_devices(hid_module) -> None:
  for device_info in hid_module.enumerate():
    vid = int(device_info.get("vendor_id") or 0)
    pid = int(device_info.get("product_id") or 0)
    manufacturer = device_info.get("manufacturer_string") or ""
    product = device_info.get("product_string") or ""
    path = device_info.get("path")
    print(f"{vid:04x}:{pid:04x} {manufacturer!r} {product!r} path={path!r}")


def _open_spacemouse(hid_module):
  last_open_error: OSError | None = None
  for device_info in hid_module.enumerate():
    if not _is_spacemouse_device(device_info):
      continue
    device = hid_module.device()
    product = _spacemouse_product_name(device_info)
    vid = int(device_info.get("vendor_id") or 0)
    pid = int(device_info.get("product_id") or 0)
    path = device_info.get("path")
    try:
      if path is not None:
        device.open_path(path)
      else:
        device.open(device_info["vendor_id"], device_info["product_id"])
      print(f"Opened {product} ({vid:04x}:{pid:04x})")
      return product, device
    except OSError as exc:
      last_open_error = OSError(
        f"Found {product} ({vid:04x}:{pid:04x}) but hidapi could not open it.\n"
        f"{SPACEMOUSE_UDEV_HINT}"
      )

  if last_open_error is not None:
    raise last_open_error
  raise OSError("No 3Dconnexion SpaceMouse HID device found.")


def _convert_buffer(byte_low: int, byte_high: int) -> float:
  value = byte_low | (byte_high << 8)
  if value >= 32768:
    value = -(65536 - value)
  return max(-1.0, min(1.0, value / 350.0))


def _scaled(value: float, deadband: float) -> float:
  return 0.0 if abs(value) < deadband else value


def _next_mode(mode: str) -> str:
  return "orientation" if mode == "position" else "position"


def _update_axes_from_packet(
  data: list[int],
  axes: list[float],
  mode: str,
  button_state: int,
  deadband: float,
) -> tuple[str, int, bool]:
  packet_type = data[0]
  reset_orientation = False
  if packet_type == 1:
    axes[1] = _scaled(-_convert_buffer(data[1], data[2]), deadband)
    axes[0] = _scaled(-_convert_buffer(data[3], data[4]), deadband)
    axes[2] = _scaled(-_convert_buffer(data[5], data[6]), deadband)
    axes[3:] = [0.0] * 3
  elif packet_type == 2:
    axes[:] = [0.0] * 6
  elif packet_type == 3 and len(data) > 1:
    buttons = int(data[1])
    pressed = buttons & ~button_state
    button_state = buttons
    axes[:] = [0.0] * 6
    if pressed & SPACEMOUSE_LEFT_BUTTON:
      reset_orientation = True
    if pressed & SPACEMOUSE_RIGHT_BUTTON:
      mode = _next_mode(mode)
      print(f"SpaceMouse mode: {mode}")
  return mode, button_state, reset_orientation


def _connect(config: BridgeConfig) -> socket.socket:
  while True:
    try:
      sock = socket.create_connection((config.host, config.port), timeout=2.0)
      sock.settimeout(None)
      print(f"Connected to SpaceMouse bridge receiver at {config.host}:{config.port}")
      return sock
    except OSError as exc:
      print(f"Waiting for bridge receiver at {config.host}:{config.port}: {exc}")
      time.sleep(config.reconnect_delay)


def _send_loop(config: BridgeConfig, device_name: str, hid_device) -> None:
  packet_size = _spacemouse_packet_size(device_name)
  axes = [0.0] * 6
  mode = "position"
  button_state = 0
  sock: socket.socket | None = None
  while True:
    if sock is None:
      sock = _connect(config)

    data = hid_device.read(packet_size, 50)
    if not data:
      continue

    mode, button_state, reset_orientation = _update_axes_from_packet(
      data,
      axes,
      mode,
      button_state,
      config.deadband,
    )
    payload = json.dumps(
      {"axes": axes, "mode": mode, "reset_orientation": reset_orientation},
      separators=(",", ":"),
    ) + "\n"
    try:
      sock.sendall(payload.encode("utf-8"))
    except OSError as exc:
      print(f"Bridge send failed, reconnecting: {exc}")
      try:
        sock.close()
      except OSError:
        pass
      sock = None


def main() -> None:
  config = _parse_args()
  try:
    import hid
  except ImportError as exc:
    raise SystemExit("Install hidapi first: python -m pip install hidapi") from exc

  if config.list_devices:
    _print_devices(hid)
    return

  try:
    device_name, hid_device = _open_spacemouse(hid)
  except OSError as exc:
    raise SystemExit(str(exc)) from exc

  try:
    _send_loop(config, device_name, hid_device)
  finally:
    hid_device.close()


if __name__ == "__main__":
  main()
