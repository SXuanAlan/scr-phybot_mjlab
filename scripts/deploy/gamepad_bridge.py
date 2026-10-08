"""Forward a local gamepad's sticks to the remote walk-play velocity command.

Left stick drives linear x/y velocity, right stick drives yaw rate. Shoulder
buttons cycle the speed preset (slow / medium / fast; medium is the default).
It streams stick packets to the play process over TCP. ``scripts/play.py --gamepad`` starts it automatically for
same-machine play.

Typical usage:

  # Same-machine play starts this bridge automatically:
  python scripts/play.py Mjlab-Walk-Phybot-C2 --checkpoint-file logs/.../model_XXXX.pt --gamepad

  # Remote gamepad usage: disable auto bridge, forward the receiver port,
  # then start this bridge on the machine the gamepad is plugged into.
  python scripts/play.py Mjlab-Walk-Phybot-C2 --checkpoint-file logs/.../model_XXXX.pt --gamepad --no-gamepad-auto-bridge
  ssh -L 8766:localhost:8766 <remote-host>
  python scripts/deploy/gamepad_bridge.py --host 127.0.0.1 --port 8766

Use --list to print detected gamepads, or --calibrate to live-print axis/button
indices so you can match --lx-axis / --ly-axis / --rx-axis to your controller.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from dataclasses import dataclass

# Use SDL's dummy video/audio drivers so pygame's joystick subsystem works on a
# headless host (e.g. the training server) where no display is available. Set
# before importing pygame. Harmless when a display exists.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Stick fractions sign-corrected to robot convention before sending:
#   fwd > 0  -> forward, left > 0 -> strafe left, turn > 0 -> yaw left (CCW).
# Defaults assume the common SDL/Xbox layout (up and left read negative; right
# stick horizontal is axis 3). Flip with the --invert-* flags or remap axes if
# your pad differs (use --calibrate to discover indices).
SPEED_NAMES = ("slow", "medium", "fast")
DEFAULT_SPEED_INDEX = 1  # medium


@dataclass(frozen=True)
class BridgeConfig:
  host: str
  port: int
  deadband: float
  rate_hz: float
  reconnect_delay: float
  joystick_index: int
  lx_axis: int
  ly_axis: int
  rx_axis: int
  invert_fwd: bool
  invert_left: bool
  invert_turn: bool
  speed_up_button: int
  speed_down_button: int
  list_devices: bool
  calibrate: bool


def _parse_args(argv: list[str] | None = None) -> BridgeConfig:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--host", default="127.0.0.1", help="Forwarded TCP host.")
  parser.add_argument("--port", type=int, default=8766, help="Forwarded TCP port.")
  parser.add_argument("--deadband", type=float, default=0.08, help="Stick deadband.")
  parser.add_argument("--rate-hz", type=float, default=50.0, help="Send rate (Hz).")
  parser.add_argument("--reconnect-delay", type=float, default=1.0, help="Reconnect seconds.")
  parser.add_argument("--joystick-index", type=int, default=0, help="Which gamepad to open.")
  parser.add_argument("--lx-axis", type=int, default=0, help="Left stick X axis index.")
  parser.add_argument("--ly-axis", type=int, default=1, help="Left stick Y axis index.")
  parser.add_argument("--rx-axis", type=int, default=3, help="Right stick X axis index.")
  parser.add_argument("--invert-fwd", action="store_true", help="Flip forward direction.")
  parser.add_argument("--invert-left", action="store_true", help="Flip strafe direction.")
  parser.add_argument("--invert-turn", action="store_true", help="Flip yaw direction.")
  parser.add_argument("--speed-up-button", type=int, default=5, help="Button to speed up (default RB).")
  parser.add_argument("--speed-down-button", type=int, default=4, help="Button to slow down (default LB).")
  parser.add_argument("--list", dest="list_devices", action="store_true", help="List gamepads and exit.")
  parser.add_argument("--calibrate", action="store_true", help="Live-print axis/button values and exit.")
  a = parser.parse_args(argv)
  return BridgeConfig(
    host=a.host,
    port=a.port,
    deadband=a.deadband,
    rate_hz=a.rate_hz,
    reconnect_delay=a.reconnect_delay,
    joystick_index=a.joystick_index,
    lx_axis=a.lx_axis,
    ly_axis=a.ly_axis,
    rx_axis=a.rx_axis,
    invert_fwd=a.invert_fwd,
    invert_left=a.invert_left,
    invert_turn=a.invert_turn,
    speed_up_button=a.speed_up_button,
    speed_down_button=a.speed_down_button,
    list_devices=a.list_devices,
    calibrate=a.calibrate,
  )


def _deadband(value: float, band: float) -> float:
  """Apply a deadband and rescale the remainder back to full [-1, 1]."""
  if abs(value) < band:
    return 0.0
  sign = 1.0 if value > 0 else -1.0
  return sign * (abs(value) - band) / (1.0 - band)


def _axis(joystick, index: int) -> float:
  if index < 0 or index >= joystick.get_numaxes():
    return 0.0
  return float(joystick.get_axis(index))


def _open_joystick(pygame, config: BridgeConfig):
  count = pygame.joystick.get_count()
  if count == 0:
    raise SystemExit("No gamepad detected. Plug one in and re-run (try --list).")
  if config.joystick_index >= count:
    raise SystemExit(f"--joystick-index {config.joystick_index} >= {count} detected.")
  joystick = pygame.joystick.Joystick(config.joystick_index)
  joystick.init()
  print(f"Opened gamepad '{joystick.get_name()}' ({joystick.get_numaxes()} axes, {joystick.get_numbuttons()} buttons)")
  return joystick


def _list_devices(pygame) -> None:
  count = pygame.joystick.get_count()
  if count == 0:
    print("No gamepads detected.")
    return
  for i in range(count):
    js = pygame.joystick.Joystick(i)
    js.init()
    print(f"[{i}] {js.get_name()} axes={js.get_numaxes()} buttons={js.get_numbuttons()}")


def _calibrate(pygame, joystick) -> None:
  print("Move sticks / press buttons. Ctrl-C to exit.")
  try:
    while True:
      pygame.event.pump()
      axes = [round(joystick.get_axis(i), 2) for i in range(joystick.get_numaxes())]
      pressed = [i for i in range(joystick.get_numbuttons()) if joystick.get_button(i)]
      print(f"axes={axes} buttons_down={pressed}    ", end="\r", flush=True)
      time.sleep(0.05)
  except KeyboardInterrupt:
    print()


def _connect(config: BridgeConfig) -> socket.socket:
  while True:
    try:
      sock = socket.create_connection((config.host, config.port), timeout=2.0)
      sock.settimeout(None)
      print(f"Connected to gamepad receiver at {config.host}:{config.port}")
      return sock
    except OSError as exc:
      print(f"Waiting for receiver at {config.host}:{config.port}: {exc}")
      time.sleep(config.reconnect_delay)


def _send_loop(pygame, config: BridgeConfig, joystick) -> None:
  speed_index = DEFAULT_SPEED_INDEX
  prev_up = prev_down = False
  period = 1.0 / config.rate_hz
  sock: socket.socket | None = None
  print(f"Speed preset: {SPEED_NAMES[speed_index]}")

  while True:
    if sock is None:
      sock = _connect(config)

    pygame.event.pump()

    # Edge-triggered speed preset cycling.
    up = _button(joystick, config.speed_up_button)
    down = _button(joystick, config.speed_down_button)
    if up and not prev_up and speed_index < len(SPEED_NAMES) - 1:
      speed_index += 1
      print(f"Speed preset: {SPEED_NAMES[speed_index]}")
    if down and not prev_down and speed_index > 0:
      speed_index -= 1
      print(f"Speed preset: {SPEED_NAMES[speed_index]}")
    prev_up, prev_down = up, down

    fwd = _deadband(-_axis(joystick, config.ly_axis), config.deadband)
    left = _deadband(-_axis(joystick, config.lx_axis), config.deadband)
    turn = _deadband(-_axis(joystick, config.rx_axis), config.deadband)
    if config.invert_fwd:
      fwd = -fwd
    if config.invert_left:
      left = -left
    if config.invert_turn:
      turn = -turn

    payload = (
      json.dumps(
        {"fwd": fwd, "left": left, "turn": turn, "speed": speed_index},
        separators=(",", ":"),
      )
      + "\n"
    )
    try:
      sock.sendall(payload.encode("utf-8"))
    except OSError as exc:
      print(f"Send failed, reconnecting: {exc}")
      try:
        sock.close()
      except OSError:
        pass
      sock = None
    time.sleep(period)


def _button(joystick, index: int) -> bool:
  if index < 0 or index >= joystick.get_numbuttons():
    return False
  return bool(joystick.get_button(index))


def main(argv: list[str] | None = None) -> None:
  config = _parse_args(argv)
  try:
    import pygame
  except ImportError as exc:
    raise SystemExit("Install pygame first: python -m pip install pygame") from exc

  pygame.init()
  pygame.joystick.init()

  if config.list_devices:
    _list_devices(pygame)
    return

  joystick = _open_joystick(pygame, config)

  if config.calibrate:
    _calibrate(pygame, joystick)
    return

  try:
    _send_loop(pygame, config, joystick)
  except KeyboardInterrupt:
    print("\nBridge stopped.")


if __name__ == "__main__":
  main()
