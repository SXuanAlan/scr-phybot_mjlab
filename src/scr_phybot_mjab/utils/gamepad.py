"""Receive gamepad joystick input for play and map it to a velocity command.

The physical gamepad lives on a local machine and forwards normalized stick
deflections over a TCP socket (see ``scripts/gamepad_bridge.py``), mirroring the
SpaceMouse bridge. This module runs the receiving side inside ``scripts/play.py``:
a background thread reads newline-delimited JSON packets and the latest state is
mapped to a ``(lin_vel_x, lin_vel_y, ang_vel_z)`` command on demand.

Packet schema (all fields optional, last value wins)::

  {"fwd": <-1..1>, "left": <-1..1>, "turn": <-1..1>, "speed": <0|1|2>}

``fwd``/``left``/``turn`` are robot-convention stick fractions already sign- and
deadband-corrected by the bridge (fwd>0 forward, left>0 strafe left, turn>0 yaw
left/CCW). ``speed`` selects the speed preset (0 slow, 1 medium, 2 fast).
"""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeedPreset:
  """Maximum command magnitudes for one speed level (robot units)."""

  name: str
  lin_vel_x: float  # max forward speed [m/s]; full-stick deflection maps here
  lin_vel_y: float  # max strafe speed [m/s]
  ang_vel_z: float  # max yaw rate [rad/s]


# Slow / medium / fast. Medium is the default. Values stay within the walk play
# command ranges (lin_vel_x in [-1.5, 2.0], lin_vel_y in [-1.0, 1.0],
# ang_vel_z in [-0.7, 0.7]); the receiver also clamps to the live ranges.
SPEED_PRESETS: tuple[SpeedPreset, ...] = (
  SpeedPreset(name="slow", lin_vel_x=0.5, lin_vel_y=0.3, ang_vel_z=0.4),
  SpeedPreset(name="medium", lin_vel_x=1.0, lin_vel_y=0.5, ang_vel_z=0.6),
  SpeedPreset(name="fast", lin_vel_x=2.0, lin_vel_y=0.8, ang_vel_z=0.7),
)
DEFAULT_SPEED_INDEX = 1  # medium


class GamepadReceiver:
  """Background TCP server that tracks the latest gamepad command.

  Listens on ``host:port`` and accepts a single bridge connection at a time.
  Reconnections are handled transparently. Thread-safe; call :meth:`command`
  from the simulation/control thread to read the mapped velocity command.
  """

  def __init__(
    self,
    host: str = "127.0.0.1",
    port: int = 8766,
    *,
    lin_vel_x_range: tuple[float, float] = (-1.5, 2.0),
    lin_vel_y_range: tuple[float, float] = (-1.0, 1.0),
    ang_vel_z_range: tuple[float, float] = (-0.7, 0.7),
  ) -> None:
    self._host = host
    self._port = port
    self._lin_vel_x_range = lin_vel_x_range
    self._lin_vel_y_range = lin_vel_y_range
    self._ang_vel_z_range = ang_vel_z_range

    self._lock = threading.Lock()
    self._fwd = 0.0
    self._left = 0.0
    self._turn = 0.0
    self._speed_index = DEFAULT_SPEED_INDEX
    self._connected = False
    self._got_packet = False

    self._stop = threading.Event()
    self._server: socket.socket | None = None
    self._thread: threading.Thread | None = None
    self._status_thread: threading.Thread | None = None

  # Lifecycle.

  def start(self) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((self._host, self._port))
    server.listen(1)
    server.settimeout(0.5)
    self._server = server
    self._thread = threading.Thread(target=self._serve, name="gamepad-rx", daemon=True)
    self._thread.start()
    self._status_thread = threading.Thread(target=self._status_loop, name="gamepad-status", daemon=True)
    self._status_thread.start()
    print(
      f"[INFO]: Gamepad receiver listening on {self._host}:{self._port} "
      f"(default speed: {SPEED_PRESETS[DEFAULT_SPEED_INDEX].name}). "
      "Waiting for scripts/gamepad_bridge.py."
    )

  def _status_loop(self) -> None:
    """Print a once-per-second heartbeat so link/axis issues are visible."""
    while not self._stop.wait(1.0):
      with self._lock:
        connected = self._connected
        got = self._got_packet
      vx, vy, wz = self.command()
      print(
        f"[gamepad] connected={connected} data={got} cmd: vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}",
        flush=True,
      )

  def close(self) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=1.0)
    if self._status_thread is not None:
      self._status_thread.join(timeout=1.0)
    if self._server is not None:
      try:
        self._server.close()
      except OSError:
        pass

  def __enter__(self) -> "GamepadReceiver":
    self.start()
    return self

  def __exit__(self, *exc) -> None:
    del exc
    self.close()

  # Server thread.

  def _serve(self) -> None:
    assert self._server is not None
    while not self._stop.is_set():
      try:
        conn, _addr = self._server.accept()
      except socket.timeout:
        continue
      except OSError:
        break
      with self._lock:
        self._connected = True
      print("[INFO]: Gamepad bridge connected.")
      try:
        self._read_connection(conn)
      finally:
        conn.close()
        with self._lock:
          self._connected = False
          # Stop the robot when the bridge drops to avoid runaway commands.
          self._fwd = self._left = self._turn = 0.0
        if not self._stop.is_set():
          print("[INFO]: Gamepad bridge disconnected; waiting for reconnect.")

  def _read_connection(self, conn: socket.socket) -> None:
    conn.settimeout(0.5)
    buffer = b""
    while not self._stop.is_set():
      try:
        chunk = conn.recv(4096)
      except socket.timeout:
        continue
      except OSError:
        break
      if not chunk:
        break
      buffer += chunk
      while b"\n" in buffer:
        line, buffer = buffer.split(b"\n", 1)
        self._apply_packet(line)

  def _apply_packet(self, line: bytes) -> None:
    line = line.strip()
    if not line:
      return
    try:
      data = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
      return
    with self._lock:
      if "fwd" in data:
        self._fwd = _clamp(float(data["fwd"]), -1.0, 1.0)
      if "left" in data:
        self._left = _clamp(float(data["left"]), -1.0, 1.0)
      if "turn" in data:
        self._turn = _clamp(float(data["turn"]), -1.0, 1.0)
      if "speed" in data:
        self._speed_index = int(data["speed"]) % len(SPEED_PRESETS)
      first = not self._got_packet
      self._got_packet = True
    if first:
      print("[INFO]: Gamepad data flowing; sticks now drive the velocity command.")

  # Read API.

  @property
  def connected(self) -> bool:
    with self._lock:
      return self._connected

  def command(self) -> tuple[float, float, float]:
    """Map the latest stick state to ``(lin_vel_x, lin_vel_y, ang_vel_z)``."""
    with self._lock:
      fwd, left, turn = self._fwd, self._left, self._turn
      preset = SPEED_PRESETS[self._speed_index]
    vx = _clamp(fwd * preset.lin_vel_x, *self._lin_vel_x_range)
    vy = _clamp(left * preset.lin_vel_y, *self._lin_vel_y_range)
    wz = _clamp(turn * preset.ang_vel_z, *self._ang_vel_z_range)
    return vx, vy, wz


def _clamp(value: float, low: float, high: float) -> float:
  return max(low, min(high, value))
