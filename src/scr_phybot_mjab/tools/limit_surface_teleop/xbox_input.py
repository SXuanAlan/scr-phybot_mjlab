"""Xbox gamepad -> engage-frame Cartesian delta + edge-detected buttons."""
from __future__ import annotations
from dataclasses import dataclass

_BTN = {
    "A": "home",
    "B": "release",
    "X": "clear",
    "Y": "save",
    "START": "engage",
    "DPAD_LEFT": "normal_gain_down",
    "DPAD_RIGHT": "normal_gain_up",
    "DPAD_DOWN": "shear_gain_down",
    "DPAD_UP": "shear_gain_up",
}


@dataclass
class InputState:
    delta_tln: tuple[float, float, float]
    buttons: frozenset


def _dz(v, dz):
    return 0.0 if abs(v) < dz else v


class XboxInput:
    def __init__(
        self,
        source,
        lin_gain: float = 0.02,
        deadzone: float = 0.15,
        shear_gain: float | None = None,
        normal_gain: float | None = None,
    ):
        self._src = source
        self._g = float(lin_gain)
        self._shear_g = self._g if shear_gain is None else float(shear_gain)
        self._normal_g = self._g if normal_gain is None else float(normal_gain)
        self._dz = deadzone
        self._prev = set()

    @property
    def gain(self) -> float:
        return self._g

    @gain.setter
    def gain(self, value: float) -> None:
        """Set every axis gain (legacy/global control)."""
        self._g = float(value)
        self._shear_g = self._g
        self._normal_g = self._g

    @property
    def shear_gain(self) -> float:
        """Gain for the up/down wall direction only."""
        return self._shear_g

    @shear_gain.setter
    def shear_gain(self, value: float) -> None:
        self._shear_g = float(value)

    @property
    def normal_gain(self) -> float:
        """Gain for the front/back wall-normal direction only."""
        return self._normal_g

    @normal_gain.setter
    def normal_gain(self, value: float) -> None:
        self._normal_g = float(value)

    def poll(self) -> InputState:
        ax = self._src.axes()
        dt = _dz(ax.get("ly", 0.0), self._dz) * self._shear_g  # stick down = +shear
        dl = _dz(ax.get("lx", 0.0), self._dz) * self._g    # lateral
        dn = (ax.get("lt", 0.0) - ax.get("rt", 0.0)) * self._normal_g  # LT pull-off / RT press-in
        now = set(self._src.pressed())
        edges = now - self._prev
        self._prev = now
        buttons = frozenset(_BTN[b] for b in edges if b in _BTN)
        return InputState((dt, dl, dn), buttons)


class PygameSource:
    """Production source: wraps a pygame joystick. Import pygame lazily."""

    def __init__(self, index: int = 0):
        import pygame
        pygame.init(); pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad found")
        self._pg = pygame
        self._js = pygame.joystick.Joystick(index); self._js.init()

    def axes(self):
        self._pg.event.pump()
        j = self._js
        # Xbox on Linux (xpad/SDL): 0=LX,1=LY,2=LT,3=RX,4=RY,5=RT
        # (triggers rest at -1, mapped -1..1 -> 0..1)
        return {"lx": j.get_axis(0), "ly": j.get_axis(1),
                "lt": (j.get_axis(2) + 1) / 2, "rt": (j.get_axis(5) + 1) / 2}

    def pressed(self):
        j = self._js
        names = {0: "A", 1: "B", 2: "X", 3: "Y", 7: "START"}
        out = {names[i] for i in names if j.get_button(i)}
        if j.get_numhats() > 0:
            hx, hy = j.get_hat(0)
            if hx > 0:
                out.add("DPAD_RIGHT")
            elif hx < 0:
                out.add("DPAD_LEFT")
            if hy > 0:
                out.add("DPAD_UP")
            elif hy < 0:
                out.add("DPAD_DOWN")
        return out
