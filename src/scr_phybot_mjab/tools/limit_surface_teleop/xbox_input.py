"""Xbox gamepad -> engage-frame Cartesian delta + edge-detected buttons."""
from __future__ import annotations
from dataclasses import dataclass

_BTN = {"A": "engage", "B": "release", "X": "clear", "Y": "save", "START": "home"}


@dataclass
class InputState:
    delta_tln: tuple[float, float, float]
    buttons: frozenset


def _dz(v, dz):
    return 0.0 if abs(v) < dz else v


class XboxInput:
    def __init__(self, source, lin_gain: float = 0.02, deadzone: float = 0.15):
        self._src = source
        self._g = lin_gain
        self._dz = deadzone
        self._prev = set()

    def poll(self) -> InputState:
        ax = self._src.axes()
        dt = _dz(-ax.get("ly", 0.0), self._dz) * self._g   # stick up = +shear
        dl = _dz(ax.get("lx", 0.0), self._dz) * self._g    # lateral
        dn = (ax.get("lt", 0.0) - ax.get("rt", 0.0)) * self._g  # LT press-in / RT pull-off
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
        # Xbox (SDL): 0=LX,1=LY,4=LT,5=RT (triggers -1..1 -> 0..1)
        return {"lx": j.get_axis(0), "ly": j.get_axis(1),
                "lt": (j.get_axis(4) + 1) / 2, "rt": (j.get_axis(5) + 1) / 2}

    def pressed(self):
        j = self._js
        names = {0: "A", 1: "B", 2: "X", 3: "Y", 7: "START"}
        return {names[i] for i in names if j.get_button(i)}
