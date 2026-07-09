from scr_phybot_mjab.tools.limit_surface_teleop.xbox_input import XboxInput, InputState

class Fake:
    def __init__(self): self._ax = {"lx": 0, "ly": 0, "lt": 0, "rt": 0}; self._btn = set()
    def axes(self): return self._ax
    def pressed(self): return self._btn

def test_stick_maps_to_shear_lateral_with_deadzone():
    f = Fake(); xi = XboxInput(f, lin_gain=1.0, deadzone=0.15)
    f._ax["ly"] = -1.0  # stick up
    f._ax["lx"] = 0.5
    f._ax["lt"] = 0.0; f._ax["rt"] = 0.0
    s = xi.poll()
    assert s.delta_tln[0] > 0          # shear up (t)
    assert abs(s.delta_tln[1] - 0.5) < 1e-9  # lateral
    assert s.delta_tln[2] == 0.0

def test_triggers_map_to_normal():
    f = Fake(); xi = XboxInput(f, lin_gain=1.0, deadzone=0.0)
    f._ax["lt"] = 1.0; f._ax["rt"] = 0.0
    assert xi.poll().delta_tln[2] == 1.0   # LT -> +1 (press-in convention)
    f._ax["lt"] = 0.0; f._ax["rt"] = 1.0
    assert xi.poll().delta_tln[2] == -1.0  # RT -> pull-off

def test_deadzone_zeroes_small_stick():
    f = Fake(); xi = XboxInput(f, lin_gain=1.0, deadzone=0.15)
    f._ax["lx"] = 0.1
    assert xi.poll().delta_tln[1] == 0.0

def test_button_edges():
    f = Fake(); xi = XboxInput(f, lin_gain=1.0)
    f._btn = {"A"}
    assert xi.poll().buttons == frozenset({"engage"})
    assert xi.poll().buttons == frozenset()  # still held -> no re-fire
