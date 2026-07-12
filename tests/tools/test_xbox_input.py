from scr_phybot_mjab.tools.limit_surface_teleop.xbox_input import PygameSource, XboxInput

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
    assert s.delta_tln[0] == -1.0      # t_hat points down-wall, so stick up is -t
    assert abs(s.delta_tln[1] - 0.5) < 1e-9  # lateral
    assert s.delta_tln[2] == 0.0

def test_triggers_map_to_normal():
    f = Fake(); xi = XboxInput(f, lin_gain=1.0, deadzone=0.0)
    f._ax["lt"] = 1.0; f._ax["rt"] = 0.0
    assert xi.poll().delta_tln[2] == 1.0   # LT -> pull-off (+n_out)
    f._ax["lt"] = 0.0; f._ax["rt"] = 1.0
    assert xi.poll().delta_tln[2] == -1.0  # RT -> press-in

def test_deadzone_zeroes_small_stick():
    f = Fake(); xi = XboxInput(f, lin_gain=1.0, deadzone=0.15)
    f._ax["lx"] = 0.1
    assert xi.poll().delta_tln[1] == 0.0

def test_button_edges():
    f = Fake(); xi = XboxInput(f, lin_gain=1.0)
    f._btn = {"A"}
    assert xi.poll().buttons == frozenset({"home"})
    assert xi.poll().buttons == frozenset()  # still held -> no re-fire
    f._btn = set(); xi.poll()
    f._btn = {"START"}
    assert xi.poll().buttons == frozenset({"engage"})

def test_shear_and_normal_gains_are_independent():
    f = Fake(); xi = XboxInput(
        f, lin_gain=2.0, shear_gain=3.0, normal_gain=4.0, deadzone=0.0
    )
    f._ax.update({"lx": 1.0, "ly": 1.0, "lt": 1.0})
    assert xi.poll().delta_tln == (3.0, 2.0, 4.0)

    xi.shear_gain = 5.0
    assert xi.poll().delta_tln == (5.0, 2.0, 4.0)
    xi.normal_gain = 6.0
    assert xi.poll().delta_tln == (5.0, 2.0, 6.0)

def test_dpad_horizontal_selects_normal_and_vertical_selects_shear():
    f = Fake(); xi = XboxInput(f)
    f._btn = {"DPAD_RIGHT"}
    assert xi.poll().buttons == frozenset({"normal_gain_up"})
    assert xi.poll().buttons == frozenset()
    f._btn = set(); xi.poll()
    f._btn = {"DPAD_LEFT"}
    assert xi.poll().buttons == frozenset({"normal_gain_down"})
    f._btn = set(); xi.poll()
    f._btn = {"DPAD_UP"}
    assert xi.poll().buttons == frozenset({"shear_gain_up"})
    f._btn = set(); xi.poll()
    f._btn = {"DPAD_DOWN"}
    assert xi.poll().buttons == frozenset({"shear_gain_down"})

def test_pygame_hat_reports_both_force_range_axes():
    class Joystick:
        def __init__(self, hat): self.hat = hat
        def get_button(self, _): return False
        def get_numhats(self): return 1
        def get_hat(self, _): return self.hat

    source = object.__new__(PygameSource)
    source._js = Joystick((1, 0))
    assert source.pressed() == {"DPAD_RIGHT"}
    source._js.hat = (-1, 0)
    assert source.pressed() == {"DPAD_LEFT"}
    source._js.hat = (0, 1)
    assert source.pressed() == {"DPAD_UP"}
    source._js.hat = (0, -1)
    assert source.pressed() == {"DPAD_DOWN"}
    source._js.hat = (1, 1)
    assert source.pressed() == {"DPAD_RIGHT", "DPAD_UP"}
