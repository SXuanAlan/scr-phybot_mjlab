"""Xbox teleop of the Rizon4 microspine pad with a live 3D limit-surface plot."""
import signal
import sys
import mujoco.viewer
from pyqtgraph.Qt import QtCore, QtWidgets
from scr_phybot_mjab.tools.limit_surface_teleop.pad_teleop import PadTeleop
from scr_phybot_mjab.tools.limit_surface_teleop.view import LimitSurfaceView
from scr_phybot_mjab.tools.limit_surface_teleop.xbox_input import XboxInput, PygameSource


class KeyboardSource:
    """Keyboard fallback matching the stick, triggers, and D-pad controls."""
    def __init__(self, view_widget):
        self._keys = set(); view_widget.keyPressEvent = self._press
        view_widget.keyReleaseEvent = self._rel
    def _press(self, e): self._keys.add(e.key())
    def _rel(self, e): self._keys.discard(e.key())
    def axes(self):
        k = self._keys; Q = QtCore.Qt
        return {"lx": (Q.Key_D in k) - (Q.Key_A in k),
                "ly": (Q.Key_K in k) - (Q.Key_I in k),
                "lt": float(Q.Key_W in k), "rt": float(Q.Key_S in k)}
    def pressed(self):
        k = self._keys; Q = QtCore.Qt
        m = {Q.Key_E: "START", Q.Key_R: "B", Q.Key_C: "X", Q.Key_V: "Y", Q.Key_H: "A",
             Q.Key_Right: "DPAD_RIGHT", Q.Key_Left: "DPAD_LEFT",
             Q.Key_Up: "DPAD_UP", Q.Key_Down: "DPAD_DOWN"}
        return {m[i] for i in m if i in k}


# Stick/trigger command a Cartesian FORCE (N), not a position delta. D-pad
# up/down changes the up/down shear-force range; D-pad left/right changes the
# front/back wall-normal force range. The two ranges are independent.
# Small default: the lateral/tension release caps are only ~15-20 N, so 300 N
# full-stick trips a forced release on the slightest push. Go up (150/300 N)
# only when hunting the ~400 N forward-shear ultimate.
FORCE_GAIN_PRESETS = (15.0, 30.0, 60.0, 150.0, 300.0)
DEFAULT_SHEAR_GAIN_INDEX = 1  # 30 N
DEFAULT_NORMAL_GAIN_INDEX = 1  # 30 N


def main():
    app = QtWidgets.QApplication(sys.argv)
    teleop = PadTeleop()
    view = LimitSurfaceView(teleop.limit_params)
    view.widget.show()
    try:
        src = PygameSource()
        print("[info] using gamepad")
    except Exception as e:
        print(f"[info] no gamepad ({e}); keyboard: i/k=up/down force, a/d=lateral, "
              "w/s=front/back force; arrows change force ranges; "
              "e=engage r=release c=clear v=save h=home")
        src = KeyboardSource(view.widget)
    shear_gain_idx = DEFAULT_SHEAR_GAIN_INDEX
    normal_gain_idx = DEFAULT_NORMAL_GAIN_INDEX
    xin = XboxInput(
        src,
        lin_gain=FORCE_GAIN_PRESETS[DEFAULT_SHEAR_GAIN_INDEX],
        shear_gain=FORCE_GAIN_PRESETS[shear_gain_idx],
        normal_gain=FORCE_GAIN_PRESETS[normal_gain_idx],
    )
    last_engaged = False
    print("[microspine] DISENGAGED (start away from wall; START/e = engage, A/h = home)")
    print(
        f"[force ranges] up/down ±{FORCE_GAIN_PRESETS[shear_gain_idx]:.0f} N "
        f"(D-pad ↑/↓), front/back ±{FORCE_GAIN_PRESETS[normal_gain_idx]:.0f} N "
        "(D-pad ←/→)"
    )
    viewer = mujoco.viewer.launch_passive(teleop.mj_model, teleop.mj_data)

    def tick():
        nonlocal shear_gain_idx, normal_gain_idx, last_engaged
        st = xin.poll()
        if "home" in st.buttons: teleop.home()
        if "engage" in st.buttons: teleop.press_engage()
        if "release" in st.buttons: teleop.executor.reset()
        if "clear" in st.buttons: view.clear()
        if "save" in st.buttons: view.save("limit_surface_manual.npz"); print("saved npz")
        if "shear_gain_up" in st.buttons or "shear_gain_down" in st.buttons:
            shear_gain_idx = min(shear_gain_idx + 1, len(FORCE_GAIN_PRESETS) - 1) \
                if "shear_gain_up" in st.buttons else max(shear_gain_idx - 1, 0)
            xin.shear_gain = FORCE_GAIN_PRESETS[shear_gain_idx]
            print(f"[up/down force range] ±{FORCE_GAIN_PRESETS[shear_gain_idx]:.0f} N")
        if "normal_gain_up" in st.buttons or "normal_gain_down" in st.buttons:
            normal_gain_idx = min(normal_gain_idx + 1, len(FORCE_GAIN_PRESETS) - 1) \
                if "normal_gain_up" in st.buttons else max(normal_gain_idx - 1, 0)
            xin.normal_gain = FORCE_GAIN_PRESETS[normal_gain_idx]
            print(f"[front/back force range] ±{FORCE_GAIN_PRESETS[normal_gain_idx]:.0f} N")
        r = teleop.apply(st.delta_tln)
        if r.engaged != last_engaged:
            if r.engaged:
                print("[microspine] ENGAGED")
            else:
                print("[microspine] FORCED RELEASE (load left the limit surface)"
                      if r.released else "[microspine] DISENGAGED")
            last_engaged = r.engaged
        teleop.sync_display(); viewer.sync()
        view.update(r.load_tln, r.margin, r.released)
        if not viewer.is_running():
            timer.stop(); app.quit()

    timer = QtCore.QTimer(); timer.timeout.connect(tick); timer.start(20)  # ~50 Hz
    # Qt's C++ event loop blocks Python's default SIGINT handling; restore the
    # OS default so Ctrl-C in the terminal kills the app immediately.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app.exec_()


if __name__ == "__main__":
    main()
