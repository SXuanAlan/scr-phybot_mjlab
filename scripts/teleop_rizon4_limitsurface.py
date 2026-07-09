"""Xbox teleop of the Rizon4 microspine pad with a live 3D limit-surface plot."""
import sys
import mujoco.viewer
from pyqtgraph.Qt import QtCore, QtWidgets
from scr_phybot_mjab.tools.limit_surface_teleop.pad_teleop import PadTeleop
from scr_phybot_mjab.tools.limit_surface_teleop.view import LimitSurfaceView
from scr_phybot_mjab.tools.limit_surface_teleop.xbox_input import XboxInput, PygameSource


class KeyboardSource:
    """Fallback: arrow keys / w-s map to the same axes via a Qt-captured key set."""
    def __init__(self, view_widget):
        self._keys = set(); view_widget.keyPressEvent = self._press
        view_widget.keyReleaseEvent = self._rel
    def _press(self, e): self._keys.add(e.key())
    def _rel(self, e): self._keys.discard(e.key())
    def axes(self):
        k = self._keys; Q = QtCore.Qt
        return {"lx": (Q.Key_Right in k) - (Q.Key_Left in k),
                "ly": (Q.Key_Down in k) - (Q.Key_Up in k),
                "lt": float(Q.Key_W in k), "rt": float(Q.Key_S in k)}
    def pressed(self):
        k = self._keys; Q = QtCore.Qt
        m = {Q.Key_E: "A", Q.Key_R: "B", Q.Key_C: "X", Q.Key_V: "Y", Q.Key_H: "START"}
        return {m[i] for i in m if i in k}


def main():
    app = QtWidgets.QApplication(sys.argv)
    teleop = PadTeleop()
    teleop.press_engage()
    view = LimitSurfaceView(teleop.limit_params)
    view.widget.show()
    try:
        src = PygameSource()
        print("[info] using gamepad")
    except Exception as e:
        print(f"[info] no gamepad ({e}); keyboard: arrows=shear/lateral, w/s=press/pull, "
              "e=engage r=release c=clear v=save h=home")
        src = KeyboardSource(view.widget)
    # Stick/trigger now command a Cartesian FORCE (N), not a position delta.
    # Full deflection ~= 300 N so the operator can reach the whole envelope
    # (tension/lateral caps ~15-20 N up to the ~400 N ultimate shear).
    xin = XboxInput(src, lin_gain=300.0)
    viewer = mujoco.viewer.launch_passive(teleop.mj_model, teleop.mj_data)

    def tick():
        st = xin.poll()
        if "home" in st.buttons: teleop.home(); teleop.press_engage()
        if "release" in st.buttons: teleop.executor.reset()
        if "clear" in st.buttons: view.clear()
        if "save" in st.buttons: view.save("limit_surface_manual.npz"); print("saved npz")
        r = teleop.apply(st.delta_tln)
        teleop.sync_display(); viewer.sync()
        view.update(r.load_tln, r.margin, r.released)
        if not viewer.is_running():
            timer.stop(); app.quit()

    timer = QtCore.QTimer(); timer.timeout.connect(tick); timer.start(20)  # ~50 Hz
    app.exec_()


if __name__ == "__main__":
    main()
