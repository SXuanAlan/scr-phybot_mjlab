"""Live pyqtgraph 3D view of the microspine limit surface."""
from __future__ import annotations
import numpy as np
from scr_phybot_mjab.microspine.limit_surface_geom import envelope_mesh


def _mesh_to_verts_faces(X, Y, Z):
    ni, nj = X.shape
    verts, faces, idx = [], [], -np.ones((ni, nj), int)
    for i in range(ni):
        for j in range(nj):
            if np.isfinite(Z[i, j]):
                idx[i, j] = len(verts); verts.append([X[i, j], Y[i, j], Z[i, j]])
    for i in range(ni - 1):
        for j in range(nj - 1):
            q = [idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]]
            if all(k >= 0 for k in q):
                faces.append([q[0], q[1], q[2]]); faces.append([q[0], q[2], q[3]])
    return np.array(verts, float), np.array(faces, int)


class LimitSurfaceView:
    def __init__(self, limit_params: dict, headless: bool = False, trail_len: int = 40):
        self.params = limit_params
        self.releases: list[np.ndarray] = []
        self._trail: list[np.ndarray] = []
        self._trail_len = trail_len
        self._headless = headless
        if headless:
            self.widget = None
            return
        import pyqtgraph.opengl as gl
        self.widget = gl.GLViewWidget()
        self.widget.setWindowTitle("Microspine limit surface")
        self.widget.setCameraPosition(distance=600)
        X, Y, Z = envelope_mesh(limit_params)
        verts, faces = _mesh_to_verts_faces(X, Y, Z)
        surf = gl.GLMeshItem(vertexes=verts, faces=faces, color=(0.34, 0.71, 0.91, 0.25),
                             smooth=True, drawEdges=False, glOptions="additive")
        self.widget.addItem(surf)
        self.widget.addItem(gl.GLAxisItem(size=gl.QtGui.QVector3D(400, 300, 300))
                            if hasattr(gl, "QtGui") else gl.GLAxisItem())
        self._live = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=12)
        self._trail_item = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=5)
        self._rel_item = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=6,
                                              color=(0.84, 0.37, 0.0, 1.0))
        for it in (self._trail_item, self._rel_item, self._live):
            self.widget.addItem(it)

    def _color(self, margin: float):
        m = max(-1.0, min(1.0, margin))
        if m >= 0:  # green -> yellow
            return (1.0 - m, 1.0, 0.0, 1.0)
        return (1.0, 1.0 + m, 0.0, 1.0)  # yellow -> red

    def update(self, load_tln, margin: float, released: bool):
        load = np.asarray(load_tln, float)
        self._trail.append(load.copy())
        if len(self._trail) > self._trail_len:
            self._trail.pop(0)
        if released:
            self.releases.append(load.copy())
        if self._headless:
            return
        self._live.setData(pos=load[None], color=self._color(margin))
        self._trail_item.setData(pos=np.array(self._trail))
        if self.releases:
            self._rel_item.setData(pos=np.array(self.releases))

    def clear(self):
        self.releases.clear()
        if not self._headless:
            self._rel_item.setData(pos=np.zeros((0, 3)))

    def save(self, path: str):
        np.savez(path, releases=np.array(self.releases) if self.releases else np.zeros((0, 3)))
