"""Live scientific 3D view of the microspine load limit surface.

The scene uses the anchor load frame throughout: ``x = f_t`` (tangential
shear), ``y = f_l`` (lateral load), and ``z = f_n`` (wall-normal tension), all
in newtons.  The admissible set is unbounded in compression, so its visible
faces are clipped only for display at the lower z-axis limit; that clipping
plane is deliberately not rendered as a physical envelope face.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

_LIMIT_KEYS = (
    "f_t_max",
    "f_rev_max",
    "mu_adhesion",
    "f_n0",
    "mu_lateral",
    "f_l0",
)

# Okabe-Ito-derived, color-blind-safe palette on white.
_ENVELOPE_RGB = (0.337, 0.706, 0.914)  # sky blue, #56B4E9
_WIRE_RGB = (0.000, 0.447, 0.698)  # blue, #0072B2
_FEATURE_RGB = (0.000, 0.353, 0.553)  # darker structural blue, #005A8D
_TRAIL_RGB = (0.302, 0.349, 0.400)  # neutral slate, #4D5966
_SAFE_RGB = (0.000, 0.620, 0.451)  # bluish green, #009E73
_WARNING_RGB = (0.902, 0.624, 0.000)  # orange, #E69F00
_RELEASE_RGB = (0.835, 0.369, 0.000)  # vermillion, #D55E00
_INK = (0.125, 0.129, 0.141, 1.0)  # #202124
_EMPTY_POINTS = np.empty((0, 3), dtype=float)


def _validate_limit_params(limit_params: Mapping[str, float]) -> dict[str, float]:
    """Return finite scalar parameters and reject degenerate envelopes."""
    missing = [key for key in _LIMIT_KEYS if key not in limit_params]
    if missing:
        raise ValueError(f"missing limit-surface parameters: {', '.join(missing)}")

    params = {key: float(limit_params[key]) for key in _LIMIT_KEYS}
    nonfinite = [key for key, value in params.items() if not math.isfinite(value)]
    if nonfinite:
        raise ValueError(f"non-finite limit-surface parameters: {', '.join(nonfinite)}")

    positive = ("f_t_max", "f_rev_max", "f_n0", "f_l0")
    nonpositive = [key for key in positive if params[key] <= 0.0]
    if nonpositive:
        raise ValueError(f"limit-surface parameters must be positive: {', '.join(nonpositive)}")
    if params["mu_adhesion"] < 0.0 or params["mu_lateral"] < 0.0:
        raise ValueError("limit-surface slopes must be non-negative")
    return params


def _mesh_to_verts_faces(X, Y, Z):
    """Convert a rectangular surface grid into indexed triangle arrays."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if X.ndim != 2 or X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError("X, Y, and Z must be equally shaped 2D arrays")

    ni, nj = X.shape
    valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    index = np.full((ni, nj), -1, dtype=int)
    vertices: list[tuple[float, float, float]] = []
    for i in range(ni):
        for j in range(nj):
            if valid[i, j]:
                index[i, j] = len(vertices)
                vertices.append((X[i, j], Y[i, j], Z[i, j]))

    faces: list[tuple[int, int, int]] = []
    for i in range(ni - 1):
        for j in range(nj - 1):
            q00, q10 = index[i, j], index[i + 1, j]
            q11, q01 = index[i + 1, j + 1], index[i, j + 1]
            if min(q00, q10, q11, q01) >= 0:
                faces.extend(((q00, q10, q11), (q00, q11, q01)))

    return (
        np.asarray(vertices, dtype=float).reshape(-1, 3),
        np.asarray(faces, dtype=int).reshape(-1, 3),
    )


def _t_samples(params: Mapping[str, float], count: int = 49) -> np.ndarray:
    """Sample both sides of the f_t=0 crease without duplicating the crease."""
    count = max(int(count), 9)
    intervals = count - 1
    reverse_fraction = params["f_rev_max"] / (params["f_rev_max"] + params["f_t_max"])
    reverse_intervals = round(intervals * reverse_fraction)
    reverse_intervals = min(intervals - 3, max(3, reverse_intervals))
    forward_intervals = intervals - reverse_intervals
    reverse = np.linspace(-params["f_rev_max"], 0.0, reverse_intervals, endpoint=False)
    forward = np.linspace(0.0, params["f_t_max"], forward_intervals + 1)
    return np.concatenate((reverse, forward))


def _capacities(params: Mapping[str, float], f_t: np.ndarray | float):
    f_t_pos = np.maximum(f_t, 0.0)
    lateral = params["mu_lateral"] * f_t_pos + params["f_l0"]
    normal = params["mu_adhesion"] * f_t_pos + params["f_n0"]
    return lateral, normal


def _envelope_patches(
    params: Mapping[str, float],
    z_floor: float,
    *,
    n_t: int = 49,
    n_cross: int = 11,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build the five physical boundary faces, clipped at ``z_floor``.

    The returned faces are the tension cap, both lateral caps, reverse-shear
    cap, and ultimate-shear cap.  There is intentionally no bottom face: the
    admissible load set continues without bound toward compression.
    """
    t = _t_samples(params, n_t)
    lateral, normal = _capacities(params, t)
    u = np.linspace(-1.0, 1.0, max(3, int(n_cross)))
    v = np.linspace(0.0, 1.0, max(3, int(n_cross)))

    patches = [
        (
            np.broadcast_to(t[:, None], (t.size, u.size)).copy(),
            lateral[:, None] * u[None, :],
            np.broadcast_to(normal[:, None], (t.size, u.size)).copy(),
        )
    ]

    for sign in (-1.0, 1.0):
        patches.append(
            (
                np.broadcast_to(t[:, None], (t.size, v.size)).copy(),
                sign * np.broadcast_to(lateral[:, None], (t.size, v.size)).copy(),
                z_floor + (normal[:, None] - z_floor) * v[None, :],
            )
        )

    for t_value in (-params["f_rev_max"], params["f_t_max"]):
        lateral_cap, normal_cap = _capacities(params, t_value)
        Y = float(lateral_cap) * np.broadcast_to(u[:, None], (u.size, v.size))
        Z = z_floor + (float(normal_cap) - z_floor) * np.broadcast_to(v[None, :], (u.size, v.size))
        patches.append((np.full_like(Y, t_value), Y.copy(), Z.copy()))
    return patches


def _wireframe_polylines(
    patch: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    rows: int = 7,
    columns: int = 7,
    omit_lower_z_boundary: bool = False,
) -> list[np.ndarray]:
    """Return a sparse parametric wireframe for one rectangular patch."""
    X, Y, Z = patch
    row_ids = np.unique(np.linspace(0, X.shape[0] - 1, min(rows, X.shape[0]), dtype=int))
    col_ids = np.unique(np.linspace(0, X.shape[1] - 1, min(columns, X.shape[1]), dtype=int))
    lines = [np.column_stack((X[i], Y[i], Z[i])) for i in row_ids]
    lines.extend(np.column_stack((X[:, j], Y[:, j], Z[:, j])) for j in col_ids)
    if omit_lower_z_boundary:
        z_floor = float(np.min(Z))
        lines = [line for line in lines if not np.allclose(line[:, 2], z_floor)]
    return lines


def _polyline_segments(polylines: list[np.ndarray]) -> np.ndarray:
    """Flatten independent polylines for ``GLLinePlotItem(mode='lines')``."""
    segments = [
        np.stack((line[:-1], line[1:]), axis=1).reshape(-1, 3)
        for line in polylines
        if len(line) >= 2
    ]
    return np.concatenate(segments, axis=0) if segments else _EMPTY_POINTS.copy()


def _envelope_feature_edges(params: Mapping[str, float], z_floor: float) -> list[np.ndarray]:
    """Return the true cap intersections and the f_t=0 surface crease."""
    t = _t_samples(params)
    lateral, normal = _capacities(params, t)
    edges = [
        np.column_stack((t, -lateral, normal)),
        np.column_stack((t, lateral, normal)),
    ]
    for t_value in (-params["f_rev_max"], 0.0, params["f_t_max"]):
        lateral_cap, normal_cap = _capacities(params, t_value)
        lateral_cap = float(lateral_cap)
        normal_cap = float(normal_cap)
        edges.extend(
            (
                np.array(((t_value, -lateral_cap, normal_cap), (t_value, lateral_cap, normal_cap))),
                np.array(((t_value, -lateral_cap, z_floor), (t_value, -lateral_cap, normal_cap))),
                np.array(((t_value, lateral_cap, z_floor), (t_value, lateral_cap, normal_cap))),
            )
        )
    return edges


def _nice_step(span: float, target_intervals: int = 6) -> float:
    """Choose a conventional 1/2/2.5/5/10 major-tick interval."""
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("axis span must be finite and positive")
    raw = span / max(1, int(target_intervals))
    magnitude = 10.0 ** math.floor(math.log10(raw))
    scaled = raw / magnitude
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if scaled <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def _axis_scale(
    data_min: float,
    data_max: float,
    *,
    target_intervals: int = 6,
    padding: float = 0.06,
) -> tuple[tuple[float, float], np.ndarray, float]:
    """Return padded limits, internal nice labeled ticks, and the tick step."""
    if not math.isfinite(data_min) or not math.isfinite(data_max) or data_max <= data_min:
        raise ValueError("axis data limits must be finite and increasing")
    span = data_max - data_min
    padded_min = data_min - padding * span
    padded_max = data_max + padding * span
    step = _nice_step(padded_max - padded_min, target_intervals)
    # Keep the view close to the physical data.  Only the ticks, not the axis
    # limits, are snapped to the nice interval; snapping both used to add large
    # empty margins and crowd the projected labels.
    axis_min = padded_min
    axis_max = padded_max
    first_tick = math.ceil(axis_min / step)
    last_tick = math.floor(axis_max / step)
    ticks = step * np.arange(first_tick, last_tick + 1, dtype=float)
    ticks[np.abs(ticks) < step * 1.0e-9] = 0.0
    return (axis_min, axis_max), ticks, step


def _format_tick(value: float, step: float) -> str:
    if abs(value) < step * 1.0e-9:
        value = 0.0
    decimals = 0
    while decimals < 8 and not math.isclose(
        step * 10**decimals,
        round(step * 10**decimals),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        decimals += 1
    return f"{value:.{decimals}f}"


class LimitSurfaceView:
    """Interactive pyqtgraph view with a persistent 3D load trajectory."""

    def __init__(
        self,
        limit_params: Mapping[str, float],
        headless: bool = False,
        trail_len: int = 400,
    ):
        if int(trail_len) <= 0:
            raise ValueError("trail_len must be positive")

        self.params = dict(limit_params)
        self._plot_params = _validate_limit_params(limit_params)
        self.releases: list[np.ndarray] = []
        self._trail: list[np.ndarray] = []
        self._trail_len = int(trail_len)
        self._headless = bool(headless)
        self._last_load: np.ndarray | None = None
        self._last_margin = math.nan
        self._last_hud_state: str | None = None
        self.widget = None
        if self._headless:
            return

        from pyqtgraph.Qt import QtWidgets
        if QtWidgets.QApplication.instance() is None:
            # Preserve the original no-display contract: bookkeeping remains
            # usable even when a caller did not start Qt, and no QWidget is
            # constructed (which would otherwise abort the process in C++).
            self._headless = True
            return

        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
        from OpenGL import GL
        from pyqtgraph.Qt import QtCore, QtGui

        class GLMathTextItem(gl.GLTextItem):
            """Screen-facing rich mathematical text anchored in 3D."""

            def __init__(self, *, html: str, **kwargs):
                super().__init__(text="", **kwargs)
                self._html = html
                self._document = QtGui.QTextDocument()
                self._document.setDocumentMargin(0.0)
                self._document.setDefaultFont(self.font)
                self._document.setHtml(html)
                self._document.adjustSize()

            def paint(self):
                if not self._html:
                    return
                self.setupGLState()
                anchor = self.compute_projection().map(
                    QtGui.QVector3D(*self.pos)
                ).toPointF()
                size = self._document.size()
                x, y = anchor.x(), anchor.y()
                if self.alignment & QtCore.Qt.AlignmentFlag.AlignRight:
                    x -= size.width()
                elif self.alignment & QtCore.Qt.AlignmentFlag.AlignHCenter:
                    x -= 0.5 * size.width()
                if self.alignment & QtCore.Qt.AlignmentFlag.AlignBottom:
                    y -= size.height()
                elif self.alignment & QtCore.Qt.AlignmentFlag.AlignVCenter:
                    y -= 0.5 * size.height()

                painter = QtGui.QPainter(self.view())
                painter.setRenderHints(
                    QtGui.QPainter.RenderHint.Antialiasing
                    | QtGui.QPainter.RenderHint.TextAntialiasing
                )
                painter.translate(x, y)
                context = QtGui.QAbstractTextDocumentLayout.PaintContext()
                context.palette.setColor(QtGui.QPalette.ColorRole.Text, self.color)
                self._document.documentLayout().draw(painter, context)
                painter.end()

        self._gl = gl
        p = self._plot_params
        lateral_max, normal_max = _capacities(p, p["f_t_max"])
        lateral_max = float(lateral_max)
        normal_max = float(normal_max)

        x_scale = _axis_scale(-p["f_rev_max"], p["f_t_max"])
        y_scale = _axis_scale(-lateral_max, lateral_max)
        # Compression has no physical envelope boundary.  Show a useful finite
        # portion of it and make the display truncation explicit in the HUD.
        z_data_min = -0.28 * normal_max
        z_scale = _axis_scale(z_data_min, normal_max)
        self._axis_limits = (x_scale[0], y_scale[0], z_scale[0])
        self._axis_ticks = (x_scale[1], y_scale[1], z_scale[1])
        self._axis_steps = (x_scale[2], y_scale[2], z_scale[2])
        self._z_floor = z_scale[0][0]

        x_limits, y_limits, z_limits = self._axis_limits
        x_span = x_limits[1] - x_limits[0]
        y_span = y_limits[1] - y_limits[0]
        z_span = z_limits[1] - z_limits[0]

        widget = gl.GLViewWidget()
        widget.setBackgroundColor((255, 255, 255, 255))
        widget.setWindowTitle("Microspine limit surface — 3D load space")
        widget.resize(1600, 1000)
        widget.setMinimumSize(1100, 760)
        center = np.array(
            (
                0.5 * (x_limits[0] + x_limits[1]),
                0.5 * (y_limits[0] + y_limits[1]),
                0.5 * (z_limits[0] + z_limits[1]) - 0.18 * z_span,
            )
        )
        widget.opts["center"] = pg.Vector(*center)
        scene_diagonal = float(np.linalg.norm((x_span, y_span, z_span)))
        widget.setCameraPosition(distance=1.18 * scene_diagonal, elevation=24, azimuth=-52)
        self.widget = widget

        # The f_n=0 reference plane is physically meaningful and also provides
        # depth cues.  Draw it explicitly so every line matches a labeled tick
        # even when the padded axis limits are not multiples of the tick step.
        grid_segments = []
        for value in x_scale[1]:
            grid_segments.extend(((value, y_limits[0], 0.0), (value, y_limits[1], 0.0)))
        for value in y_scale[1]:
            grid_segments.extend(((x_limits[0], value, 0.0), (x_limits[1], value, 0.0)))
        grid = gl.GLLinePlotItem(
            pos=np.asarray(grid_segments, dtype=float).reshape(-1, 3),
            color=(0.796, 0.824, 0.851, 0.58),
            width=0.8,
            mode="lines",
            antialias=True,
            glOptions="translucent",
        )
        grid.setDepthValue(-30)
        widget.addItem(grid)
        self._grid = grid

        # Translucent physical faces plus a sparse parametric wireframe.  A
        # separate darker feature layer makes the true cap intersections crisp.
        self._surface_items = []
        self._wire_items = []
        patches = _envelope_patches(p, self._z_floor)
        face_alphas = (0.18, 0.105, 0.105, 0.085, 0.085)
        for patch_index, (patch, alpha) in enumerate(zip(patches, face_alphas, strict=True)):
            vertices, faces = _mesh_to_verts_faces(*patch)
            surface = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                color=(*_ENVELOPE_RGB, alpha),
                drawEdges=False,
                smooth=False,
                computeNormals=False,
                glOptions="translucent",
            )
            surface.setDepthValue(-20)
            widget.addItem(surface)
            self._surface_items.append(surface)

            wire = gl.GLLinePlotItem(
                pos=_polyline_segments(
                    _wireframe_polylines(patch, omit_lower_z_boundary=patch_index > 0)
                ),
                color=(*_WIRE_RGB, 0.38),
                width=1.0,
                mode="lines",
                antialias=True,
                glOptions="translucent",
            )
            wire.setDepthValue(-10)
            widget.addItem(wire)
            self._wire_items.append(wire)

        feature_edges = gl.GLLinePlotItem(
            pos=_polyline_segments(_envelope_feature_edges(p, self._z_floor)),
            color=(*_FEATURE_RGB, 0.92),
            width=1.8,
            mode="lines",
            antialias=True,
            glOptions="translucent",
        )
        feature_edges.setDepthValue(-5)
        widget.addItem(feature_edges)
        self._feature_edges = feature_edges

        # Explicit black axes, major ticks, and screen-facing labels replace
        # GLAxisItem, whose colors and unlabeled scale are unsuitable on white.
        # Put the labeled axes at a plot-box corner so their values do not run
        # through the translucent envelope; faint zero lines retain the load
        # frame's physical origin on the f_n=0 reference plane.
        axis_font = QtGui.QFont("DejaVu Serif", 15)
        tick_font = QtGui.QFont("DejaVu Sans", 10)
        self._axis_items = []
        self._axis_text_items = []

        def add_segments(segments, color=_INK, width=1.4):
            item = gl.GLLinePlotItem(
                pos=np.asarray(segments, dtype=float).reshape(-1, 3),
                color=color,
                width=width,
                mode="lines",
                antialias=True,
                glOptions="opaque",
            )
            item.setDepthValue(10)
            widget.addItem(item)
            self._axis_items.append(item)
            return item

        def add_text(
            position,
            text,
            *,
            font=tick_font,
            color=(32, 33, 36),
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom,
        ):
            item = gl.GLTextItem(
                pos=np.asarray(position, dtype=float),
                text=text,
                color=color,
                font=font,
                alignment=alignment,
            )
            item.setDepthValue(30)
            widget.addItem(item)
            self._axis_text_items.append(item)
            return item

        def add_math_text(position, html, *, alignment):
            item = GLMathTextItem(
                pos=np.asarray(position, dtype=float),
                html=html,
                color=(32, 33, 36),
                font=axis_font,
                alignment=alignment,
            )
            item.setDepthValue(31)
            widget.addItem(item)
            self._axis_text_items.append(item)
            return item

        add_segments(
            (
                (x_limits[0], y_limits[0], z_limits[0]),
                (x_limits[1], y_limits[0], z_limits[0]),
                (x_limits[0], y_limits[0], z_limits[0]),
                (x_limits[0], y_limits[1], z_limits[0]),
                (x_limits[0], y_limits[0], z_limits[0]),
                (x_limits[0], y_limits[0], z_limits[1]),
            ),
            width=1.6,
        )
        add_segments(
            (
                (x_limits[0], 0.0, 0.0),
                (x_limits[1], 0.0, 0.0),
                (0.0, y_limits[0], 0.0),
                (0.0, y_limits[1], 0.0),
                (0.0, 0.0, z_limits[0]),
                (0.0, 0.0, z_limits[1]),
            ),
            color=(0.56, 0.59, 0.63, 1.0),
            width=0.8,
        )

        tick_x = 0.012 * x_span
        tick_y = 0.012 * y_span
        x_tick_segments = []
        for value in x_scale[1]:
            x_tick_segments.extend(
                ((value, y_limits[0] - tick_y, z_limits[0]), (value, y_limits[0] + tick_y, z_limits[0]))
            )
            add_text(
                (value, y_limits[0] - 3.4 * tick_y, z_limits[0]),
                _format_tick(value, x_scale[2]),
                alignment=(
                    QtCore.Qt.AlignmentFlag.AlignHCenter
                    | QtCore.Qt.AlignmentFlag.AlignTop
                ),
            )
        add_segments(x_tick_segments, width=1.1)

        y_tick_segments = []
        for value in y_scale[1]:
            y_tick_segments.extend(
                ((x_limits[0] - tick_x, value, z_limits[0]), (x_limits[0] + tick_x, value, z_limits[0]))
            )
            if value != y_limits[0]:
                add_text(
                    (x_limits[0] - 4.2 * tick_x, value, z_limits[0]),
                    _format_tick(value, y_scale[2]),
                    alignment=(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    ),
                )
        add_segments(y_tick_segments, width=1.1)

        z_tick_segments = []
        for value in z_scale[1]:
            z_tick_segments.extend(
                ((x_limits[0] - tick_x, y_limits[0], value), (x_limits[0] + tick_x, y_limits[0], value))
            )
            if value != z_limits[0]:
                add_text(
                    (x_limits[0] - 4.2 * tick_x, y_limits[0], value),
                    _format_tick(value, z_scale[2]),
                    alignment=(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    ),
                )
        add_segments(z_tick_segments, width=1.1)

        add_math_text(
            (x_limits[1] + 0.025 * x_span, y_limits[0], z_limits[0]),
            "<i>f</i><sub><i>t</i></sub>&thinsp;[N]",
            alignment=(
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            ),
        )
        add_math_text(
            (x_limits[0] - 0.13 * x_span, 0.0, z_limits[0]),
            "<i>f</i><sub><i>l</i></sub>&thinsp;[N]",
            alignment=(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            ),
        )
        add_math_text(
            (x_limits[0], y_limits[0], z_limits[1] + 0.075 * z_span),
            "<i>f</i><sub><i>n</i></sub>&thinsp;[N]",
            alignment=(
                QtCore.Qt.AlignmentFlag.AlignHCenter
                | QtCore.Qt.AlignmentFlag.AlignBottom
            ),
        )

        # Dynamic layers: faded line history, white marker halos, the live load,
        # all recorded releases, and one annotation for the latest release.
        self._trail_item = gl.GLLinePlotItem(
            pos=_EMPTY_POINTS.copy(),
            color=(*_TRAIL_RGB, 0.82),
            width=2.2,
            mode="line_strip",
            antialias=True,
            glOptions="translucent",
        )
        self._release_halo = gl.GLScatterPlotItem(
            pos=_EMPTY_POINTS.copy(), size=17, color=(1.0, 1.0, 1.0, 0.95), pxMode=True, glOptions="opaque"
        )
        self._rel_item = gl.GLScatterPlotItem(
            pos=_EMPTY_POINTS.copy(), size=12, color=(*_RELEASE_RGB, 1.0), pxMode=True, glOptions="opaque"
        )
        self._live_halo = gl.GLScatterPlotItem(
            pos=_EMPTY_POINTS.copy(), size=19, color=(1.0, 1.0, 1.0, 0.95), pxMode=True, glOptions="opaque"
        )
        self._live = gl.GLScatterPlotItem(
            pos=_EMPTY_POINTS.copy(), size=13, color=self._color(math.nan), pxMode=True, glOptions="opaque"
        )
        for depth, item in enumerate(
            (self._trail_item, self._release_halo, self._rel_item, self._live_halo, self._live), start=20
        ):
            item.setDepthValue(depth)
            widget.addItem(item)

        self._release_label = gl.GLTextItem(
            pos=np.zeros(3), text="", color=(213, 94, 0), font=QtGui.QFont("DejaVu Sans", 10)
        )
        self._release_label.setDepthValue(40)
        widget.addItem(self._release_label)

        # The transparent envelope otherwise writes to the depth buffer before
        # the interior trajectory is drawn.  Standard alpha blending without a
        # depth test keeps the measured path and event markers readable through
        # all viewing angles (unlike pyqtgraph's additive preset on white).
        overlay_gl_options = {
            GL.GL_DEPTH_TEST: False,
            GL.GL_BLEND: True,
            GL.GL_CULL_FACE: False,
            "glBlendFuncSeparate": (
                GL.GL_SRC_ALPHA,
                GL.GL_ONE_MINUS_SRC_ALPHA,
                GL.GL_ONE,
                GL.GL_ONE_MINUS_SRC_ALPHA,
            ),
        }
        for item in (
            self._trail_item,
            self._release_halo,
            self._rel_item,
            self._live_halo,
            self._live,
        ):
            item.setGLOptions(overlay_gl_options)

        # HUD and legend are true 2D overlays, so values and mathematical
        # notation remain legible while the user rotates the 3D scene.
        self._hud = QtWidgets.QLabel(widget)
        self._hud.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._hud.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._hud.setFont(QtGui.QFont("DejaVu Sans", 10))
        self._hud.setMargin(0)

        self._legend = QtWidgets.QLabel(widget)
        self._legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._legend.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._legend.setText(
            "<div style='color:#202124; font-family:DejaVu Sans; font-size:10pt;'>"
            "<b>LEGEND</b><br>"
            "<table cellspacing='3' cellpadding='1'>"
            "<tr><td><span style='color:#56B4E9; font-size:15pt;'>■</span>"
            "<span style='color:#0072B2;'>━</span></td>"
            "<td style='font-family:DejaVu Serif;'>∂<i>𝒜</i></td>"
            "<td>limit envelope</td></tr>"
            "<tr><td><span style='color:#4D5966;'>━━━━</span></td>"
            "<td style='font-family:DejaVu Serif;'><b><i>f</i></b>(<i>t</i>)</td>"
            "<td>load path</td></tr>"
            "<tr><td><span style='color:#009E73; font-size:14pt;'>●</span></td>"
            "<td style='font-family:DejaVu Serif;'><b><i>f</i></b><sub><i>k</i></sub></td>"
            "<td>current load</td></tr>"
            "<tr><td><span style='color:#D55E00; font-size:14pt;'>●</span></td>"
            "<td style='font-family:DejaVu Serif;'><b><i>f</i></b><sub>rel</sub></td>"
            "<td>release</td></tr>"
            "</table></div>"
        )
        self._legend.setStyleSheet(
            "QLabel {"
            " background-color: rgba(255, 255, 255, 232);"
            " border: 1px solid rgba(74, 85, 104, 105);"
            " border-radius: 3px;"
            " padding: 9px 11px;"
            "}"
        )

        overlay_layout = QtWidgets.QGridLayout(widget)
        overlay_layout.setContentsMargins(18, 18, 18, 18)
        overlay_layout.setHorizontalSpacing(24)
        overlay_layout.setVerticalSpacing(0)
        overlay_layout.addWidget(
            self._hud,
            0,
            0,
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
        )
        overlay_layout.addWidget(
            self._legend,
            0,
            1,
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop,
        )
        overlay_layout.setColumnStretch(0, 1)
        overlay_layout.setColumnStretch(1, 1)
        overlay_layout.setRowStretch(1, 1)
        self._overlay_layout = overlay_layout

        self._update_hud(None, math.nan, released=False)
        self._hud.show()
        self._legend.show()

    def _color(self, margin: float):
        """Map envelope margin to safe teal, warning amber, and failure red."""
        try:
            value = float(margin)
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value):
            return (0.40, 0.43, 0.47, 1.0)

        value = max(-1.0, min(1.0, value))
        safe = np.array(_SAFE_RGB)
        warning = np.array(_WARNING_RGB)
        failure = np.array(_RELEASE_RGB)
        if value >= 0.0:
            rgb = warning + value * (safe - warning)
        else:
            rgb = warning + (-value) * (failure - warning)
        return (*rgb.tolist(), 1.0)

    def _outside_axis_limits(self, load: np.ndarray) -> bool:
        if self._headless:
            return False
        return any(load[i] < limits[0] or load[i] > limits[1] for i, limits in enumerate(self._axis_limits))

    def _update_hud(self, load: np.ndarray | None, margin: float, *, released: bool):
        if load is None:
            load_values = ("--.-", "--.-", "--.-")
            margin_text = "--.--"
            state = "WAITING FOR DATA"
            accent = "#6b7280"
            outside = False
        else:
            load_values = tuple(f"{value:+.1f}" for value in load)
            margin_text = f"{margin:+.3f}" if math.isfinite(margin) else "  nan"
            outside = self._outside_axis_limits(load)
            if released:
                state, accent = "RELEASE EVENT", "#D55E00"
            elif not math.isfinite(margin):
                state, accent = "INVALID MARGIN", "#6b7280"
            elif margin < 0.0:
                state, accent = "OUTSIDE ENVELOPE", "#D55E00"
            elif margin < 0.15:
                state, accent = "NEAR LIMIT", "#E69F00"
            else:
                state, accent = "WITHIN ENVELOPE", "#009E73"

        if outside:
            state += " / VIEW RANGE EXCEEDED"
        load_cells = "".join(
            "<td style='font-family:DejaVu Serif; font-size:12pt;'>"
            f"<i>f</i><sub><i>{axis}</i></sub></td>"
            "<td style='font-family:DejaVu Sans Mono; padding-right:13px;'>"
            f"<b>{value}</b>&thinsp;<span style='font-family:DejaVu Sans;'>N</span></td>"
            for axis, value in zip(("t", "l", "n"), load_values, strict=True)
        )
        self._hud.setText(
            "<div style='color:#202124; font-family:DejaVu Sans; font-size:10pt;'>"
            "<table cellspacing='2' cellpadding='1'>"
            "<tr><td colspan='3'><b>REAL-TIME LOAD</b>&thinsp;"
            "<span style='font-family:DejaVu Serif;'>"
            "<b><i>f</i></b><sub><i>k</i></sub></span></td>"
            f"<td colspan='3' align='right' style='color:{accent};'>"
            f"<b>● {state}</b></td></tr>"
            f"<tr>{load_cells}</tr>"
            "<tr><td colspan='2' style='font-family:DejaVu Serif;'>"
            "<i>m</i>(<b><i>f</i></b><sub><i>k</i></sub>) = "
            f"<b>{margin_text}</b></td>"
            "<td colspan='2' style='font-family:DejaVu Serif;'>"
            f"<i>N</i><sub>rel</sub> = <b>{len(self.releases)}</b></td>"
            "<td colspan='2'></td></tr>"
            "</table></div>"
        )
        if accent != self._last_hud_state:
            self._hud.setStyleSheet(
                "QLabel {"
                " color: #17191c;"
                " background-color: rgba(255, 255, 255, 232);"
                " border: 1px solid rgba(74, 85, 104, 105);"
                f" border-left: 4px solid {accent};"
                " border-radius: 3px;"
                " padding: 7px 10px;"
                "}"
            )
            self._last_hud_state = accent
        self._hud.adjustSize()

    def update(self, load_tln, margin: float, released: bool):
        """Append one load sample and update all live 3D/2D layers."""
        load = np.asarray(load_tln, dtype=float)
        if load.shape != (3,):
            raise ValueError(f"load_tln must have shape (3,), got {load.shape}")
        if not np.all(np.isfinite(load)):
            raise ValueError("load_tln must contain only finite values")

        try:
            margin_value = float(margin)
        except (TypeError, ValueError) as exc:
            raise ValueError("margin must be a scalar") from exc

        released = bool(released)
        self._trail.append(load.copy())
        if len(self._trail) > self._trail_len:
            del self._trail[: len(self._trail) - self._trail_len]
        if released:
            self.releases.append(load.copy())
        self._last_load = load.copy()
        self._last_margin = margin_value
        if self._headless:
            return

        trail = np.asarray(self._trail, dtype=float).reshape(-1, 3)
        if len(trail) > 1:
            alpha = np.linspace(0.22, 0.92, len(trail))
            colors = np.column_stack((np.tile(_TRAIL_RGB, (len(trail), 1)), alpha))
            self._trail_item.setData(pos=trail, color=colors)
        else:
            self._trail_item.setData(pos=trail, color=(*_TRAIL_RGB, 0.92))

        live_color = (*_RELEASE_RGB, 1.0) if released else self._color(margin_value)
        self._live_halo.setData(pos=load[None, :])
        self._live.setData(pos=load[None, :], color=live_color)

        if released:
            release_points = np.asarray(self.releases, dtype=float).reshape(-1, 3)
            self._release_halo.setData(pos=release_points)
            self._rel_item.setData(pos=release_points)
            offset = np.array((0.015 * (self._axis_limits[0][1] - self._axis_limits[0][0]), 0.0, 0.0))
            self._release_label.setData(pos=load + offset, text=f"release #{len(self.releases)}")

        self._update_hud(load, margin_value, released=released)

    def clear(self):
        """Clear trajectory/release history while leaving the current marker."""
        self.releases.clear()
        self._trail.clear()
        if self._headless:
            return
        self._trail_item.setData(pos=_EMPTY_POINTS.copy(), color=(*_TRAIL_RGB, 0.82))
        self._release_halo.setData(pos=_EMPTY_POINTS.copy())
        self._rel_item.setData(pos=_EMPTY_POINTS.copy())
        self._release_label.setData(text="")
        self._update_hud(self._last_load, self._last_margin, released=False)

    def save(self, path: str):
        """Save release points and the parameter vector used by the figure."""
        releases = np.asarray(self.releases, dtype=float).reshape(-1, 3)
        params = np.array([self._plot_params[key] for key in _LIMIT_KEYS], dtype=float)
        np.savez(path, releases=releases, params=params)
