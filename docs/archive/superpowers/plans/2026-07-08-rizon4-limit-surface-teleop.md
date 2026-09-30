# Rizon4 Microspine Manual Limit-Surface Teleop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local-display tool where an Xbox controller drives the Rizon4 microspine pad in task space while a live pyqtgraph 3D plot shows the current anchor load vs. the theory envelope and freezes a point wherever the grip releases.

**Architecture:** Single process, single thread. Qt (pyqtgraph) owns the event loop; a ~50 Hz `QTimer` runs one tick: poll gamepad → accumulate leash-clamped TCP target → `env.step()` → sync the MuJoCo native window → read the affordance loads → update the 3D view. Pure-logic units (envelope geometry, input mapping) are unit-tested; env/GUI units are verified by a headless scripted test and a manual spike.

**Tech Stack:** Python, mjlab + mujoco_warp (existing), numpy, torch; new: `pyqtgraph` + `PyQt5`, `pygame`.

## Global Constraints

- Reuse the microspine core unchanged: read `affordance.anchor_load_tnl`, `affordance.hold_margin`, `affordance.forced_release_this_step`; do NOT edit `src/scr_phybot_mjab/microspine/*`.
- Env config for the tool (verbatim, proven in the sweep): constraint-only weld; joint actuator `stiffness=50.0`, `damping=14.0`; IK `posture_weight=0.0`; restore `MicrospineLimitSurfaceCfg` defaults (do NOT keep the bench's `f_n0=200`/`f_l0=200`).
- Engage frame convention: `t̂` = down-wall shear, `l̂` = lateral, `n̂` = outward normal (pull-off +). Load vector order is `(f_t, f_l, f_n)`.
- Rizon4 constants: EE body `"microspine_pad"`, TCP site `"microspine_tcp"`, action is a 7D absolute TCP pose `[x,y,z,qw,qx,qy,qz]`.
- `num_envs = 1` (one bench arm/pad).
- Run everything from the `env_mjlab` conda env with `MUJOCO_GL` unset (native display), controller on the local machine.
- Commit after every task. Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

- `src/scr_phybot_mjab/microspine/limit_surface_geom.py` — pure-numpy envelope math (release radius, mesh, margin), shared by the tool and future sweep refactors.
- `src/scr_phybot_mjab/tools/limit_surface_teleop/__init__.py`
- `src/scr_phybot_mjab/tools/limit_surface_teleop/xbox_input.py` — `XboxInput`: gamepad → engage-frame delta + button edges.
- `src/scr_phybot_mjab/tools/limit_surface_teleop/pad_teleop.py` — `PadTeleop`: Rizon4 env (soft arm, true envelope), press/engage/home, leash-clamped target, load readout.
- `src/scr_phybot_mjab/tools/limit_surface_teleop/view.py` — `LimitSurfaceView`: pyqtgraph GLViewWidget (envelope + live point + trail + frozen releases).
- `scripts/teleop_rizon4_limitsurface.py` — `main`: build the three, wire the QTimer, open both windows; keyboard fallback.
- `tests/microspine/test_limit_surface_geom.py`, `tests/tools/test_xbox_input.py`, `tests/tools/test_pad_teleop.py` (GPU-gated), `tests/tools/test_limit_surface_view.py`.
- `pyproject.toml` — add `[project.optional-dependencies] bench = ["pyqtgraph", "PyQt5", "pygame"]`.

---

### Task 0: Spike — MuJoCo native window + Qt window coexist (throwaway)

**Files:** Create (throwaway, not committed): `/tmp/spike_qt_mujoco.py`

**Interfaces:** none (go/no-go gate).

- [ ] **Step 1: Write the spike**

```python
import mujoco, mujoco.viewer, numpy as np
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph.opengl as gl

m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><joint/><geom size='.1'/></body></worldbody></mujoco>")
d = mujoco.MjData(m)
app = QtWidgets.QApplication([])
w = gl.GLViewWidget(); w.show()
sp = gl.GLScatterPlotItem(pos=np.zeros((1, 3))); w.addItem(sp)
viewer = mujoco.viewer.launch_passive(m, d)
t = {"n": 0}
def tick():
    mujoco.mj_step(m, d); viewer.sync()
    t["n"] += 1
    sp.setData(pos=np.array([[np.sin(t["n"] * 0.05), 0, 0]]))
    if t["n"] > 1000 or not viewer.is_running():
        timer.stop(); viewer.close(); app.quit()
timer = QtCore.QTimer(); timer.timeout.connect(tick); timer.start(20)
app.exec_()
print("SPIKE OK")
```

- [ ] **Step 2: Run it and watch**

Run: `conda run -n env_mjlab python /tmp/spike_qt_mujoco.py`
Expected: BOTH a MuJoCo window and a pyqtgraph window open; the scatter point oscillates; prints `SPIKE OK`. If the two windows deadlock or one never opens, STOP and switch the plan's loop to Approach-B (run MuJoCo `launch_passive` and pump Qt via `app.processEvents()` inside a plain `while viewer.is_running()` loop instead of a QTimer). Record which approach worked in the commit message of Task 5.

---

### Task 1: `limit_surface_geom` — envelope math (pure numpy)

**Files:**
- Create: `src/scr_phybot_mjab/microspine/limit_surface_geom.py`
- Test: `tests/microspine/test_limit_surface_geom.py`

**Interfaces:**
- Produces:
  - `LIMIT_KEYS: tuple[str, ...] = ("f_t_max", "f_rev_max", "mu_adhesion", "f_n0", "mu_lateral", "f_l0")`
  - `release_radius(a, b, c, p: dict) -> np.ndarray` — smallest pull magnitude along unit dir `(a,b,c)` in `(t,l,n)` that leaves the envelope; `inf` where compression-dominated. Vectorized over numpy arrays.
  - `envelope_mesh(p: dict, n_phi: int = 90, n_theta: int = 140, r_clip: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — `(X, Y, Z)` = `(f_t, f_l, f_n)` grids; `NaN` where the surface is open/clipped.
  - `envelope_margin(load_tln: np.ndarray, p: dict) -> np.ndarray` — signed min-margin (≥0 inside, <0 outside); `load_tln[..., 0:3]` = `(f_t, f_l, f_n)`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from scr_phybot_mjab.microspine.limit_surface_geom import (
    release_radius, envelope_mesh, envelope_margin, LIMIT_KEYS,
)

P = {"f_t_max": 400.0, "f_rev_max": 20.0, "mu_adhesion": 0.4,
     "f_n0": 15.0, "mu_lateral": 0.6, "f_l0": 15.0}

def test_release_radius_pure_pulloff_hits_f_n0():
    # pure +n (a=0,b=0,c=1): releases via tension at f_n0
    r = release_radius(np.array([0.0]), np.array([0.0]), np.array([1.0]), P)
    assert np.isclose(r[0], P["f_n0"], atol=1e-6)

def test_release_radius_pure_shear_hits_ultimate():
    r = release_radius(np.array([1.0]), np.array([0.0]), np.array([0.0]), P)
    assert np.isclose(r[0], P["f_t_max"], atol=1e-6)

def test_release_radius_pure_compression_is_inf():
    r = release_radius(np.array([0.0]), np.array([0.0]), np.array([-1.0]), P)
    assert not np.isfinite(r[0])

def test_margin_positive_inside_negative_outside():
    inside = envelope_margin(np.array([[0.0, 0.0, 5.0]]), P)   # f_n=5 < 15
    outside = envelope_margin(np.array([[0.0, 0.0, 30.0]]), P)  # f_n=30 > 15
    assert inside[0] > 0 and outside[0] < 0

def test_mesh_shape_and_points_on_surface():
    X, Y, Z = envelope_mesh(P, n_phi=30, n_theta=40)
    assert X.shape == (30, 40)
    finite = np.isfinite(Z)
    loads = np.stack([X[finite], Y[finite], Z[finite]], axis=-1)
    assert np.all(np.abs(envelope_margin(loads, P)) < 1e-3)  # mesh lies on the surface
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n env_mjlab pytest tests/microspine/test_limit_surface_geom.py -v`
Expected: FAIL with `ModuleNotFoundError: ...limit_surface_geom`.

- [ ] **Step 3: Write the implementation** (lifted verbatim from the validated sweep scripts)

```python
"""Pure-numpy microspine limit-surface geometry: release radius, mesh, margin."""
from __future__ import annotations
import math
import numpy as np

LIMIT_KEYS = ("f_t_max", "f_rev_max", "mu_adhesion", "f_n0", "mu_lateral", "f_l0")


def release_radius(a, c_lat, c_norm, p):
    a = np.asarray(a, float); b = np.asarray(c_lat, float); c = np.asarray(c_norm, float)
    inf = np.inf
    ap = np.maximum(a, 0.0)
    r_rev = np.where(a < 0, p["f_rev_max"] / np.maximum(-a, 1e-9), inf)
    r_ult = np.where(a > 0, p["f_t_max"] / np.maximum(a, 1e-9), inf)
    dt = c - p["mu_adhesion"] * ap
    r_ten = np.where(dt > 1e-9, p["f_n0"] / np.maximum(dt, 1e-9), inf)
    dl = np.abs(b) - p["mu_lateral"] * ap
    r_lat = np.where(dl > 1e-9, p["f_l0"] / np.maximum(dl, 1e-9), inf)
    return np.minimum.reduce([r_rev, r_ult, r_ten, r_lat])


def envelope_mesh(p, n_phi=90, n_theta=140, r_clip=None):
    if r_clip is None:
        r_clip = 1.6 * p["f_t_max"]
    phi = np.linspace(0.03, math.pi - 0.03, n_phi)
    th = np.linspace(0.0, 2 * math.pi, n_theta)
    pp, tt = np.meshgrid(phi, th, indexing="ij")
    a = np.cos(pp); b = np.sin(pp) * np.cos(tt); c = np.sin(pp) * np.sin(tt)
    r = release_radius(a, b, c, p)
    r = np.where(np.isfinite(r) & (r < r_clip), r, np.nan)
    return r * a, r * b, r * c


def envelope_margin(load_tln, p):
    L = np.asarray(load_tln, float)
    f_t, f_l, f_n = L[..., 0], L[..., 1], L[..., 2]
    ft_pos = np.maximum(f_t, 0.0)
    cap_n = p["mu_adhesion"] * ft_pos + p["f_n0"]
    cap_l = p["mu_lateral"] * ft_pos + p["f_l0"]
    m1 = (f_t + p["f_rev_max"]) / p["f_rev_max"]
    m2 = (cap_n - f_n) / cap_n
    m3 = (cap_l - np.abs(f_l)) / cap_l
    m4 = (p["f_t_max"] - f_t) / p["f_t_max"]
    return np.minimum.reduce([m1, m2, m3, m4])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n env_mjlab pytest tests/microspine/test_limit_surface_geom.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/scr_phybot_mjab/microspine/limit_surface_geom.py tests/microspine/test_limit_surface_geom.py
git commit -m "feat(microspine): shared limit-surface geometry (release radius, mesh, margin)"
```

---

### Task 2: `XboxInput` — gamepad → engage-frame delta + buttons

**Files:**
- Create: `src/scr_phybot_mjab/tools/limit_surface_teleop/__init__.py` (empty)
- Create: `src/scr_phybot_mjab/tools/limit_surface_teleop/xbox_input.py`
- Test: `tests/tools/test_xbox_input.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass InputState: delta_tln: tuple[float, float, float]; buttons: frozenset[str]` where buttons ⊆ `{"engage","release","clear","save","home"}` (edge-pressed since last poll).
  - `class XboxInput:` `__init__(self, source, lin_gain: float = 0.02, deadzone: float = 0.15)`; `poll(self) -> InputState`. `source` is any object exposing `axes(self) -> dict[str,float]` (`lx, ly, lt, rt`) and `pressed(self) -> set[str]` (raw button names `A,B,X,Y,START`), so tests inject a fake and production wraps `pygame`.
  - Mapping (per Global Constraints frame): `dt = -ly` (stick up = +t̂ shear up), `dl = lx`, `dn = (lt - rt)` (LT press-in = −n̂... i.e. into wall; RT pull-off = +n̂). All scaled by `lin_gain`, deadzoned. Buttons: A→engage, B→release, X→clear, Y→save, START→home.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n env_mjlab pytest tests/tools/test_xbox_input.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n env_mjlab pytest tests/tools/test_xbox_input.py -v`
Expected: 4 passed. (`PygameSource` is not exercised by unit tests — verified manually in Task 5.)

- [ ] **Step 5: Commit**

```bash
git add src/scr_phybot_mjab/tools/limit_surface_teleop/__init__.py src/scr_phybot_mjab/tools/limit_surface_teleop/xbox_input.py tests/tools/test_xbox_input.py
git commit -m "feat(tools): XboxInput mapping (stick=shear/lateral, triggers=normal, button edges)"
```

---

### Task 3: `PadTeleop` — Rizon4 env (soft arm, true envelope), press/engage, load readout

**Files:**
- Create: `src/scr_phybot_mjab/tools/limit_surface_teleop/pad_teleop.py`
- Test: `tests/tools/test_pad_teleop.py` (GPU-gated: `@pytest.mark.skipif(not torch.cuda.is_available())`)

**Interfaces:**
- Consumes: `rizon4_microspine_env_cfg`, `MicrospineLimitSurfaceCfg` defaults, `limit_surface_geom.LIMIT_KEYS`.
- Produces:
  - `@dataclass Reading: load_tln: np.ndarray (3,); margin: float; released: bool; engaged: bool`
  - `class PadTeleop:` `__init__(self, device="cuda:0")`; attrs `.env, .affordance, .executor, .limit_params: dict` (the true-envelope values, keyed by `LIMIT_KEYS`).
  - `press_engage(self) -> None` — ramp the TCP target into the wall until engaged + null the preload (lifted from `rizon4_sweep_collect.run_batch` press/settle block).
  - `home(self) -> None` — reset env, re-press.
  - `apply(self, delta_tln: tuple[float,float,float]) -> Reading` — rotate the delta into world via the current engage frame, add to the accumulated TCP target, clamp the target's lead over the measured TCP to `leash` (default 0.02 m), `env.step(target)`, return a fresh `Reading`.

- [ ] **Step 1: Write the failing (GPU-gated) test**

```python
import numpy as np, pytest, torch
from scr_phybot_mjab.microspine.limit_surface_geom import envelope_margin

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

def test_press_engages_and_pulloff_releases_on_envelope():
    from scr_phybot_mjab.tools.limit_surface_teleop.pad_teleop import PadTeleop
    pt = PadTeleop()
    pt.press_engage()
    assert pt.apply((0.0, 0.0, 0.0)).engaged
    r = None
    for _ in range(400):  # ramp a pull-off (+n) drag until it releases
        r = pt.apply((0.0, 0.0, -0.0004))
        if r.released:
            break
    assert r is not None and r.released
    assert abs(envelope_margin(r.load_tln[None], pt.limit_params)[0]) < 0.2
    pt.env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n env_mjlab pytest tests/tools/test_pad_teleop.py -v`
Expected: FAIL with `ModuleNotFoundError` (or skip if no GPU — then verify on a GPU box).

- [ ] **Step 3: Write the implementation** (env build/press lifted verbatim from the validated `rizon4_sweep_collect.py`)

```python
"""Rizon4 microspine bench teleop: soft arm, true envelope, leash-clamped target."""
from __future__ import annotations
from dataclasses import dataclass
import mujoco
import numpy as np
import torch
import scr_phybot_mjab.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from scr_phybot_mjab.microspine.config import MicrospineLimitSurfaceCfg
from scr_phybot_mjab.microspine.limit_surface_geom import LIMIT_KEYS
from scr_phybot_mjab.tasks.microspine_rizon4.config.rizon4.env_cfgs import (
    rizon4_microspine_env_cfg,
)


@dataclass
class Reading:
    load_tln: np.ndarray
    margin: float
    released: bool
    engaged: bool


class PadTeleop:
    def __init__(self, device: str = "cuda:0", leash: float = 0.02):
        self.device = device
        self.leash = leash
        cfg = rizon4_microspine_env_cfg(play=False)
        cfg.actions["cartesian_pose"].posture_weight = 0.0
        for a in cfg.scene.entities["robot"].articulation.actuators:
            a.stiffness = 50.0; a.damping = 14.0
        d = MicrospineLimitSurfaceCfg()
        m = cfg.commands["microspine_affordance"].microspine_cfg
        for owner in (m, cfg.events["microspine_grip"].params["microspine_cfg"]):
            owner.limit_surface.f_n0_range = d.f_n0_range
            owner.limit_surface.f_l0_range = d.f_l0_range
        cfg.scene.num_envs = 1
        cfg.episode_length_s = 1.0e9
        self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
        self.affordance = self.env.command_manager._terms["microspine_affordance"]
        self.executor = self.env.event_manager.get_term_cfg("microspine_grip").func
        self.robot = self.env.scene["robot"]
        self.wall = self.env.scene["wall"]
        self._pid = self.robot.find_bodies(["microspine_pad"])[0]
        self._half = float(self.env.sim.mj_model.geom("wall/wall_geom").size[0])
        self.mj_model = self.env.sim.mj_model
        self.mj_data = mujoco.MjData(self.mj_model)  # CPU mirror for the native viewer
        mid = lambda r: 0.5 * (r[0] + r[1])
        ls = m.limit_surface
        self.limit_params = {
            "f_t_max": mid(ls.f_t_max_range), "f_rev_max": mid(ls.f_rev_max_range),
            "mu_adhesion": mid(ls.mu_adhesion_range), "f_n0": mid(ls.f_n0_range),
            "mu_lateral": mid(ls.mu_lateral_range), "f_l0": mid(ls.f_l0_range),
        }
        self._target = None
        self.home()

    def _pad_pose(self):
        pid = self._pid[0]
        return (self.robot.data.body_link_pos_w[:, pid].clone(),
                self.robot.data.body_link_quat_w[:, pid].clone())

    def sync_display(self):
        self.mj_data.qpos[:] = self.env.sim.data.qpos[0].detach().cpu().numpy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _step_target(self, pos, quat):
        act = torch.zeros(1, 7, device=self.device)
        act[:, 0:3] = pos; act[:, 3:7] = quat
        self.env.step(act)

    def home(self):
        self.env.reset(); self.executor.reset()
        p0, q0 = self._pad_pose()
        wallx = self.wall.data.root_link_pos_w[:, 0]
        tx = wallx - self._half + 0.02
        act = torch.zeros(1, 7, device=self.device)
        act[:, 1:3] = p0[:, 1:3]; act[:, 3:7] = q0
        for i in range(130):
            frac = min(1.0, i / 90.0)
            act[:, 0] = p0[:, 0] + frac * (tx - p0[:, 0])
            self.env.step(act)
        pos, quat = self._pad_pose()
        self._target = pos.clone(); self._quat = quat.clone()

    def press_engage(self):
        # null the press preload by re-seating the anchor at the settled pad
        for _ in range(6):
            for _ in range(6):
                self._step_target(self._target, self._quat)
            self.executor.anchor_pos_w = self.executor.pad_center_world().clone()
        for _ in range(10):
            self._step_target(self._target, self._quat)

    def apply(self, delta_tln) -> Reading:
        t_hat, l_hat, n_hat = self.affordance._engage_frame()
        t_hat, l_hat, n_hat = t_hat[:, 0], l_hat[:, 0], n_hat[:, 0]
        dt, dl, dn = delta_tln
        dworld = dt * t_hat + dl * l_hat + dn * n_hat  # (1,3)
        self._target = self._target + dworld
        pad, _ = self._pad_pose()
        lead = self._target - pad
        n = torch.linalg.norm(lead, dim=-1, keepdim=True).clamp_min(1e-9)
        lead = lead * torch.clamp(self.leash / n, max=1.0)
        self._target = pad + lead
        self._step_target(self._target, self._quat)
        load = self.affordance.anchor_load_tnl[0, 0].detach().cpu().numpy()
        return Reading(
            load_tln=load,
            margin=float(self.affordance.hold_margin[0].item()),
            released=bool(self.affordance.forced_release_this_step[0, 0].item()),
            engaged=bool(self.affordance.engaged[0, 0].item()),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n env_mjlab pytest tests/tools/test_pad_teleop.py -v`
Expected: 1 passed (on a GPU box). If the release margin assertion is flaky, widen the drag budget (loop count) — do NOT relax the envelope tolerance below 0.2.

- [ ] **Step 5: Commit**

```bash
git add src/scr_phybot_mjab/tools/limit_surface_teleop/pad_teleop.py tests/tools/test_pad_teleop.py
git commit -m "feat(tools): PadTeleop Rizon4 bench (soft arm, true envelope, leash target)"
```

---

### Task 4: `LimitSurfaceView` — pyqtgraph 3D (envelope + live point + trail + frozen releases)

**Files:**
- Create: `src/scr_phybot_mjab/tools/limit_surface_teleop/view.py`
- Test: `tests/tools/test_limit_surface_view.py`

**Interfaces:**
- Consumes: `limit_surface_geom.envelope_mesh`, `envelope_margin`.
- Produces:
  - `class LimitSurfaceView:` `__init__(self, limit_params: dict)`; attrs `.widget` (GLViewWidget), `.releases: list[np.ndarray]`.
  - `update(self, load_tln: np.ndarray, margin: float, released: bool) -> None` — set the live point at `load_tln`, colour it `green(margin>0.3)/yellow/red(margin<0)`, append to a fixed-length trail, and if `released` append `load_tln.copy()` to `self.releases` (redrawn as a frozen scatter).
  - `clear(self) -> None` — empty `self.releases` and the frozen scatter.
  - `save(self, path: str) -> None` — `np.savez(path, releases=np.array(self.releases))`.
- Guard: the GL items are only created if `QApplication.instance()` exists; the release/trail bookkeeping (lists) works without a display so it is unit-testable.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np, os
from scr_phybot_mjab.tools.limit_surface_teleop.view import LimitSurfaceView

P = {"f_t_max": 400.0, "f_rev_max": 20.0, "mu_adhesion": 0.4,
     "f_n0": 15.0, "mu_lateral": 0.6, "f_l0": 15.0}

def test_release_appends_one_frozen_point_and_saves(tmp_path):
    v = LimitSurfaceView(P, headless=True)   # headless: no GL items
    v.update(np.array([10.0, 0, 5.0]), margin=0.5, released=False)
    assert len(v.releases) == 0
    v.update(np.array([0.0, 0, 15.0]), margin=0.0, released=True)
    assert len(v.releases) == 1
    p = tmp_path / "pts.npz"
    v.save(str(p))
    assert np.load(p)["releases"].shape == (1, 3)
    v.clear()
    assert len(v.releases) == 0

def test_margin_to_color():
    v = LimitSurfaceView(P, headless=True)
    assert v._color(0.5)[1] > v._color(0.5)[0]     # green-ish
    assert v._color(-0.5)[0] > v._color(-0.5)[1]   # red-ish
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n env_mjlab pytest tests/tools/test_limit_surface_view.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n env_mjlab pytest tests/tools/test_limit_surface_view.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/scr_phybot_mjab/tools/limit_surface_teleop/view.py tests/tools/test_limit_surface_view.py
git commit -m "feat(tools): LimitSurfaceView pyqtgraph 3D (envelope + live/trail/frozen points)"
```

---

### Task 5: `main` — wire the QTimer, two windows, keyboard fallback

**Files:**
- Create: `scripts/teleop_rizon4_limitsurface.py`

**Interfaces:**
- Consumes: `PadTeleop`, `XboxInput`/`PygameSource`, `LimitSurfaceView`.
- Produces: a runnable CLI (`python scripts/teleop_rizon4_limitsurface.py`). No importable API.

- [ ] **Step 1: Write the script**

```python
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
    xin = XboxInput(src)
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
```

- [ ] **Step 2: Manual verification (local display box with a gamepad)**

Run: `conda run -n env_mjlab python scripts/teleop_rizon4_limitsurface.py`
Expected: MuJoCo window + limit-surface window both open; the pad is pressed on the wall (engaged); moving the left stick / triggers moves the live point inside the envelope, colour shifting green→red near the edge; dragging past the edge freezes an orange point; `c` clears, `v` saves `limit_surface_manual.npz`. If Task 0 recorded that the QTimer approach deadlocks, replace `app.exec_()` + QTimer with a `while viewer.is_running(): tick(); app.processEvents()` loop.

- [ ] **Step 3: Commit**

```bash
git add scripts/teleop_rizon4_limitsurface.py
git commit -m "feat(tools): Rizon4 microspine manual limit-surface teleop entrypoint"
```

---

### Task 6: dependencies + docs

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:** none.

- [ ] **Step 1: Add the optional extra to `pyproject.toml`**

Under `[project.optional-dependencies]` (create the table if absent):

```toml
bench = ["pyqtgraph", "PyQt5", "pygame"]
```

- [ ] **Step 2: Install and smoke-import**

Run: `conda run -n env_mjlab pip install -e ".[bench]"`
Then: `conda run -n env_mjlab python -c "import pyqtgraph.opengl, pygame; print('bench deps ok')"`
Expected: `bench deps ok`.

- [ ] **Step 3: Document in `README.md`** (append under a "Bench tools" heading)

```markdown
### Manual limit-surface teleop (Rizon4)

Xbox controller drives the pad in task space; a live 3D plot shows the anchor
load vs. the theory envelope and freezes a point where the grip releases.

```bash
pip install -e ".[bench]"
python scripts/teleop_rizon4_limitsurface.py
```

Left stick = shear/lateral, LT/RT = press-in/pull-off, A=engage, B=release,
X=clear points, Y=save (`limit_surface_manual.npz`), Start=home.
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml README.md
git commit -m "docs(tools): bench extra + limit-surface teleop usage"
```

---

## Self-Review

**Spec coverage:** goal → Tasks 3/4/5; soft-arm + true-envelope config → Task 3 `__init__` (matches Global Constraints verbatim); controls → Task 2 + Task 5 keyboard fallback; envelope+live point+frozen releases+margin colour → Task 4; save/clear → Tasks 4/5; single-thread QTimer loop + integration risk → Task 0 spike + Task 5; module boundaries → one file each; deps → Task 6. No core microspine edits (Global Constraints). All covered.

**Placeholder scan:** every code step has complete code; no TBD/TODO; the one hard-to-unit-test path (`PygameSource`, GL rendering, MuJoCo window) is covered by the Task 0 spike + Task 3 GPU test + Task 5 manual step, called out explicitly rather than faked.

**Type consistency:** `delta_tln: tuple[float,float,float]` (Tasks 2,3,5), `InputState.buttons: frozenset[str]` with the `{"engage","release","clear","save","home"}` vocabulary (Tasks 2,5), `Reading(load_tln, margin, released, engaged)` (Tasks 3,5), `limit_params: dict` keyed by `LIMIT_KEYS` (Tasks 1,3,4), `LimitSurfaceView.update(load_tln, margin, released)` / `.clear()` / `.save(path)` / `.releases` (Tasks 4,5). Consistent.
