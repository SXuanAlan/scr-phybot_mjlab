# Rizon4 Microspine — Manual Limit-Surface Teleop Tool (Design)

Date: 2026-07-08
Status: implemented; loading layer revised (see Revision).

## Revision 2026-07-08 — loading = task-space force control (OSC)

Supersedes the position-target-drag + leash mechanism described in the
"Configuration"/"Controls"/"Architecture" sections below. Reason: the
position-drag loads the anchor only through the arm's compliance, so with a soft
arm the reachable load saturates (~30 N) and with a stiff arm a closed-chain
preload (~230 N) inflates the caps. Neither maps the full envelope.

The `PadTeleop` loading layer is now **operational-space (task-space force)
control**, mirroring how the real Flexiv Rizon4 RDK runs (we cannot use the
Flexiv controller in MuJoCo — it drives the real robot, and the microspine
physics lives in MuJoCo — so the sim implements an OSC stand-in):

- Control law (per tick, single env, computed on the CPU `mjData` mirror already
  kept for the viewer): `τ = Jᵀ·F_cmd + Nᵀ·τ₀`, applied to the arm DOFs via
  `qfrc_applied` (the arm's position control is disabled; gravity is already
  compensated by the model's `gravcomp=1.0`).
  - `J` = site Jacobian of `microspine_tcp` (`mujoco.mj_jacSite`); `F_cmd` = the
    Xbox-commanded Cartesian force in the engage frame → world.
  - Null-space: `N = I − J⁺J` (kinematic; upgrade to the inertia-weighted
    `N = I − Jᵀ J̄ᵀ` with `M = mj_fullM` if needed); `τ₀ = −K_p(q−q_rest) −
    K_d·q̇` (posture + null-space damping). N projects τ₀ off the task, so posture
    keeps the arm well-conditioned with **zero task-space preload**.
  - Task-space damping `−D_task·ẋ` folded into `F_cmd` for stability.
- Xbox axes now command a **force** (N), not a position delta. `PadTeleop.apply`
  takes `force_tln`. `XboxInput`, `LimitSurfaceView`, `main`, `limit_surface_geom`
  are unchanged in structure (the stick still emits a 3-vector, now interpreted
  as force; the force gain replaces the position gain).
- Align the force gain / stiffness-damping magnitudes to the Flexiv RDK Cartesian
  motion-force defaults so sim tuning transfers to the real robot.

## Goal

An interactive bench tool for **hand-mapping the microspine limit surface** on the
Flexiv Rizon4 arm. An Xbox controller drives the microspine pad in **task space**
(Cartesian); a live **pyqtgraph 3D plot** shows the current anchor load as a point
moving relative to the theory envelope, and **freezes the point where the grip
releases**. Over a session the frozen points accumulate into a hand-measured limit
surface.

Runs on a **local machine with a display** (native MuJoCo window + a pyqtgraph
window + controller plugged into the same machine), single process.

## Non-goals

- No headless/browser (Viser) path — local screen only.
- No wrist-orientation control (right stick unused).
- No changes to the microspine core (`microspine/*`) or the registered task cfg —
  the tool sets its own env config at runtime.
- Not an automated sweep (that already exists in `validate_microspine_constraint`);
  this is human-driven.

## Configuration (fixed inside the tool)

Build the Rizon4 env with `attachment_mode="constraint"` (weld — the production
scheme), then override at runtime so all load directions are reachable and
releases are meaningful:

- **Soft arm**: joint actuator `stiffness ≈ 50`, `damping ≈ 14`, and IK
  `posture_weight = 0` (free the null-space). A stiff joint servo welded to the
  wall builds a ~230 N closed-chain preload that inflates the tension/lateral caps
  out of reach; softening removes it (validated: 98% of sweep directions release).
- **True envelope**: restore `MicrospineLimitSurfaceCfg` defaults
  (`f_n0`, `f_l0` back to ~15, not the bench's de-fanged 200) so the plotted
  envelope is the real one and releases land on it.
- Gravity: vertical wall (arm gravity-compensated as in the bench cfg).

## Controls (Xbox, via `pygame`)

Deltas accumulate into the TCP target pose; a leash limits how far the target may
lead the measured TCP (reused from the existing teleop, retuned for the soft arm).
Axes map to the engage frame (t̂ down-wall shear, l̂ lateral, n̂ outward normal),
converted to world deltas each tick.

| Control | Action | Frame |
| --- | --- | --- |
| Left stick ↕ | shear up / down | ±t̂ |
| Left stick ↔ | lateral | ±l̂ |
| LT / RT | press-in / pull-off | ∓n̂ |
| A | engage (press on) | |
| B | release / re-seat anchor | |
| X | clear accumulated release points | |
| Y | save accumulated points (npz) | |
| Start | return arm to home | |
| Right stick | unused | |

## Architecture — single-threaded, Qt-driven

Qt (pyqtgraph) owns the main event loop. A `QTimer` at ~50 Hz drives one tick:

1. `XboxInput.poll()` → Cartesian delta + button events.
2. `PadTeleop.step(delta, buttons)` → accumulate TCP target (leash-clamped),
   `env.step()`.
3. MuJoCo native window `viewer.sync()`.
4. Read `affordance.anchor_load_tnl`, `hold_margin`, `forced_release_this_step`.
5. `LimitSurfaceView.update(load, margin, released)` → move live point (margin
   color) + trail; on release append a frozen point.

Single thread → no data races. **Integration risk to validate first:** MuJoCo's
`mujoco.viewer.launch_passive` native window coexisting with a Qt window driven
from a `QTimer`. On Linux this is expected to work; step 0 of implementation is a
throwaway spike proving both windows update from one Qt timer before building the
rest.

## Components (independently testable)

- **`XboxInput`** — reads the gamepad (`pygame`), returns a `CartesianCommand`
  (per-tick delta in the engage frame) + edge-detected button events. Depends on:
  pygame. Testable with a fake event source.
- **`PadTeleop`** — owns the Rizon4 env (soft arm, true envelope), the accumulated
  TCP target, the leash, and engage/release/home actions. Exposes `step()` and the
  latest `(anchor_load_tnl, hold_margin, forced_release)`. Depends on: mjlab env,
  microspine affordance. Testable headless (no plot, scripted deltas).
- **`LimitSurfaceView`** — pyqtgraph `GLViewWidget`: static envelope mesh (from
  `r_theory` over a (φ,θ) grid) + axes; live margin-colored point + trail;
  accumulating frozen release points; `clear()` / `save(path)`. Depends on:
  pyqtgraph, numpy. Testable by feeding synthetic loads.
- **`main`** — builds the three, wires the `QTimer`, opens both windows.

## Data flow

Per tick, `PadTeleop` reads three affordance buffers already maintained by the
microspine code — no core changes:

- `anchor_load_tnl` (E,3) → current (f_t, f_l, f_n) point.
- `hold_margin` (E,) → point color (green safe → red at edge).
- `forced_release_this_step` (E,) → freeze-a-point trigger.

`num_envs = 1` for the bench (one arm, one pad). Release points saved as an npz of
`(f_t, f_l, f_n)` rows on `Y`.

## Error handling

- No controller found → clear message, offer keyboard fallback (arrow/WASD map to
  the same deltas) so the tool still runs.
- Pad not engaged when the user drags → loads read ~0; the live point sits at the
  origin; no release points. (Expected; A to engage.)
- Non-finite loads → already clamped in `compute_anchor_loads` (nan_to_num).

## Testing

- `PadTeleop` headless: scripted press → engage; scripted pull in a fixed
  direction → a release; assert the release load lands on the envelope (reuse the
  `_envelope_margin` check from the sweep). No display needed.
- `LimitSurfaceView`: feed a known load path, assert the live point position and
  that a release appends exactly one frozen point; snapshot the mesh.
- `XboxInput`: fake pygame events → assert the mapped deltas/buttons.
- Manual: the QTimer spike + a short hand session.

## File layout

- `scripts/teleop_rizon4_limitsurface.py` — `main` + the three components (split
  into a small `scr_phybot_mjab/tools/limit_surface_teleop/` package if it grows).
- Reuses: `rizon4_microspine_env_cfg`, the microspine affordance, and the
  `r_theory` / envelope-mesh helper (lift the shared helper out of the sweep
  scripts into a small module so both the tool and `validate_...` use one copy).
- New dependency: `pyqtgraph` (+ PyQt5/PySide) and `pygame`, added to
  `pyproject.toml` optional extras (e.g. `[bench]`).
