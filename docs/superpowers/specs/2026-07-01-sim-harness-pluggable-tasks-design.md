# Sim Harness + Pluggable Tasks with Per-Task Controller Config

**Date:** 2026-07-01
**Status:** Design (approved, pre-implementation)
**Scope:** `deploy/bdml_phybot_deploy_c2/`
**Related:** [2026-06-28-deploy-refactor-design.md](2026-06-28-deploy-refactor-design.md) (§ TaskManager/TaskRegistry, §10.1 sim-vs-real PD)

## 1. Motivation & Goal

Today, adding or swapping a policy task (e.g. the user's own `walk_amp`) requires editing
the sim apps by hand:

- `WalkTask(model_path)` takes a ctor argument, so it cannot use the no-arg
  `REGISTER_TASK` self-registration; `sim_walk`/`sim_teleop` build the TaskManager
  manually (`manager.add(std::make_unique<WalkTask>(path))`).
- Controller→task routing is hardcoded in `backends/mujoco/teleop_loop.h::ModelToTask()`
  (`model==2 → WALK, else ZERO`), and axis clamps are hardcoded in `ApplyCmdAndTick`.
- `sim_walk` and `sim_teleop` duplicate the backend/manager/loop setup.

**Goal:** a unified, reusable sim harness where adding a task costs only:

1. a **task class** (the policy/obs logic — your algorithm, unavoidable),
2. **one `REGISTER_TASK` line**,
3. a task **`<task>.yaml`** (policy path + params),
4. a task **`controller.yaml`** (per-task controller config: select code + axis ranges),
5. **one CMake source line** (the compiler must see the new `.cpp`),
6. **one publisher `buttons.yaml` line** (bind an Xbox button → select code).

No edits to the harness, the sim apps, or the routing code.

## 2. Target Developer Experience (walk_amp)

```
tasks/walk_amp/
  walk_amp_task.h / .cpp        # WalkAmpTask + REGISTER_TASK(TaskId::WALK_AMP,"walk_amp",WalkAmpTask)
  config/
    walk_amp.yaml               # policy_path + obs/action params
    controller.yaml             # select_code: 3, per-axis ranges
```

```cpp
// walk_amp_task.cpp (tail)
REGISTER_TASK(phybot::TaskId::WALK_AMP, "walk_amp", WalkAmpTask)
```

```yaml
# tasks/walk_amp/config/controller.yaml
select_code: 3
vx: { range: [-0.4, 1.0] }     # normalized stick [-1,1] -> m/s for THIS task
vy: { range: [-0.1, 0.1] }
wz: { range: [-1.0, 1.0] }     # rad/s
```

```cmake
# CMakeLists.txt: add to the phybot_tasks sources (one line)
tasks/walk_amp/walk_amp_task.cpp
```

```yaml
# publisher/config/buttons.yaml (one line)
X: 3        # Xbox X button selects code 3 (walk_amp)
```

Rebuild → `sim_walk --gui`, `sim_teleop --gui`, headless, and the real-robot backend
all pick up `walk_amp` automatically.

## 3. Architecture Overview

```
 Publisher process (generic controller bridge)         Sim/robot process
 ┌───────────────────────────────────────┐             ┌──────────────────────────────┐
 │ JsController → ControllerState          │             │        SimHarness            │
 │ MapControllerToDrive (deadzone,         │   LCM       │  ┌────────────────────────┐  │
 │   normalize to [-1,1]) + buttons.yaml   │ ───────────▶│  │ TaskRouter             │  │
 │   (button → select_code)                │  DriveCmd   │  │  code→TaskId (routing) │  │
 │ ip_encode → DriveCmd{code,vx,vy,wz}     │  {code,     │  │  active task ranges    │  │
 └───────────────────────────────────────┘   vx,vy,wz}  │  │  (shape axes→setpoints)│  │
                                                          │  └────────────────────────┘  │
                                                          │  TaskManager (registry-built)│
                                                          │  MujocoBackend               │
                                                          │  MujocoViewer (optional,--gui)│
                                                          └──────────────────────────────┘
```

Two decoupled halves, connected only by the `select_code` integer + normalized axes:

- **Publisher** knows nothing about tasks. It maps Xbox → normalized `DriveCmd`
  and stamps a `select_code` from `buttons.yaml`.
- **Harness** owns all task knowledge: it builds the TaskManager from the registry,
  builds the `code→TaskId` routing and per-task axis ranges from each task's
  `controller.yaml`, and each tick routes + shapes + ticks + steps.

## 4. Component: Config-carried policy path (self-registration enabler)

**Problem:** `WalkTask(model_path)` can't self-register (no-arg factory).

**Fix:** move the policy path into config.
- `WalkConfig` gains `std::string policy_path;` loaded from `tasks/walk/config/walk.yaml`
  (`policy_path: <abs-or-relative>`; for tests/CMake, resolvable against the task config dir
  or an absolute path baked by CMake as today for the golden `cpg_policy.pt`).
- `WalkTask()` default ctor loads its `WalkConfig`, reads `cfg_.policy_path`, constructs the
  `PolicyRunner`. The optional `WalkTask(model_path)` overload stays for tests (explicit path
  wins over config).
- Restores `REGISTER_TASK(TaskId::WALK, "walk", WalkTask)` as a one-liner.

**Same pattern** for `WalkAmpTask` and any future policy task.

## 5. Component: Per-task controller config

Each task folder holds `config/controller.yaml`:

```yaml
select_code: 2                 # required: publisher `model`/code that activates this task
vx: { range: [min, max] }      # optional per-axis: normalized [-1,1] -> physical setpoint
vy: { range: [min, max] }
wz: { range: [min, max] }
```

- `ControllerConfig` struct: `int select_code; AxisRange vx, vy, wz;`
  where `AxisRange{double lo, hi;}` maps normalized `n∈[-1,1]` →
  `n>=0 ? n*hi : n*(-lo)` (i.e. `[-1,1]` spans `[lo,hi]` with 0→0), then clamped to `[lo,hi]`.
- Loaded by the harness for every **registered** task (see §6 discovery). ZERO's
  `controller.yaml` needs only `select_code: 1` (it ignores axes).
- Absent axis block → that axis is held at 0 for that task.

**Rationale for normalized-axis split:** the publisher can't know the active task, so
physical scaling (m/s ranges) must live task-side. The publisher sends normalized intent;
each task applies its own ranges. Different policies get different feel with zero shared state.

## 6. Component: Data-driven routing (TaskRouter)

Replaces `teleop_loop.h::ModelToTask()`.

```cpp
class TaskRouter {
 public:
  // Built from every registered task's controller.yaml.
  void addTask(TaskId id, const ControllerConfig& cc);   // registers select_code + ranges
  TaskId route(int code) const;                           // code→TaskId; unknown→kDefault (ZERO)
  void shape(TaskId active, const DriveCmd& cmd, DataPackage& io) const;  // normalized→setpoints
 private:
  std::array<TaskId, kMaxSelectCode> code_to_task_;       // dense small map
  std::array<ControllerConfig, (size_t)TaskId::COUNT> cfg_;
};
```

- `route`: unknown/zero code → `TaskId::ZERO` (safe power-on).
- `shape`: look up the active task's `ControllerConfig`, map `cmd.vx/vy/wz` (normalized)
  through its `AxisRange`s into `io.js_vx_desire / js_vy_desire / js_OmegaZ_desire`.
- Collision detection: two tasks with the same `select_code` → hard error at startup
  (fail fast, don't silently shadow).

**Config discovery:** one CMake compile-def `PHYBOT_TASKS_DIR` = path to `tasks/`. For each
registered task *name* (from `REGISTER_TASK`'s NAME), the router loads
`${PHYBOT_TASKS_DIR}/<name>/config/controller.yaml`. Adding a task needs no routing/CMake
change beyond compiling its `.cpp`.

## 7. Component: SimHarness library (glfw-free core) + separate GUI runner

**Invariant to preserve:** the headless test binaries must not link glfw/GL/X11. So
`SimHarness` is **glfw-free** (backend + router + manager + `tick` + headless loop). The
windowed loop lives in a *separate* unit that only the apps link.

`backends/mujoco/sim_harness.{h,cpp}` — reusable, headless, no glfw:

```cpp
class SimHarness {
 public:
  struct Options { std::string xml_path; };
  explicit SimHarness(const Options& opt);   // load backend; registry.loadInto(manager);
                                             // build TaskRouter from configs; start ZERO.

  // One control step: route code→task-switch, shape axes→setpoints, read→tick→write→step.
  void tick(const DriveCmd& cmd);

  // Command source: returns the latest DriveCmd each iteration (fixed, or poll-LCM).
  using CommandSource = std::function<DriveCmd()>;
  // Headless loop. seconds<=0 → run until `stop` returns true (default: never).
  void runHeadless(CommandSource source, double seconds,
                   std::function<bool()> stop = {});

  MujocoBackend& backend();          // exposes model()/data() for the GUI runner
  TaskManager&   manager();
};
```

`backends/mujoco/sim_harness_gui.{h,cpp}` — links `mujoco_gui`, compiled into the apps only:

```cpp
// Physics thread runs harness.tick(source()); main thread runs MujocoViewer::renderLoop().
// Reuses the exact threading in mujoco_viewer.h (Load on physics thread, mutex around tick).
void RunGui(SimHarness& h, SimHarness::CommandSource source, const char* title);
```

- `sim_walk` = build `SimHarness{xml}`; if `--gui` → `RunGui(h, fixedSource, "sim_walk")`
  else `h.runHeadless(fixedSource, seconds)`.
- `sim_teleop` = build harness; source = `[&]{ lcm.handleTimeout(0); return holder.latest; }`;
  `--gui` → `RunGui(...)` else `runHeadless(..., 0)`.
- `ApplyCmdAndTick`/`teleop_loop.h` is absorbed into `SimHarness::tick` (router replaces the
  hardcoded clamps/ModelToTask). `test_teleop_smoke` retargets onto `SimHarness` (headless
  core only → no glfw).

## 8. Component: Publisher (generic bridge + buttons.yaml)

- `MapControllerToDrive` changes: emit **normalized** `vx/vy/wz ∈ [-1,1]` (deadzone applied,
  no physical scaling) instead of today's `*0.6 / *0.1 / *1.0`.
- `code` (was `model`) comes from `publisher/config/buttons.yaml`: `{ B: 2, Y: 1, X: 3, ... }`
  — rising-edge of a button stamps its code (latching: last pressed wins, like today's
  B→2 / Y→1). Default at startup: `1` (ZERO) — safe.
- Wire format (`input_protocol` / `DriveCmd`) is UNCHANGED in layout: `{code, vx, vy, wz}`
  (rename `model`→`code` only). Bit compatibility with the real-robot link is preserved;
  only the **semantics** of vx/vy/wz become normalized (documented).

## 9. Build & Linkage (WHOLE_ARCHIVE)

Self-registration via static-init is stripped when tasks live in a `STATIC` lib (the
`task_registry.h` header already warns about this; today it "works" only because
`zero_task.cpp` is compiled directly into each target).

- New `phybot_tasks` STATIC lib: `tasks/*/**_task.cpp` + their config `.cpp` deps.
- Linked with **whole-archive** so the *final executables* retain the registrars:
  `target_link_libraries(sim_harness PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,phybot_tasks>")`
  (CMake ≥ 3.24, propagates to consuming exes; fallback
  `-Wl,--whole-archive … -Wl,--no-whole-archive` for older CMake — the tree requires ≥3.12,
  so provide the fallback). Every exe relying on registry-loaded tasks links `sim_harness`.
- **Verification:** a ctest that builds a target linking `phybot_tasks` whole-archive and
  asserts `TaskRegistry::instance().has(TaskId::WALK)` and `has(TaskId::ZERO)` are true
  (i.e. registrars survived). This is the guard against silent stripping.

## 10. Command flow (end to end)

1. Publisher: Xbox → `ControllerState` → `MapControllerToDrive` → `DriveCmd{code, vxN, vyN, wzN}`
   (normalized) → `ip_encode` → LCM `Robot_ControlData`.
2. Harness `run` loop each tick: `cmd = source()` (poll LCM or fixed).
3. `SimHarness::tick(cmd)`:
   a. `TaskId want = router.route(cmd.code)`; set `io.requested_task = want`.
   b. `router.shape(want, cmd, io)` → `io.js_*_desire` in physical units for that task.
   c. `backend.readToPackage(io)` → `manager.tick(io)` (performs the switch + runs active task)
      → `backend.writeFromPackage(io)` → `backend.step()`.
4. GUI: identical, but under the viewer mutex on the physics thread.

## 11. Real-robot & backward-compat considerations

- **Wire layout unchanged** (`{code,vx,vy,wz}`), so the real-robot LCM link and the golden
  `input_protocol` ctest stay bit-compatible. `model`→`code` is a rename only.
- **Normalized-axis semantics** is the one behavioral change: the publisher no longer bakes
  m/s scaling; the active task does. The real-robot backend (B-group, deferred) will reuse
  the same `SimHarness`/`TaskRouter` shaping, so command feel is defined once, in configs.
- ZERO→policy power-on transition (spec §ZeroState) is preserved: harness starts ZERO, and
  routing defaults unknown/`0` codes to ZERO.

## 12. Testing strategy

- **`walk_e2e` / `obs_alignment` / `walk_sim`** must stay bit-exact / green: `WalkTask` via
  config-loaded policy path must load the same `cpg_policy.pt` and produce identical obs.
  (Explicit-path overload keeps the golden tests pinned.)
- **New `task_registry_link` ctest** (§9): registrars survive whole-archive linking.
- **New `task_router` ctest:** `select_code` map (route + collision error), `shape` axis math
  (normalized→ranges, 0→0, clamp, absent-axis→0).
- **`teleop_smoke`** retargeted onto `SimHarness::tick`; still LCM-free (inject `DriveCmd`).
- **Publisher `input_mapping` ctest** updated for normalized output + `buttons.yaml` codes.
- Full suite stays headless / 10-plus tests green; GUI remains dev-only and unlinked by tests.

## 13. Migration (files touched)

**Added**
- `backends/mujoco/sim_harness.{h,cpp}` (glfw-free core)
- `backends/mujoco/sim_harness_gui.{h,cpp}` (links `mujoco_gui`; apps only)
- `core/include/phybot/task_router.h` + `core/src/task_router.cpp`
- `core/include/phybot/controller_config.h` + loader (`core/src/controller_config.cpp`)
- `publisher/config/buttons.yaml` + a small buttons loader
- `tasks/walk/config/controller.yaml`, `tasks/zero/config/controller.yaml`
- ctests: `task_registry_link`, `task_router`

**Changed**
- `core/include/phybot/walk_config.h` + `walk_config.cpp`: add `policy_path`.
- `tasks/walk/walk_task.{h,cpp}`: default-ctor loads policy from config; add `REGISTER_TASK`.
- `tasks/walk/config/walk.yaml`: add `policy_path`.
- `apps/sim_walk.cpp`, `apps/sim_teleop.cpp`: reduce to thin `SimHarness` shells.
- `publisher/include/phybot/input_mapping.h`: normalized output + `buttons.yaml` codes;
  `input_protocol.h`: `model`→`code` rename.
- `CMakeLists.txt`: `phybot_tasks` lib (whole-archive), `PHYBOT_TASKS_DIR` def, wire harness.
- `backends/mujoco/teleop_loop.h`: removed (absorbed into `SimHarness::tick`); update the two
  callers.

## 14. Out of scope (YAGNI)

- Dynamic/plugin (`dlopen`) task loading — tasks are compiled in.
- Hot-reload of configs.
- Per-task **button-layout** remapping (which stick does what) — publisher keeps a fixed
  stick→axis mapping; only ranges + select_code are per-task. (Full raw-controller-state
  forwarding is a possible future extension, noted but not built.)
- Mimic/other unimplemented tasks — the mechanism supports them, but only ZERO + WALK
  (+ the walk_amp example) are wired now.
