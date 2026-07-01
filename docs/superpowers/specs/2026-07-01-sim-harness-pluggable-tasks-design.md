# Sim Harness + Pluggable Tasks (Decentralized Identity, Per-Task Controller Config)

**Date:** 2026-07-01
**Status:** Design (approved: Option 2 — decentralized identity)
**Scope:** `deploy/bdml_phybot_deploy_c2/`
**Related:** [2026-06-28-deploy-refactor-design.md](2026-06-28-deploy-refactor-design.md)

## 1. Motivation & Goal

Adding or swapping a policy task (e.g. the user's `walk_amp`) must not require editing central
files. Today it does: `WalkTask(model_path)` can't self-register (no-arg factory needed); the
`TaskId` enum in `core/` centrally lists every task; routing is hardcoded in
`teleop_loop.h::ModelToTask`; and `sim_walk`/`sim_teleop` duplicate the setup. Task-specific
files (`walk_*`, `zero_*`) also live in `core/`, coupling core to individual tasks.

**Goal:** adding a task costs only files **inside that task's folder** (+ one CMake source line +
one publisher button line), with `core/` fully task-independent:

```
tasks/walk_amp/
  walk_amp_task.h / .cpp        # policy logic + REGISTER_TASK("walk_amp", WalkAmpTask)
  config/
    walk_amp.yaml               # policy_path + obs/action params
    controller.yaml             # select_code: 3  (this task's identity) + axis ranges
  model/walk_amp.pt             # the policy (or referenced from anywhere via config)
```
```cmake
# CMakeLists.txt: one line into the phybot_tasks source list
tasks/walk_amp/walk_amp_task.cpp
```
```yaml
# publisher/config/buttons.yaml: one line
X: 3        # Xbox X → select_code 3 (walk_amp)
```
→ Rebuild. Headless / GUI / teleop / real-robot all pick it up. **No `core/`, harness, app, or
routing edits.**

## 2. Decentralized task identity (Option 2)

**A task's identity is its `select_code`** — a small positive int the task declares in its own
`controller.yaml`. There is no central enum.

- **Remove** `core/task_id.h` (`enum class TaskId`). Replace with `core/task_code.h`:
  ```cpp
  namespace phybot {
  using TaskCode = int;                 // a task's id == its controller.yaml select_code
  constexpr TaskCode kNoTask = 0;       // "none / stay / use default" sentinel
  constexpr int kMaxTaskCode = 64;      // capacity for code-indexed arrays (no heap)
  }
  ```
- `Task` base drops its stored id; identity is external. `checkChange()` returns `kNoTask` (stay)
  by default, so a task never references the (nonexistent) central list:
  ```cpp
  class Task {
   public:
    explicit Task(std::string name) : name_(std::move(name)) {}
    virtual ~Task() = default;
    const std::string& name() const { return name_; }
    virtual void enter(DataPackage&) {}
    virtual void run(DataPackage&) = 0;
    virtual void exit(DataPackage&) {}
    virtual TaskCode checkChange(DataPackage&) { return kNoTask; }  // kNoTask = stay
    virtual bool canExit(DataPackage&) { return true; }
   protected:
    std::string name_;
  };
  ```
- `DataPackage.requested_task` / `current_task` become `TaskCode` (default `kNoTask`).
- `TaskManager` is **code-indexed over a fixed capacity**, not `COUNT`:
  ```cpp
  class TaskManager {
   public:
    void add(std::unique_ptr<Task> t, TaskCode code);   // place a task at its code
    void start(TaskCode code, DataPackage& io);
    void tick(DataPackage& io);   // switch to io.requested_task if a task holds it, else stay; run
    TaskCode current() const;
   private:
    std::array<std::unique_ptr<Task>, kMaxTaskCode> tasks_{};
    TaskCode current_{kNoTask};
    Task*    cur_{nullptr};
  };
  ```
  `tick` ignores out-of-range / unregistered codes (safe), preserving today's bounds-safety tests.
- `TaskRegistry` becomes a **name-keyed append list** (capacity `kMaxTaskCode`), not enum-indexed:
  ```cpp
  class TaskRegistry {
   public:
    using Factory = std::function<std::unique_ptr<Task>()>;
    void add(const char* name, Factory f);        // append; dup name → hard error
    int  size() const;
    const char* name(int i) const;
    std::unique_ptr<Task> make(int i) const;
    static TaskRegistry& instance();
  };
  #define REGISTER_TASK(NAME, CLASS) /* static-init: instance().add(NAME, []{return new CLASS();}) */
  ```
  `REGISTER_TASK("walk", WalkTask)` — **name only**; the code comes from the folder's
  `controller.yaml`, paired in by the harness (§7).
- **Default task:** exactly one task's `controller.yaml` sets `default: true` (the safe power-on /
  standby — ZERO). The harness starts it; the router maps unknown / `kNoTask` codes to it.
- **Startup validation (fail-fast):** duplicate `select_code`, duplicate name, `select_code` out
  of `[1, kMaxTaskCode)`, or ≠1 `default` task → throw with a clear message.

**Robustness note:** identity is a validated `int` (not a stringly-typed call site). Transitions
are driven by `select_code` ints from the controller; task names are used only for
registration/display. This keeps the robustness that motivated the original enum while removing
the central manifest.

## 3. core/ vs tasks/ boundary (relocation)

`core/` holds only task-independent code; each task folder is self-contained.

**Stays in `core/`:** `task.h`, `task_code.h`, `task_manager.h`, `task_registry.h`,
`data_package.h`, `robot_backend.h`, `robot_consts.h`, `policy_runner.{h,cpp}` (generic JIT
wrapper — any policy task reuses it), `input_protocol.{h,cpp}`, `lcm_env.h`, `cubic_traj.h`,
`gravity.h`, plus the new `controller_config.{h,cpp}`, `task_router.{h,cpp}`. Generic tests
(`task_manager`, `input_protocol`, `lcm_loopback`).

**Moves `core/ → tasks/walk/`:** `walk_config.{h,cpp}`, `walk_obs.h`, `walk_history.h`,
`walk_action.h`, `test_walk_*`, `test_obs_alignment`, `test/golden/` (walk golden + `cpg_policy.pt`).
Includes become `walk/walk_config.h` etc.

**Moves `core/ → tasks/zero/`:** `zero_config.{h,cpp}`, `zero_task.h`, `test_zero_task`.

## 4. Config-carried policy path (self-registration enabler)

`WalkConfig` gains `std::string policy_path`, loaded from `tasks/walk/config/walk.yaml` and
resolved via `PHYBOT_ROOT` (absolute → as-is; relative → `PHYBOT_ROOT/<p>`). `WalkTask()`
default-ctor loads the config and builds its `PolicyRunner` from `cfg_.policy_path`; the
`WalkTask(model_path)` overload stays so the golden tests pin `cpg_policy.pt` exactly. Same
pattern for any future policy task.

## 5. Per-task controller config

`tasks/<task>/config/controller.yaml`:
```yaml
select_code: 2                 # this task's identity (int); routing + manager key
default: false                 # exactly one task sets true (the power-on/standby task)
vx: { range: [-0.4, 1.0] }     # normalized stick [-1,1] → physical setpoint for THIS task
vy: { range: [-0.1, 0.1] }
wz: { range: [-1.0, 1.0] }
```
```cpp
struct AxisRange { double lo, hi; double apply(double n) const; };  // n∈[-1,1]→[lo,hi], 0→0, clamp
struct ControllerConfig { TaskCode select_code; bool is_default; AxisRange vx, vy, wz; };
ControllerConfig load_controller_config(const char* task_name);     // PHYBOT_TASKS_DIR/<n>/config/
```
ZERO's `controller.yaml`: `select_code: 1`, `default: true` (axes omitted → held at 0).

## 6. TaskRouter (routing + shaping)

Replaces `teleop_loop.h::ModelToTask`. Works purely in code space:
```cpp
class TaskRouter {
 public:
  void addTask(TaskCode code, const ControllerConfig& cc, bool is_default);  // validates uniqueness
  TaskCode route(TaskCode code) const;    // registered code → itself; unknown/kNoTask → default
  void shape(TaskCode active, const DriveCmd& cmd, DataPackage& io) const;   // normalized→setpoints
};
```
`route` guarantees a valid, registered code (falls back to the default task). `shape` maps
`cmd.vx/vy/wz` (normalized) through the active code's `AxisRange`s into `io.js_*_desire` and sets
`io.requested_task = active`.

## 7. SimHarness (glfw-free core) + separate GUI runner

**Invariant:** headless test binaries must not link glfw/GL/X11. `SimHarness` is glfw-free.

`backends/mujoco/sim_harness.{h,cpp}`:
```cpp
class SimHarness {
 public:
  struct Options { std::string xml_path; };
  explicit SimHarness(const Options&);   // load backend; pair registry names ↔ controller.yaml
                                         // codes → manager.add + router.addTask; start default task.
  void tick(const DriveCmd& cmd);        // route code→switch; shape axes; read→tick→write→step
  using CommandSource = std::function<DriveCmd()>;
  void runHeadless(CommandSource src, double seconds, std::function<bool()> stop = {});
  MujocoBackend& backend(); TaskManager& manager();
};
```
Init pairing (the heart of decentralization):
```cpp
for (int i = 0; i < registry.size(); ++i) {
  const char* name = registry.name(i);
  ControllerConfig cc = load_controller_config(name);
  manager_.add(registry.make(i), cc.select_code);
  router_.addTask(cc.select_code, cc, cc.is_default);
}
manager_.start(router_.defaultCode(), io_);   // power-on = the default task (ZERO)
```
`backends/mujoco/sim_harness_gui.{h,cpp}` (links `mujoco_gui`, apps-only):
```cpp
void RunGui(SimHarness& h, SimHarness::CommandSource src, const char* title);
// physics thread runs h.tick(src()); main thread runs MujocoViewer::renderLoop() (mutex per tick)
```
`sim_walk`/`sim_teleop` become thin shells (fixed source vs LCM source; `--gui` → `RunGui`).
`teleop_loop.h`/`ApplyCmdAndTick` are absorbed into `SimHarness::tick`.

## 8. Publisher (generic bridge + buttons.yaml)

- `MapControllerToDrive` emits **normalized** `vx/vy/wz ∈ [-1,1]` (deadzone only; no physical
  scaling — that's now per-task, task-side) and a `model` = `select_code` from
  `publisher/config/buttons.yaml` (`{ Y:1, B:2, X:3 }`, button→code, last-pressed latches;
  startup default `1`).
- `DriveCmd` wire layout is UNCHANGED: `{model, vx, vy, wz}`. The field stays named `model` and is
  documented as carrying the select-code. Preserves the `input_protocol` golden and the real-robot
  link byte-for-byte.

## 9. Build & linkage (whole-archive)

Self-registration via static-init is stripped inside a `STATIC` lib. New `phybot_tasks` STATIC
lib (`tasks/*/**_task.cpp` + config `.cpp` deps) linked **whole-archive** into consumers so
registrars survive: `$<LINK_LIBRARY:WHOLE_ARCHIVE,phybot_tasks>` (CMake ≥3.24; `-Wl,--whole-archive`
fallback for ≥3.12). A `task_registry_link` ctest asserts registrars survive (guards silent
stripping). `PHYBOT_TASKS_DIR`, `PHYBOT_ROOT` compile-defs for config discovery.

## 10. Command flow (end to end)

1. Publisher: Xbox → `DriveCmd{model=code, vxN, vyN, wzN}` (normalized) → LCM `Robot_ControlData`.
2. Harness loop: `cmd = source()`.
3. `SimHarness::tick(cmd)`: `code = router.route(cmd.model)`; `router.shape(code, cmd, io)`;
   `backend.readToPackage → manager.tick(io) → backend.writeFromPackage → backend.step`.
4. GUI: identical, on the physics thread under the viewer mutex.

## 11. Real-robot & backward-compat

- Wire layout unchanged → real-robot LCM link + `input_protocol` golden stay bit-exact.
- Only behavioral change: publisher axes are normalized (task applies m/s ranges). The real-robot
  backend (B-group, deferred) reuses the same `SimHarness`/`TaskRouter`, so command feel is defined
  once, in per-task configs.
- Power-on ZERO→policy transition preserved: harness starts the `default` task; unknown/0 codes
  route to it.

## 12. Testing strategy

- `walk_e2e` / `obs_alignment` / `walk_sim` stay bit-exact / green (WalkTask(explicit_path) pins
  the golden; relocation is a pure move).
- New: `task_registry_link` (registrars survive whole-archive), `task_router` (route+default+shape+
  dup-code error), `controller_config` (parse select_code/default/ranges).
- `task_manager` test rewritten for `TaskCode` ints + capacity (bounds-safety cases preserved).
- `teleop_smoke` retargeted onto `SimHarness` (headless core; no glfw).
- Full suite green; GUI dev-only and unlinked by tests.

## 13. Migration (files)

**Removed:** `core/include/phybot/task_id.h` (→ `task_code.h`); `backends/mujoco/teleop_loop.h`.

**Moved:** walk_* → `tasks/walk/` (+ `test/golden/`); zero_* → `tasks/zero/` (see §3).

**Added:** `core/include/phybot/task_code.h`, `controller_config.{h,cpp}`, `task_router.{h,cpp}`;
`backends/mujoco/sim_harness.{h,cpp}`, `sim_harness_gui.{h,cpp}`; each task's `config/controller.yaml`;
`publisher/config/buttons.yaml` + loader; ctests `task_registry_link`, `task_router`, `controller_config`.

**Changed:** `task.h` (drop id, `checkChange→kNoTask`), `task_manager.h`, `task_registry.h`
(name-keyed), `data_package.h` (`TaskCode`); `walk_task.{h,cpp}` (name-only register, config policy),
`zero_task.{h,cpp}`; `walk_config.{h,cpp}` (+policy_path); publisher `input_mapping.h`; apps;
`CMakeLists.txt` (phybot_tasks whole-archive, relocation paths, `PHYBOT_ROOT`/`PHYBOT_TASKS_DIR`).

## 14. Out of scope (YAGNI)

- Dynamic (`dlopen`) task loading; hot-reload.
- Per-task **stick-layout** remapping (which stick does what) — publisher keeps a fixed
  stick→axis mapping; only ranges + select_code are per-task.
- Task-to-task forced transitions by name — the `checkChange` return type supports a code, but no
  helper is built; real tasks return `kNoTask`. Mimic and other tasks: mechanism ready, not wired.
