# SimHarness + Pluggable Tasks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make adding a policy task (e.g. `walk_amp`) cost only a task class + one `REGISTER_TASK` line + `<task>.yaml` + `controller.yaml` + one CMake source line + one publisher button line — via a registry-built, config-routed, reusable `SimHarness`.

**Architecture:** Move each policy task's model path into its config so it self-registers (no-arg factory). A `TaskRouter` built from each task's `controller.yaml` replaces the hardcoded `ModelToTask`/clamps. A glfw-free `SimHarness` (backend + registry-built `TaskManager` + router + `tick`) unifies `sim_walk`/`sim_teleop`; a separate `sim_harness_gui` runner adds the window without dragging glfw into headless tests. Task registrars survive via whole-archive linking.

**Tech Stack:** C++17, CMake ≥3.12, CTest, yaml-cpp, libtorch, MuJoCo 3.1.1, LCM.

## Global Constraints

- Work only in `deploy/bdml_phybot_deploy_c2/`; build in-dir (`cmake -S . -B build`).
- All existing ctests stay green; `walk_e2e` + `obs_alignment` stay **bit-exact** (0.000e+00 / float-noise) — never route the golden through the new normalization; golden tests keep `WalkTask(explicit_path)`.
- Headless test binaries must NOT link glfw/GL/X11 (only `sim_walk`/`sim_teleop` link the GUI).
- `DriveCmd` wire layout is UNCHANGED (`{model, vx, vy, wz}`); the field stays named `model` and the router consumes it as the select-code (avoids churn + preserves the `input_protocol` golden and real-robot naming). This is a deliberate deviation from spec §8's rename.
- Config paths in yaml resolve via `PHYBOT_ROOT` (absolute → as-is, relative → `PHYBOT_ROOT/<p>`).
- Commit after every task. Co-author trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: Config-carried policy path (self-registration enabler)

**Files:**
- Modify: `core/include/phybot/walk_config.h` (add `policy_path` field + `PHYBOT_ROOT` resolve helper decl)
- Modify: `core/src/walk_config.cpp` (load `policy_path`, resolve via `PHYBOT_ROOT`)
- Modify: `tasks/walk/config/walk.yaml` (add `policy_path`)
- Modify: `tasks/walk/walk_task.cpp` (default ctor: if `model_path` empty, use `cfg_.policy_path`)
- Modify: `CMakeLists.txt` (add `PHYBOT_ROOT` compile-def to `walk_task` + config-loading test targets)
- Test: `core/test/test_walk_config_policy.cpp` (new)

**Interfaces:**
- Consumes: existing `WalkConfig`, `walk_config()` loader, `WalkTask(model_path="")`.
- Produces: `WalkConfig::policy_path` (std::string, absolute after load); `WalkTask()` default-ctor loads the golden CPG policy from config; `phybot::ResolveRoot(const std::string&)` helper.

- [ ] **Step 1: Write the failing test**

`core/test/test_walk_config_policy.cpp`:
```cpp
#include "phybot/walk_config.h"
#include <cstdio>
#include <string>
int main() {
  phybot::WalkConfig c = phybot::walk_config();
  // policy_path must be populated and absolute (resolved against PHYBOT_ROOT).
  if (c.policy_path.empty() || c.policy_path[0] != '/') {
    std::printf("FAIL policy_path=%s\n", c.policy_path.c_str()); return 1;
  }
  // must point at the CPG policy the sim uses.
  if (c.policy_path.find("cpg_policy.pt") == std::string::npos) {
    std::printf("FAIL not cpg_policy: %s\n", c.policy_path.c_str()); return 1;
  }
  std::printf("OK %s\n", c.policy_path.c_str()); return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Add target + test to `CMakeLists.txt` near the other config tests (mirror `test_walk_numeric`'s
compile-defs `CONFIG_DIR`/`WALK_CONFIG_DIR`, and add `PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"`):
```cmake
add_executable(test_walk_config_policy core/test/test_walk_config_policy.cpp
    core/src/walk_config.cpp core/src/zero_config.cpp)
target_include_directories(test_walk_config_policy PRIVATE core/include core/test)
target_compile_definitions(test_walk_config_policy PRIVATE
    CONFIG_DIR="${CONFIG_DIR_PATH}" WALK_CONFIG_DIR="${WALK_CONFIG_DIR_PATH}"
    ZERO_CONFIG_DIR="${ZERO_CONFIG_DIR_PATH}" PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_walk_config_policy PRIVATE "${YAML_CPP_LIB}")
add_test(NAME walk_config_policy COMMAND test_walk_config_policy)
```
Run: `cmake -S . -B build >/dev/null && cmake --build build --target test_walk_config_policy -j2 && (cd build && ctest -R walk_config_policy --output-on-failure)`
Expected: FAIL (no `policy_path` member yet → compile error, then value error).

- [ ] **Step 3: Add the field + resolver + loader**

`walk_config.h` — add to the struct and declare the resolver:
```cpp
  std::string policy_path;   // absolute path to the policy .pt (resolved from config)
```
```cpp
// Resolve a config path against PHYBOT_ROOT (absolute → unchanged).
std::string ResolveRoot(const std::string& p);
```
`walk_config.cpp` — after the existing loads, add (guard `PHYBOT_ROOT` like `WALK_CONFIG_DIR`):
```cpp
#ifndef PHYBOT_ROOT
#error "PHYBOT_ROOT must be defined via target_compile_definitions"
#endif
std::string phybot::ResolveRoot(const std::string& p) {
  if (!p.empty() && p[0] == '/') return p;
  return std::string(PHYBOT_ROOT) + "/" + p;
}
```
and inside `walk_config()` populate it from `walk.yaml`:
```cpp
  cfg.policy_path = ResolveRoot(walk["policy_path"].as<std::string>());
```
`tasks/walk/config/walk.yaml` — add:
```yaml
policy_path: core/test/golden/cpg_policy.pt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake -S . -B build >/dev/null && cmake --build build --target test_walk_config_policy -j2 && (cd build && ctest -R walk_config_policy --output-on-failure)`
Expected: PASS (`OK /…/core/test/golden/cpg_policy.pt`).

- [ ] **Step 5: WalkTask default-ctor uses config policy; keep golden pinned**

`tasks/walk/walk_task.cpp` — in the ctor, when no explicit path is given, fall back to config.
Locate where `model_path_` is used to build the runner; make the resolved path:
```cpp
// model_path arg wins (tests pin the golden); empty → load from walk.yaml (sim/registry path).
std::string path = model_path.empty() ? cfg_.policy_path : model_path;
// ...use `path` to construct PolicyRunner instead of model_path_...
```
Ensure `cfg_` is loaded (via `walk_config()`) before this line. Add `PHYBOT_ROOT` to the
`walk_task` target's `target_compile_definitions` in `CMakeLists.txt`.

- [ ] **Step 6: Verify golden bit-exact + build**

Run: `cmake --build build -j2 && (cd build && ctest -R 'walk_e2e|obs_alignment|walk_sim|walk_config_policy' --output-on-failure)`
Expected: all PASS; `walk_e2e` still 0.000e+00 (WalkTask(explicit_path) unchanged).

- [ ] **Step 7: Commit**
```bash
git add core/include/phybot/walk_config.h core/src/walk_config.cpp \
        tasks/walk/config/walk.yaml tasks/walk/walk_task.cpp \
        core/test/test_walk_config_policy.cpp CMakeLists.txt
git commit -m "feat(walk): load policy_path from config so WalkTask() self-registers"
```

---

### Task 2: WalkTask self-registration + whole-archive linkage

**Files:**
- Modify: `tasks/walk/walk_task.cpp` (add `REGISTER_TASK`)
- Modify: `CMakeLists.txt` (`phybot_tasks` STATIC lib whole-archived; `PHYBOT_ROOT`/config defs on it)
- Test: `core/test/test_task_registry_link.cpp` (new)

**Interfaces:**
- Consumes: `TaskRegistry`, `REGISTER_TASK`, `WalkTask()` (Task 1), `ZeroState` (already registered).
- Produces: `phybot_tasks` lib; after whole-archive link, `TaskRegistry::instance().has(TaskId::ZERO)` and `has(TaskId::WALK)` are true.

- [ ] **Step 1: Write the failing test**

`core/test/test_task_registry_link.cpp`:
```cpp
#include "phybot/task_registry.h"
#include <cstdio>
int main() {
  auto& r = phybot::TaskRegistry::instance();
  bool ok = r.has(phybot::TaskId::ZERO) && r.has(phybot::TaskId::WALK);
  std::printf("zero=%d walk=%d\n", r.has(phybot::TaskId::ZERO), r.has(phybot::TaskId::WALK));
  return ok ? 0 : 1;
}
```

- [ ] **Step 2: Add REGISTER_TASK for WALK**

`tasks/walk/walk_task.cpp` (end of file, inside/after the `phybot` code as `zero_task.cpp` does):
```cpp
#include "phybot/task_registry.h"
REGISTER_TASK(phybot::TaskId::WALK, "walk", phybot::WalkTask)
```

- [ ] **Step 3: Create phybot_tasks lib + whole-archive; run test to verify it fails without whole-archive**

`CMakeLists.txt` — add after `walk_task`:
```cmake
# All self-registering tasks. Must be whole-archived into consumers or static-init
# registrars get stripped (see task_registry.h note).
add_library(phybot_tasks STATIC tasks/walk/walk_task.cpp tasks/zero/zero_task.cpp
    core/src/walk_config.cpp core/src/zero_config.cpp)
target_include_directories(phybot_tasks PUBLIC core/include tasks)
target_compile_definitions(phybot_tasks PRIVATE
    CONFIG_DIR="${CONFIG_DIR_PATH}" WALK_CONFIG_DIR="${WALK_CONFIG_DIR_PATH}"
    ZERO_CONFIG_DIR="${ZERO_CONFIG_DIR_PATH}" PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(phybot_tasks PUBLIC policy_runner "${YAML_CPP_LIB}")
set_property(TARGET phybot_tasks PROPERTY CXX_STANDARD 17)

# Helper: whole-archive expression with a fallback for CMake < 3.24.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  set(PHYBOT_TASKS_WHOLE "$<LINK_LIBRARY:WHOLE_ARCHIVE,phybot_tasks>")
else()
  set(PHYBOT_TASKS_WHOLE -Wl,--whole-archive phybot_tasks -Wl,--no-whole-archive)
endif()

add_executable(test_task_registry_link core/test/test_task_registry_link.cpp)
target_link_libraries(test_task_registry_link PRIVATE ${PHYBOT_TASKS_WHOLE})
add_test(NAME task_registry_link COMMAND test_task_registry_link)
```
Run: `cmake -S . -B build >/dev/null && cmake --build build --target test_task_registry_link -j2 && (cd build && ctest -R task_registry_link --output-on-failure)`
Expected: PASS (`zero=1 walk=1`). (If you temporarily drop `${PHYBOT_TASKS_WHOLE}` to plain
`phybot_tasks`, it FAILS — that is the stripping this guards against.)

- [ ] **Step 4: Commit**
```bash
git add tasks/walk/walk_task.cpp core/test/test_task_registry_link.cpp CMakeLists.txt
git commit -m "feat(tasks): register WalkTask; phybot_tasks lib whole-archived so registrars survive"
```

---

### Task 3: ControllerConfig struct + per-task loader

**Files:**
- Create: `core/include/phybot/controller_config.h`
- Create: `core/src/controller_config.cpp`
- Create: `tasks/walk/config/controller.yaml`, `tasks/zero/config/controller.yaml`
- Modify: `CMakeLists.txt` (compile `controller_config.cpp` where needed; `PHYBOT_TASKS_DIR` def)
- Test: `core/test/test_controller_config.cpp` (new)

**Interfaces:**
- Consumes: yaml-cpp, `ResolveRoot` (Task 1), `TaskId`.
- Produces:
  ```cpp
  struct AxisRange { double lo, hi; double apply(double n) const; };  // n∈[-1,1] → [lo,hi], 0→0
  struct ControllerConfig { int select_code; AxisRange vx, vy, wz; };
  // Load tasks/<name>/config/controller.yaml (path via PHYBOT_TASKS_DIR).
  ControllerConfig load_controller_config(const char* task_name);
  ```

- [ ] **Step 1: Write the failing test**

`core/test/test_controller_config.cpp`:
```cpp
#include "phybot/controller_config.h"
#include <cmath>
#include <cstdio>
static bool near(double a, double b){ return std::fabs(a-b) < 1e-9; }
int main() {
  auto w = phybot::load_controller_config("walk");
  if (w.select_code != 2) { std::printf("FAIL walk code=%d\n", w.select_code); return 1; }
  if (!near(w.vx.apply(1.0), 1.0) || !near(w.vx.apply(-1.0), -0.4) ||
      !near(w.vx.apply(0.0), 0.0) || !near(w.vx.apply(2.0), 1.0)) {  // clamp
    std::printf("FAIL vx map\n"); return 1; }
  auto z = phybot::load_controller_config("zero");
  if (z.select_code != 1) { std::printf("FAIL zero code=%d\n", z.select_code); return 1; }
  std::printf("OK\n"); return 0;
}
```

- [ ] **Step 2: Run to verify it fails**

Add target (defs `PHYBOT_TASKS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tasks"`, `PHYBOT_ROOT`):
```cmake
add_executable(test_controller_config core/test/test_controller_config.cpp
    core/src/controller_config.cpp core/src/walk_config.cpp core/src/zero_config.cpp)
target_include_directories(test_controller_config PRIVATE core/include core/test)
target_compile_definitions(test_controller_config PRIVATE
    PHYBOT_TASKS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tasks"
    PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    CONFIG_DIR="${CONFIG_DIR_PATH}" WALK_CONFIG_DIR="${WALK_CONFIG_DIR_PATH}"
    ZERO_CONFIG_DIR="${ZERO_CONFIG_DIR_PATH}")
target_link_libraries(test_controller_config PRIVATE "${YAML_CPP_LIB}")
add_test(NAME controller_config COMMAND test_controller_config)
```
Run: `cmake -S . -B build >/dev/null && cmake --build build --target test_controller_config -j2`
Expected: FAIL (header/impl missing).

- [ ] **Step 3: Implement header + loader + yamls**

`core/include/phybot/controller_config.h`:
```cpp
#pragma once
#include <algorithm>
namespace phybot {
struct AxisRange {
  double lo{0.0}, hi{0.0};
  // Map normalized n∈[-1,1] so 0→0, +1→hi, −1→lo (asymmetric), then clamp to [lo,hi].
  double apply(double n) const {
    double v = (n >= 0.0) ? n * hi : n * (-lo);
    return std::clamp(v, lo, hi);
  }
};
struct ControllerConfig {
  int select_code{0};
  AxisRange vx, vy, wz;
};
// Loads PHYBOT_TASKS_DIR/<task_name>/config/controller.yaml. Missing axis → {0,0}.
ControllerConfig load_controller_config(const char* task_name);
}  // namespace phybot
```
`core/src/controller_config.cpp`:
```cpp
#include "phybot/controller_config.h"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <string>
#ifndef PHYBOT_TASKS_DIR
#error "PHYBOT_TASKS_DIR must be defined via target_compile_definitions"
#endif
namespace phybot {
static AxisRange axis(const YAML::Node& n) {
  AxisRange a;
  if (n && n["range"]) { a.lo = n["range"][0].as<double>(); a.hi = n["range"][1].as<double>(); }
  return a;
}
ControllerConfig load_controller_config(const char* task_name) {
  const std::string p =
      std::string(PHYBOT_TASKS_DIR) + "/" + task_name + "/config/controller.yaml";
  const YAML::Node y = YAML::LoadFile(p);
  ControllerConfig c;
  c.select_code = y["select_code"].as<int>();
  c.vx = axis(y["vx"]); c.vy = axis(y["vy"]); c.wz = axis(y["wz"]);
  return c;
}
}  // namespace phybot
```
`tasks/walk/config/controller.yaml`:
```yaml
select_code: 2
vx: { range: [-0.4, 1.0] }
vy: { range: [-0.1, 0.1] }
wz: { range: [-1.0, 1.0] }
```
`tasks/zero/config/controller.yaml`:
```yaml
select_code: 1
```

- [ ] **Step 4: Run to verify it passes**

Run: `cmake -S . -B build >/dev/null && cmake --build build --target test_controller_config -j2 && (cd build && ctest -R controller_config --output-on-failure)`
Expected: PASS (`OK`).

- [ ] **Step 5: Commit**
```bash
git add core/include/phybot/controller_config.h core/src/controller_config.cpp \
        tasks/walk/config/controller.yaml tasks/zero/config/controller.yaml \
        core/test/test_controller_config.cpp CMakeLists.txt
git commit -m "feat(controller): per-task controller.yaml (select_code + axis ranges) + loader"
```

---

### Task 4: TaskRouter (routing + axis shaping)

**Files:**
- Create: `core/include/phybot/task_router.h`
- Create: `core/src/task_router.cpp`
- Test: `core/test/test_task_router.cpp` (new)

**Interfaces:**
- Consumes: `ControllerConfig` (Task 3), `TaskId`, `DriveCmd` (`core/include/phybot/input_protocol.h`), `DataPackage`.
- Produces:
  ```cpp
  class TaskRouter {
   public:
    void addTask(TaskId id, const ControllerConfig& cc);  // throws on duplicate select_code
    TaskId route(int code) const;                         // unknown → TaskId::ZERO
    void shape(TaskId active, const DriveCmd& cmd, DataPackage& io) const;
  };
  ```

- [ ] **Step 1: Write the failing test**

`core/test/test_task_router.cpp`:
```cpp
#include "phybot/task_router.h"
#include "phybot/input_protocol.h"
#include "phybot/data_package.h"
#include <cmath>
#include <cstdio>
static bool near(double a,double b){return std::fabs(a-b)<1e-9;}
int main() {
  using namespace phybot;
  TaskRouter r;
  ControllerConfig walk; walk.select_code=2; walk.vx={-0.4,1.0}; walk.wz={-1.0,1.0};
  ControllerConfig zero; zero.select_code=1;
  r.addTask(TaskId::WALK, walk);
  r.addTask(TaskId::ZERO, zero);
  if (r.route(2)!=TaskId::WALK || r.route(1)!=TaskId::ZERO || r.route(9)!=TaskId::ZERO){
    std::printf("FAIL route\n"); return 1; }
  DriveCmd cmd; cmd.model=2; cmd.vx=1.0f; cmd.wz=-1.0f;
  DataPackage io; r.shape(TaskId::WALK, cmd, io);
  if(!near(io.js_vx_desire,1.0)||!near(io.js_OmegaZ_desire,-1.0)){ std::printf("FAIL shape\n"); return 1; }
  // duplicate code must throw
  bool threw=false; try { ControllerConfig d; d.select_code=2; r.addTask(TaskId::MIMIC,d);}catch(...){threw=true;}
  if(!threw){ std::printf("FAIL dup\n"); return 1; }
  std::printf("OK\n"); return 0;
}
```

- [ ] **Step 2: Run to verify it fails**
```cmake
add_executable(test_task_router core/test/test_task_router.cpp
    core/src/task_router.cpp)
target_include_directories(test_task_router PRIVATE core/include core/test
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcm/include")
add_test(NAME task_router COMMAND test_task_router)
```
Run: `cmake -S . -B build >/dev/null && cmake --build build --target test_task_router -j2`
Expected: FAIL (missing header/impl).

- [ ] **Step 3: Implement router**

`core/include/phybot/task_router.h`:
```cpp
#pragma once
#include <array>
#include <cstddef>
#include "phybot/task_id.h"
#include "phybot/controller_config.h"
namespace phybot {
struct DriveCmd; struct DataPackage;
class TaskRouter {
 public:
  static constexpr int kMaxCode = 64;
  void addTask(TaskId id, const ControllerConfig& cc);
  TaskId route(int code) const;
  void shape(TaskId active, const DriveCmd& cmd, DataPackage& io) const;
 private:
  std::array<int, (std::size_t)TaskId::COUNT> code_{};          // TaskId → select_code (0=unset)
  std::array<ControllerConfig, (std::size_t)TaskId::COUNT> cfg_{};
  std::array<TaskId, kMaxCode> byCode_{};                       // filled with ZERO
  bool byCodeInit_{false};
  void ensureInit();
};
}  // namespace phybot
```
`core/src/task_router.cpp`:
```cpp
#include "phybot/task_router.h"
#include "phybot/input_protocol.h"
#include "phybot/data_package.h"
#include <stdexcept>
namespace phybot {
void TaskRouter::addTask(TaskId id, const ControllerConfig& cc) {
  const int code = cc.select_code;
  if (code <= 0 || code >= kMaxCode) throw std::runtime_error("select_code out of range");
  if (!byCodeInit_) { byCode_.fill(TaskId::ZERO); byCodeInit_ = true; }
  // A select_code already held by a DIFFERENT task is a hard error (no silent shadow).
  for (std::size_t i = 0; i < (std::size_t)TaskId::COUNT; ++i)
    if (code_[i] == code && (TaskId)i != id) throw std::runtime_error("duplicate select_code");
  code_[(std::size_t)id] = code;
  cfg_[(std::size_t)id] = cc;
  byCode_[code] = id;
}
TaskId TaskRouter::route(int code) const {
  if (code > 0 && code < kMaxCode && byCodeInit_) return byCode_[code];
  return TaskId::ZERO;
}
void TaskRouter::shape(TaskId active, const DriveCmd& cmd, DataPackage& io) const {
  const ControllerConfig& c = cfg_[(std::size_t)active];
  io.requested_task  = active;
  io.js_vx_desire     = c.vx.apply(cmd.vx);
  io.js_vy_desire     = c.vy.apply(cmd.vy);
  io.js_OmegaZ_desire = c.wz.apply(cmd.wz);
}
}  // namespace phybot
```
> Note: simplify the duplicate check to the single `for` loop shown (delete the dead `if`
> above it). It throws when a different TaskId already holds `code`.

- [ ] **Step 4: Run to verify it passes**

Run: `cmake -S . -B build >/dev/null && cmake --build build --target test_task_router -j2 && (cd build && ctest -R task_router --output-on-failure)`
Expected: PASS (`OK`).

- [ ] **Step 5: Commit**
```bash
git add core/include/phybot/task_router.h core/src/task_router.cpp \
        core/test/test_task_router.cpp CMakeLists.txt
git commit -m "feat(router): data-driven TaskRouter (code→TaskId + per-task axis shaping)"
```

---

### Task 5: SimHarness (glfw-free core) + retarget teleop_smoke

**Files:**
- Create: `backends/mujoco/sim_harness.h`, `backends/mujoco/sim_harness.cpp`
- Modify: `core/test/test_teleop_smoke.cpp` (drive via `SimHarness`)
- Modify: `CMakeLists.txt` (`sim_harness` lib; retarget `test_teleop_smoke`)
- Remove: `backends/mujoco/teleop_loop.h` (absorbed) — after updating callers
- Test: reuse `teleop_smoke` (ZERO→WALK via codes)

**Interfaces:**
- Consumes: `MujocoBackend`, `TaskManager`, `TaskRegistry::loadInto`, `TaskRouter`, `load_controller_config`, `TaskRegistry::name`, `DriveCmd`.
- Produces:
  ```cpp
  class SimHarness {
   public:
    struct Options { std::string xml_path; };
    explicit SimHarness(const Options&);
    void tick(const DriveCmd& cmd);
    using CommandSource = std::function<DriveCmd()>;
    void runHeadless(CommandSource src, double seconds, std::function<bool()> stop = {});
    MujocoBackend& backend(); TaskManager& manager();
  };
  ```

- [ ] **Step 1: Write/adapt the failing test**

Rewrite `core/test/test_teleop_smoke.cpp` body to use the harness (keep XML_PATH/POLICY_PATH defs):
```cpp
#include "phybot/sim_harness.h"
#include "phybot/input_protocol.h"
#include <cstdio>
int main() {
  phybot::SimHarness h({XML_PATH});
  phybot::DriveCmd zero; zero.model = 1;              // ZERO
  for (int i=0;i<200;++i) h.tick(zero);
  double hz = h.backend().baseHeight();
  phybot::DriveCmd walk; walk.model = 2; walk.vx = 0.6f;   // WALK, normalized
  for (int i=0;i<1500;++i) h.tick(walk);
  double hw = h.backend().baseHeight();
  bool ok = hz > 0.4 && hw > 0.4;                    // stood, then walked without falling
  std::printf("h_zero=%.3f h_walk=%.3f -> %s\n", hz, hw, ok?"OK":"FALL");
  return ok?0:1;
}
```
> `sim_harness.h` re-exports the public header path `phybot/sim_harness.h`; create it under
> `backends/mujoco/` and add `backends/mujoco` to include dirs (as today), or place the header
> so `#include "phybot/sim_harness.h"` resolves — mirror how `mujoco_backend.h` is included.
> Simplest: put the header at `backends/mujoco/sim_harness.h` and include it as
> `#include "sim_harness.h"` (backends/mujoco is already on the include path). Use that form.

- [ ] **Step 2: Implement SimHarness**

`backends/mujoco/sim_harness.h`:
```cpp
#pragma once
#include <functional>
#include <string>
#include "mujoco_backend.h"
#include "phybot/task_manager.h"
#include "phybot/task_router.h"
namespace phybot {
struct DriveCmd;
class SimHarness {
 public:
  struct Options { std::string xml_path; };
  explicit SimHarness(const Options& opt);
  void tick(const DriveCmd& cmd);
  using CommandSource = std::function<DriveCmd()>;
  void runHeadless(CommandSource src, double seconds, std::function<bool()> stop = {});
  MujocoBackend& backend() { return backend_; }
  TaskManager&   manager() { return manager_; }
 private:
  MujocoBackend backend_;
  TaskManager   manager_;
  TaskRouter    router_;
  DataPackage   io_{};        // persists sensor/command state across ticks
};
}  // namespace phybot
```
(Add `#include "phybot/data_package.h"` to this header for the `io_` member.)
`backends/mujoco/sim_harness.cpp`:
```cpp
#include "sim_harness.h"
#include "phybot/task_registry.h"
#include "phybot/controller_config.h"
#include "phybot/input_protocol.h"
#include "phybot/data_package.h"
#include <ctime>
namespace phybot {
SimHarness::SimHarness(const Options& opt) : backend_(opt.xml_path) {
  io_.control_period = 0.002;
  auto& reg = TaskRegistry::instance();
  reg.loadInto(manager_);
  for (std::size_t i = 0; i < (std::size_t)TaskId::COUNT; ++i) {
    TaskId id = (TaskId)i;
    if (!reg.has(id)) continue;
    router_.addTask(id, load_controller_config(reg.name(id)));
  }
  manager_.start(TaskId::ZERO, io_);   // safe power-on
}
void SimHarness::tick(const DriveCmd& cmd) {
  TaskId want = router_.route(cmd.model);
  router_.shape(want, cmd, io_);       // sets requested_task + js_*_desire
  backend_.readToPackage(io_);
  manager_.tick(io_);
  backend_.writeFromPackage(io_);
  backend_.step();
}
void SimHarness::runHeadless(CommandSource src, double seconds, std::function<bool()> stop) {
  const timespec period{0, 2 * 1000 * 1000};
  long total = seconds > 0 ? (long)(seconds / 0.002) : -1;
  for (long t = 0; total < 0 || t < total; ++t) {
    if (stop && stop()) break;
    tick(src());
    nanosleep(&period, nullptr);
  }
}
}  // namespace phybot
```

- [ ] **Step 3: Wire CMake + build**

`CMakeLists.txt`:
```cmake
add_library(sim_harness STATIC backends/mujoco/sim_harness.cpp
    core/src/task_router.cpp core/src/controller_config.cpp)
target_include_directories(sim_harness PUBLIC core/include backends/mujoco tasks
    PRIVATE "${MUJOCO_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcm/include")
target_compile_definitions(sim_harness PRIVATE
    PHYBOT_TASKS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tasks"
    PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(sim_harness PUBLIC mujoco_backend ${PHYBOT_TASKS_WHOLE} "${YAML_CPP_LIB}")
set_property(TARGET sim_harness PROPERTY CXX_STANDARD 17)
```
Retarget the smoke test (drop `walk_task`/`zero_task.cpp`, link `sim_harness`):
```cmake
add_executable(test_teleop_smoke core/test/test_teleop_smoke.cpp)
target_include_directories(test_teleop_smoke PRIVATE core/include backends/mujoco tasks
    "${MUJOCO_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcm/include")
target_compile_definitions(test_teleop_smoke PRIVATE
    POLICY_PATH="${POLICY_PT_PATH}" XML_PATH="${PHYBOT_XML_PATH}")
target_link_libraries(test_teleop_smoke PRIVATE sim_harness)
set_property(TARGET test_teleop_smoke PROPERTY CXX_STANDARD 17)
add_test(NAME teleop_smoke COMMAND test_teleop_smoke)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cmake -S . -B build >/dev/null && cmake --build build --target test_teleop_smoke -j2 && (cd build && ctest -R teleop_smoke --output-on-failure)`
Expected: PASS (`h_zero≈0.7 h_walk>0.4 -> OK`).

- [ ] **Step 5: Commit**
```bash
git add backends/mujoco/sim_harness.h backends/mujoco/sim_harness.cpp \
        core/test/test_teleop_smoke.cpp CMakeLists.txt
git commit -m "feat(harness): glfw-free SimHarness (registry+router+backend); retarget teleop_smoke"
```

---

### Task 6: GUI runner (sim_harness_gui) — apps only

**Files:**
- Create: `backends/mujoco/sim_harness_gui.h`, `backends/mujoco/sim_harness_gui.cpp`
- Modify: `CMakeLists.txt` (`sim_harness_gui` lib linking `mujoco_gui` + `sim_harness`)

**Interfaces:**
- Consumes: `SimHarness` (Task 5), `MujocoViewer` (`backends/mujoco/mujoco_viewer.h`), `DriveCmd`.
- Produces: `void RunGui(SimHarness& h, SimHarness::CommandSource src, const char* title);`

- [ ] **Step 1: Implement the GUI runner**

`backends/mujoco/sim_harness_gui.h`:
```cpp
#pragma once
#include "sim_harness.h"
namespace phybot {
// Physics thread runs h.tick(src()); main thread runs the MujocoViewer render loop.
void RunGui(SimHarness& h, SimHarness::CommandSource src, const char* title);
}  // namespace phybot
```
`backends/mujoco/sim_harness_gui.cpp`:
```cpp
#include "sim_harness_gui.h"
#include "mujoco_viewer.h"
#include "phybot/input_protocol.h"
#include <thread>
#include <mutex>
#include <ctime>
namespace phybot {
void RunGui(SimHarness& h, SimHarness::CommandSource src, const char* title) {
  MujocoViewer viewer(title);
  std::thread physics([&]{
    viewer.loadModel(h.backend().model(), h.backend().data());
    const timespec dt{0, 2 * 1000 * 1000};
    while (!viewer.exitRequested()) {
      {
        std::lock_guard<std::recursive_mutex> lk(viewer.mutex());
        if (viewer.running()) { h.tick(src()); viewer.trackCamera(); }
        else { /* paused: keep source drained if it polls LCM */ src(); }
      }
      nanosleep(&dt, nullptr);
    }
  });
  viewer.renderLoop();
  physics.join();
}
}  // namespace phybot
```
> The paused branch calls `src()` (so an LCM-polling source keeps draining) but does not step.
> `tick` already steps physics; do NOT also `mj_forward` here (the backend owns stepping).

- [ ] **Step 2: Wire CMake**
```cmake
add_library(sim_harness_gui STATIC backends/mujoco/sim_harness_gui.cpp)
target_include_directories(sim_harness_gui PUBLIC backends/mujoco
    PRIVATE "${MUJOCO_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcm/include")
target_link_libraries(sim_harness_gui PUBLIC sim_harness mujoco_gui)
set_property(TARGET sim_harness_gui PROPERTY CXX_STANDARD 17)
```

- [ ] **Step 3: Build to verify it links**

Run: `cmake -S . -B build >/dev/null && cmake --build build --target sim_harness_gui -j2`
Expected: `Built target sim_harness_gui` (compiles + links against mujoco_gui).

- [ ] **Step 4: Commit**
```bash
git add backends/mujoco/sim_harness_gui.h backends/mujoco/sim_harness_gui.cpp CMakeLists.txt
git commit -m "feat(harness): RunGui runner (physics thread + MujocoViewer), apps-only link"
```

---

### Task 7: Publisher — normalized axes + buttons.yaml

**Files:**
- Create: `publisher/config/buttons.yaml`
- Create: `publisher/include/phybot/buttons_config.h`, `publisher/src/buttons_config.cpp`
- Modify: `publisher/include/phybot/input_mapping.h` (normalized output + button→code via config)
- Modify: `publisher/test/test_input_mapping.cpp` (assert normalized + codes)
- Modify: `CMakeLists.txt` (buttons_config; PHYBOT_ROOT def where needed)

**Interfaces:**
- Consumes: `ControllerState`, `MappingState`, `DriveCmd`, yaml-cpp.
- Produces:
  ```cpp
  struct ButtonsConfig { int zero_code, walk_code, /*extensible*/ ...; int code_for(const std::string& btn) const; };
  ButtonsConfig load_buttons_config();   // publisher/config/buttons.yaml
  // MapControllerToDrive now: vx/vy/wz ∈ [-1,1] (deadzone only), model=code from buttons.
  ```

- [ ] **Step 1: Write the failing test**

Update `publisher/test/test_input_mapping.cpp` to assert normalized output + config codes:
```cpp
// left stick fully forward → vx == +1.0 (normalized, not *0.6)
ApplyJsEvent(JS_EVENT_AXIS, 1, -32767, st);   // left_y up
DriveCmd d = MapControllerToDrive(st, ms);
CHECK(fapprox(d.vx, 1.0f));
// B press → model == walk_code from buttons.yaml (2)
ApplyJsEvent(JS_EVENT_BUTTON, 1, 1, st);
d = MapControllerToDrive(st, ms);
CHECK(d.model == 2);
```
(Keep existing structure; adjust expected values from the old `*0.6` to `1.0`.)

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build --target test_input_mapping -j2 && (cd build && ctest -R input_mapping --output-on-failure)`
Expected: FAIL (old scale `0.6`).

- [ ] **Step 3: Implement buttons config + normalized mapping**

`publisher/config/buttons.yaml`:
```yaml
# Xbox button → select_code (must match each task's controller.yaml select_code).
Y: 1        # ZERO
B: 2        # walk (CPG)
# X: 3      # walk_amp (uncomment when added)
```
`publisher/include/phybot/buttons_config.h`:
```cpp
#pragma once
#include <string>
#include <map>
namespace phybot {
struct ButtonsConfig {
  std::map<std::string, int> code;      // button name → select_code
  int code_for(const std::string& b, int fallback) const {
    auto it = code.find(b); return it == code.end() ? fallback : it->second;
  }
};
ButtonsConfig load_buttons_config();     // PHYBOT_ROOT/publisher/config/buttons.yaml
}  // namespace phybot
```
`publisher/src/buttons_config.cpp`:
```cpp
#include "phybot/buttons_config.h"
#include <yaml-cpp/yaml.h>
#ifndef PHYBOT_ROOT
#error "PHYBOT_ROOT must be defined"
#endif
namespace phybot {
ButtonsConfig load_buttons_config() {
  ButtonsConfig c;
  const YAML::Node y = YAML::LoadFile(std::string(PHYBOT_ROOT) + "/publisher/config/buttons.yaml");
  for (auto it = y.begin(); it != y.end(); ++it)
    c.code[it->first.as<std::string>()] = it->second.as<int>();
  return c;
}
}  // namespace phybot
```
`publisher/include/phybot/input_mapping.h` — change scaling to normalized (deadzone only) and
resolve `model` from a `ButtonsConfig` member of `MappingState` (load once). Concretely: replace
`vx = -dz(left_y)*0.6` → `vx = -dz(left_y)`, `vy = dz(left_x)` (drop `*0.1`),
`wz = -dz(right_x)` (drop `*1.0`); on B rising set `model = buttons.code_for("B", 2)`, on Y set
`buttons.code_for("Y", 1)`. Keep deadzone `dz` and latching semantics.

- [ ] **Step 4: Run to verify it passes**

Run: `cmake -S . -B build >/dev/null && cmake --build build --target test_input_mapping -j2 && (cd build && ctest -R input_mapping --output-on-failure)`
Expected: PASS.

- [ ] **Step 5: Verify input_protocol golden still bit-exact**

Run: `(cd build && ctest -R input_protocol --output-on-failure)`
Expected: PASS (wire layout unchanged; only mapping values changed).

- [ ] **Step 6: Commit**
```bash
git add publisher/config/buttons.yaml publisher/include/phybot/buttons_config.h \
        publisher/src/buttons_config.cpp publisher/include/phybot/input_mapping.h \
        publisher/test/test_input_mapping.cpp CMakeLists.txt
git commit -m "feat(publisher): normalized axes + buttons.yaml (button→select_code)"
```

---

### Task 8: Thin apps + remove teleop_loop.h

**Files:**
- Rewrite: `apps/sim_walk.cpp`, `apps/sim_teleop.cpp` (SimHarness shells; `--gui`→`RunGui`)
- Delete: `backends/mujoco/teleop_loop.h`
- Modify: `CMakeLists.txt` (sim_walk/sim_teleop link `sim_harness` + `sim_harness_gui`)

**Interfaces:**
- Consumes: `SimHarness`, `RunGui`, `DriveCmd`, `ip_decode` (sim_teleop), LCM.
- Produces: final apps; no new interfaces.

- [ ] **Step 1: Rewrite sim_walk.cpp**
```cpp
#include "sim_harness.h"
#include "sim_harness_gui.h"
#include "phybot/input_protocol.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#ifndef XML_PATH
#define XML_PATH "assets/robot/phybot_c2/xml/phybot_c2.xml"
#endif
int main(int argc, char** argv) {
  bool gui=false; std::vector<const char*> pos;
  for (int i=1;i<argc;++i){ std::string a=argv[i]; if(a=="--gui")gui=true; else pos.push_back(argv[i]); }
  const char* xml = (pos.size()>0 && pos[0][0]) ? pos[0] : XML_PATH;
  double vx = (pos.size()>1 && pos[1][0]) ? std::atof(pos[1]) : 0.4;   // normalized [-1,1]
  double sec = (pos.size()>2 && pos[2][0]) ? std::atof(pos[2]) : 5.0;
  phybot::SimHarness h({xml});
  auto src = [&]{ phybot::DriveCmd c; c.model=2; c.vx=(float)vx; return c; };  // WALK, forward
  if (gui) phybot::RunGui(h, src, "sim_walk");
  else {
    h.runHeadless(src, sec);
    std::printf("[sim_walk] height=%.4f x=%.4f\n", h.backend().baseHeight(), h.backend().baseX());
  }
  return 0;
}
```
> `sim_walk`'s vx is now normalized (task applies the range). Document in the usage comment.

- [ ] **Step 2: Rewrite sim_teleop.cpp**
```cpp
#include "sim_harness.h"
#include "sim_harness_gui.h"
#include "phybot/input_protocol.h"
#include "phybot/lcm_env.h"
#include <lcm/lcm-cpp.hpp>
#include <cstdio>
#include <string>
#include <vector>
#ifndef XML_PATH
#define XML_PATH "assets/robot/phybot_c2/xml/phybot_c2.xml"
#endif
namespace {
struct Holder {
  phybot::DriveCmd latest;   // default model=0 → ZERO
  void on(const lcm::ReceiveBuffer*, const std::string&, const custom_lcm::system2Algorithm_packet* p){
    phybot::DriveCmd d; if (phybot::ip_decode(*p, d)) latest = d;
  }
};
}
int main(int argc, char** argv) {
  bool gui=false; std::vector<const char*> pos;
  for (int i=1;i<argc;++i){ std::string a=argv[i]; if(a=="--gui")gui=true; else pos.push_back(argv[i]); }
  const char* xml = (pos.size()>0 && pos[0][0]) ? pos[0] : XML_PATH;
  phybot::AnnounceLcmInit("sim_teleop");
  lcm::LCM lcm; if(!lcm.good()){ std::printf("[sim_teleop] LCM init failed\n"); return 1; }
  Holder holder; lcm.subscribe("Robot_ControlData", &Holder::on, &holder);
  phybot::SimHarness h({xml});
  auto src = [&]{ lcm.handleTimeout(0); return holder.latest; };
  if (gui) phybot::RunGui(h, src, "sim_teleop");
  else h.runHeadless(src, 0);   // forever
  return 0;
}
```

- [ ] **Step 3: Delete teleop_loop.h + wire CMake**

`rm backends/mujoco/teleop_loop.h`. Update sim_walk/sim_teleop targets:
```cmake
target_link_libraries(sim_walk PRIVATE sim_harness sim_harness_gui)
target_compile_definitions(sim_walk PRIVATE XML_PATH="${PHYBOT_XML_PATH}")
# ...
target_link_libraries(sim_teleop PRIVATE sim_harness sim_harness_gui phybot_lcm)
target_compile_definitions(sim_teleop PRIVATE XML_PATH="${PHYBOT_XML_PATH}")
```
Remove old `walk_task`/`zero_task.cpp`/`mujoco_gui` direct links from these two targets (now
pulled transitively via `sim_harness`/`sim_harness_gui`). Confirm no other file `#include`s
`teleop_loop.h`.

- [ ] **Step 4: Full build + verify**

Run: `cmake -S . -B build >/dev/null && cmake --build build -j2 && (cd build && ctest --output-on-failure)`
Expected: all tests PASS (10 originals retargeted + `walk_config_policy`, `task_registry_link`,
`controller_config`, `task_router`). Then:
Run: `./build/sim_walk` → prints height≈0.7 (WALK via harness, headless).
Run: `timeout 4 stdbuf -oL ./build/sim_walk --gui > /tmp/g.log 2>&1; grep -q sim_walk /tmp/g.log && echo GUI-STARTED`

- [ ] **Step 5: Commit**
```bash
git add apps/sim_walk.cpp apps/sim_teleop.cpp CMakeLists.txt
git rm backends/mujoco/teleop_loop.h
git commit -m "refactor(apps): sim_walk/sim_teleop as thin SimHarness shells; drop teleop_loop.h"
```

---

## Done-when
- Adding `walk_amp` = `WalkAmpTask` + `REGISTER_TASK` + `walk_amp.yaml` + `controller.yaml`
  (`select_code:3`) + one `phybot_tasks` source line + `X: 3` in `buttons.yaml`. No harness/app edits.
- `ctest` green (incl. 4 new tests); `walk_e2e`/`obs_alignment` bit-exact; `sim_walk` headless
  WALK_OK; GUI apps link `mujoco_gui`, headless tests do not.
