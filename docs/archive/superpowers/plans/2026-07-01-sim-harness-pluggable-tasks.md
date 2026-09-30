# SimHarness + Pluggable Tasks (Decentralized Identity) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`).

**Goal:** Make adding a task cost only files in its own folder (+1 CMake line, +1 publisher button line). A task's identity is its `controller.yaml select_code`; `core/` is task-independent.

**Architecture:** Drop the central `TaskId` enum → `TaskCode` (int) over a fixed-capacity, code-indexed FSM. Relocate `walk_*`/`zero_*` out of `core/` into their task folders. Pair name-keyed registry entries with each task's `controller.yaml` code in a glfw-free `SimHarness`; a separate `RunGui` adds the window. Whole-archive the tasks lib so registrars survive.

**Tech Stack:** C++17, CMake ≥3.12, CTest, yaml-cpp, libtorch, MuJoCo 3.1.1, LCM.

**Design source:** `docs/superpowers/specs/2026-07-01-sim-harness-pluggable-tasks-design.md` (read §2 identity, §3 relocation before starting).

## Global Constraints

- Work in `deploy/bdml_phybot_deploy_c2/`; build in-dir. Commit after every task with trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- `walk_e2e` + `obs_alignment` stay **bit-exact**; golden tests keep `WalkTask(explicit_path)`.
- Headless tests never link glfw/GL/X11 (only `sim_walk`/`sim_teleop`).
- `DriveCmd` wire layout unchanged: `{model, vx, vy, wz}`; field stays `model`, carries the select-code.
- yaml paths resolve via `PHYBOT_ROOT` (abs → as-is, rel → `PHYBOT_ROOT/<p>`).
- After every task: `cmake -S . -B build && cmake --build build -j2 && (cd build && ctest)` must be green.

---

### Task 1: Relocate task-specific files out of core (pure move)

No behavior change — this is a move + include/CMake fixup verified by ctest staying green.

**Files:**
- Move `core/include/phybot/{walk_config.h,walk_obs.h,walk_history.h,walk_action.h}` → `tasks/walk/`
- Move `core/src/walk_config.cpp` → `tasks/walk/walk_config.cpp`
- Move `core/test/{test_walk_e2e,test_walk_numeric,test_walk_sim,test_obs_alignment}.cpp` → `tasks/walk/test/`
- Move `core/test/golden/` → `tasks/walk/test/golden/`
- Move `core/include/phybot/{zero_config.h,zero_task.h}` → `tasks/zero/`, `core/src/zero_config.cpp` → `tasks/zero/zero_config.cpp`, `core/test/test_zero_task.cpp` → `tasks/zero/test/`
- Modify: every `#include "phybot/walk_*.h"`→`"walk/walk_*.h"`, `"phybot/zero_*.h"`→`"zero/zero_*.h"`; `CMakeLists.txt` paths + `GOLDEN_DIR`/`POLICY_PT_PATH`.

- [ ] **Step 1: Move the files with git mv**
```bash
cd deploy/bdml_phybot_deploy_c2
mkdir -p tasks/walk/test tasks/zero/test
git mv core/include/phybot/walk_config.h  tasks/walk/walk_config.h
git mv core/include/phybot/walk_obs.h     tasks/walk/walk_obs.h
git mv core/include/phybot/walk_history.h tasks/walk/walk_history.h
git mv core/include/phybot/walk_action.h  tasks/walk/walk_action.h
git mv core/src/walk_config.cpp           tasks/walk/walk_config.cpp
git mv core/test/test_walk_e2e.cpp        tasks/walk/test/test_walk_e2e.cpp
git mv core/test/test_walk_numeric.cpp    tasks/walk/test/test_walk_numeric.cpp
git mv core/test/test_walk_sim.cpp        tasks/walk/test/test_walk_sim.cpp
git mv core/test/test_obs_alignment.cpp   tasks/walk/test/test_obs_alignment.cpp
git mv core/test/golden                   tasks/walk/test/golden
git mv core/include/phybot/zero_config.h  tasks/zero/zero_config.h
git mv core/include/phybot/zero_task.h    tasks/zero/zero_task.h
git mv core/src/zero_config.cpp           tasks/zero/zero_config.cpp
git mv core/test/test_zero_task.cpp       tasks/zero/test/test_zero_task.cpp
```

- [ ] **Step 2: Fix includes**

Rewrite the moved headers' include paths across the tree (tasks are on the `tasks/` include path,
so `#include "walk/walk_config.h"`, `#include "zero/zero_task.h"`). Find every reference:
```bash
grep -rlE 'phybot/(walk_|zero_)' --include='*.h' --include='*.cpp' . | grep -v build
```
For each hit replace `phybot/walk_config.h`→`walk/walk_config.h`, `phybot/walk_obs.h`→`walk/walk_obs.h`,
`phybot/walk_history.h`→`walk/walk_history.h`, `phybot/walk_action.h`→`walk/walk_action.h`,
`phybot/zero_config.h`→`zero/zero_config.h`, `phybot/zero_task.h`→`zero/zero_task.h`.
(Files: `walk_task.h/.cpp`, `walk_obs.h`, `walk_action.h`, `zero_task.cpp`, the moved tests, apps.)

- [ ] **Step 3: Fix CMakeLists paths**

Update every `core/src/walk_config.cpp`→`tasks/walk/walk_config.cpp`,
`core/src/zero_config.cpp`→`tasks/zero/zero_config.cpp`, the test source paths, and the golden dir
defs: `GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tasks/walk/test/golden"`,
`POLICY_PT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tasks/walk/test/golden/cpg_policy.pt"`. Add include dirs
`tasks/walk`/`tasks/zero` where walk/zero headers are consumed (or rely on `tasks` already being on
the include path — verify with the build).

- [ ] **Step 4: Build + full ctest (must stay green, bit-exact)**

Run: `cmake -S . -B build >/dev/null && cmake --build build -j2 && (cd build && ctest --output-on-failure)`
Expected: 10/10 PASS; `walk_e2e` still 0.000e+00.

- [ ] **Step 5: Commit**
```bash
git add -A
git commit -m "refactor(core): relocate walk_*/zero_* out of core into their task folders"
```

---

### Task 2: Decentralized identity — TaskCode over fixed capacity

Atomic swap of `enum TaskId` → `TaskCode` (int). Keep the tree compiling + ctest green; apps are
minimally adapted (they become thin shells in Task 10).

**Files:**
- Create: `core/include/phybot/task_code.h`; Remove: `core/include/phybot/task_id.h`
- Modify: `task.h`, `task_manager.h`, `task_registry.h`, `data_package.h`, `zero_task.{h,cpp}`,
  `walk_task.{h,cpp}`, `core/test/test_task_manager.cpp`, `backends/mujoco/teleop_loop.h`,
  `apps/sim_walk.cpp`, `apps/sim_teleop.cpp`, `CMakeLists.txt`

**Interfaces (produced):**
```cpp
using TaskCode = int; constexpr TaskCode kNoTask = 0; constexpr int kMaxTaskCode = 64;
class Task { explicit Task(std::string name); const std::string& name() const;
             virtual TaskCode checkChange(DataPackage&){return kNoTask;} /*enter/run/exit/canExit*/ };
class TaskManager { void add(std::unique_ptr<Task>, TaskCode); void start(TaskCode, DataPackage&);
                    void tick(DataPackage&); TaskCode current() const; };
class TaskRegistry { void add(const char* name, Factory); int size() const; const char* name(int) const;
                     std::unique_ptr<Task> make(int) const; static TaskRegistry& instance(); };
#define REGISTER_TASK(NAME, CLASS)   // name-keyed
```

- [ ] **Step 1: task_code.h + rewrite Task base**

Create `core/include/phybot/task_code.h`:
```cpp
#pragma once
namespace phybot {
using TaskCode = int;                 // a task's id == its controller.yaml select_code
constexpr TaskCode kNoTask = 0;       // "none / stay / use default"
constexpr int kMaxTaskCode = 64;      // capacity for code-indexed arrays
}  // namespace phybot
```
`core/include/phybot/task.h` — drop the id; `#include "phybot/task_code.h"` (remove task_id.h):
```cpp
#pragma once
#include <string>
#include <utility>
#include "phybot/task_code.h"
namespace phybot {
struct DataPackage;
class Task {
 public:
  explicit Task(std::string name) : name_(std::move(name)) {}
  virtual ~Task() = default;
  const std::string& name() const { return name_; }
  virtual void enter(DataPackage&) {}
  virtual void run(DataPackage&) = 0;
  virtual void exit(DataPackage&) {}
  virtual TaskCode checkChange(DataPackage&) { return kNoTask; }   // kNoTask = stay
  virtual bool canExit(DataPackage&) { return true; }
 protected:
  std::string name_;
};
}  // namespace phybot
```
`rm core/include/phybot/task_id.h`.

- [ ] **Step 2: TaskManager code-indexed + TaskRegistry name-keyed + DataPackage**

`task_manager.h`:
```cpp
#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include "phybot/task_code.h"
#include "phybot/task.h"
#include "phybot/data_package.h"
namespace phybot {
class TaskManager {
 public:
  void add(std::unique_ptr<Task> t, TaskCode code) {
    if (code > 0 && code < kMaxTaskCode) tasks_[(std::size_t)code] = std::move(t);
  }
  void start(TaskCode code, DataPackage& io) {
    switchTo(code, io); io.current_task = current_;
  }
  void tick(DataPackage& io) {
    TaskCode want = io.requested_task;
    if (cur_) { TaskCode self = cur_->checkChange(io); if (self != kNoTask) want = self; }
    if (want != current_ && valid(want) && (!cur_ || cur_->canExit(io))) switchTo(want, io);
    io.current_task = current_;
    if (cur_) cur_->run(io);
  }
  TaskCode current() const { return current_; }
 private:
  bool valid(TaskCode c) const { return c > 0 && c < kMaxTaskCode && tasks_[(std::size_t)c]; }
  void switchTo(TaskCode c, DataPackage& io) {
    if (!valid(c)) return;
    if (cur_) cur_->exit(io);
    current_ = c; cur_ = tasks_[(std::size_t)c].get(); cur_->enter(io);
  }
  std::array<std::unique_ptr<Task>, kMaxTaskCode> tasks_{};
  TaskCode current_{kNoTask};
  Task* cur_{nullptr};
};
}  // namespace phybot
```
`task_registry.h` — name-keyed append list:
```cpp
#pragma once
#include <array>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include "phybot/task_code.h"
#include "phybot/task.h"
namespace phybot {
class TaskRegistry {
 public:
  using Factory = std::function<std::unique_ptr<Task>()>;
  static TaskRegistry& instance() { static TaskRegistry r; return r; }
  void add(const char* name, Factory f) {
    for (int i = 0; i < n_; ++i) if (std::strcmp(names_[i], name) == 0)
      throw std::runtime_error("duplicate task name");
    if (n_ >= kMaxTaskCode) throw std::runtime_error("too many tasks");
    names_[n_] = name; factories_[n_] = std::move(f); ++n_;
  }
  int size() const { return n_; }
  const char* name(int i) const { return names_[i]; }
  std::unique_ptr<Task> make(int i) const { return factories_[i](); }
 private:
  TaskRegistry() = default;
  int n_{0};
  std::array<const char*, kMaxTaskCode> names_{};
  std::array<Factory, kMaxTaskCode> factories_{};
};
}  // namespace phybot
#define REGISTER_TASK(NAME, CLASS)                                            \
  namespace { struct CLASS##_Reg { CLASS##_Reg() {                            \
    ::phybot::TaskRegistry::instance().add(                                   \
        (NAME), [] { return std::unique_ptr<::phybot::Task>(new CLASS()); }); \
  } } CLASS##_reg_instance; }
```
`data_package.h` — change the two fields:
```cpp
#include "phybot/task_code.h"
// ...
  TaskCode requested_task{kNoTask};
  TaskCode current_task{kNoTask};
```

- [ ] **Step 3: Update tasks (name-only ctor + register)**

`tasks/zero/zero_task.h`: `ZeroState() : Task("zero") {}` (drop `TaskId::ZERO`).
`tasks/zero/zero_task.cpp`: `REGISTER_TASK("zero", ZeroState)`.
`tasks/walk/walk_task.cpp`: base ctor `Task("walk")`; `REGISTER_TASK("walk", phybot::WalkTask)`.
Remove any `#include "phybot/task_id.h"`.

- [ ] **Step 4: Rewrite test_task_manager.cpp for codes**

Replace `TaskId::ZERO`→`1`, `TaskId::WALK_AMP`→`2`, `TaskId::MIMIC`→`3` (unregistered),
`TaskId::COUNT`→`kMaxTaskCode` (out-of-range sentinel), and `Task(id,"n")`→`Task("n")`. Registry
checks: `reg.size()`, `reg.name(i)`. Keep the bounds-safety cases (`requested_task=3` unregistered →
stays; `=kMaxTaskCode` out-of-range → stays). `manager.add(make(...), code)` passes explicit codes.

- [ ] **Step 5: Minimally adapt teleop_loop.h + apps to codes (thin-shell rewrite is Task 10)**

`teleop_loop.h`: `ModelToTask` → `int ModelToTask(int8_t m){ return m==2?2:1; }` (codes); `io.requested_task`
is a code. sim_walk/sim_teleop: replace `TaskId::ZERO`→`1`, `TaskId::WALK`→`2`; `manager.add(make_unique<ZeroState>())`
→ `manager.add(std::make_unique<ZeroState>(), 1)`, WalkTask → code 2; `TaskName` switches on int.

- [ ] **Step 6: Build + full ctest green**

Run: `cmake -S . -B build >/dev/null && cmake --build build -j2 && (cd build && ctest --output-on-failure)`
Expected: all PASS; `walk_e2e` bit-exact.

- [ ] **Step 7: Commit**
```bash
git add -A && git rm --cached core/include/phybot/task_id.h 2>/dev/null; \
git commit -m "refactor(core): TaskId enum -> TaskCode int over fixed capacity; name-keyed registry"
```

---

### Task 3: ControllerConfig + per-task controller.yaml

**Files:** Create `core/include/phybot/controller_config.h`, `core/src/controller_config.cpp`,
`tasks/walk/config/controller.yaml`, `tasks/zero/config/controller.yaml`; Modify `CMakeLists.txt`;
Test `core/test/test_controller_config.cpp`.

**Interfaces (produced):**
```cpp
struct AxisRange { double lo{0}, hi{0}; double apply(double n) const; };
struct ControllerConfig { TaskCode select_code{0}; bool is_default{false}; AxisRange vx, vy, wz; };
ControllerConfig load_controller_config(const char* task_name);   // PHYBOT_TASKS_DIR/<n>/config/
```

- [ ] **Step 1: Failing test** — `core/test/test_controller_config.cpp`:
```cpp
#include "phybot/controller_config.h"
#include <cmath>
#include <cstdio>
static bool nr(double a,double b){return std::fabs(a-b)<1e-9;}
int main(){ using namespace phybot;
  auto w=load_controller_config("walk");
  if(w.select_code!=2||w.is_default) {std::printf("FAIL walk\n");return 1;}
  if(!nr(w.vx.apply(1),1.0)||!nr(w.vx.apply(-1),-0.4)||!nr(w.vx.apply(0),0)||!nr(w.vx.apply(2),1.0)){std::printf("FAIL map\n");return 1;}
  auto z=load_controller_config("zero");
  if(z.select_code!=1||!z.is_default){std::printf("FAIL zero\n");return 1;}
  std::printf("OK\n");return 0; }
```

- [ ] **Step 2: Add target (defs `PHYBOT_TASKS_DIR`), run → FAIL**
```cmake
add_executable(test_controller_config core/test/test_controller_config.cpp core/src/controller_config.cpp)
target_include_directories(test_controller_config PRIVATE core/include core/test)
target_compile_definitions(test_controller_config PRIVATE PHYBOT_TASKS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tasks")
target_link_libraries(test_controller_config PRIVATE "${YAML_CPP_LIB}")
add_test(NAME controller_config COMMAND test_controller_config)
```
Run: `cmake -S . -B build >/dev/null && cmake --build build --target test_controller_config -j2` → FAIL.

- [ ] **Step 3: Implement** — `controller_config.h`:
```cpp
#pragma once
#include <algorithm>
#include "phybot/task_code.h"
namespace phybot {
struct AxisRange { double lo{0}, hi{0};
  double apply(double n) const { double v=(n>=0)?n*hi:n*(-lo); return std::clamp(v,lo,hi); } };
struct ControllerConfig { TaskCode select_code{0}; bool is_default{false}; AxisRange vx, vy, wz; };
ControllerConfig load_controller_config(const char* task_name);
}
```
`controller_config.cpp`:
```cpp
#include "phybot/controller_config.h"
#include <yaml-cpp/yaml.h>
#include <string>
#ifndef PHYBOT_TASKS_DIR
#error "PHYBOT_TASKS_DIR must be defined"
#endif
namespace phybot {
static AxisRange axis(const YAML::Node& n){ AxisRange a; if(n&&n["range"]){a.lo=n["range"][0].as<double>();a.hi=n["range"][1].as<double>();} return a; }
ControllerConfig load_controller_config(const char* name){
  const YAML::Node y=YAML::LoadFile(std::string(PHYBOT_TASKS_DIR)+"/"+name+"/config/controller.yaml");
  ControllerConfig c; c.select_code=y["select_code"].as<int>();
  c.is_default=y["default"]?y["default"].as<bool>():false;
  c.vx=axis(y["vx"]); c.vy=axis(y["vy"]); c.wz=axis(y["wz"]); return c; }
}
```
`tasks/walk/config/controller.yaml`:
```yaml
select_code: 2
default: false
vx: { range: [-0.4, 1.0] }
vy: { range: [-0.1, 0.1] }
wz: { range: [-1.0, 1.0] }
```
`tasks/zero/config/controller.yaml`:
```yaml
select_code: 1
default: true
```

- [ ] **Step 4: Run → PASS**, then **Step 5: Commit**
```bash
git add core/include/phybot/controller_config.h core/src/controller_config.cpp \
        tasks/walk/config/controller.yaml tasks/zero/config/controller.yaml \
        core/test/test_controller_config.cpp CMakeLists.txt
git commit -m "feat(controller): per-task controller.yaml (select_code/default/ranges) + loader"
```

---

### Task 4: Config policy_path + WalkTask self-registration

**Files:** Modify `tasks/walk/walk_config.h`/`.cpp` (add `policy_path`, `ResolveRoot`),
`tasks/walk/config/walk.yaml` (`policy_path`), `tasks/walk/walk_task.cpp` (default→config path),
`CMakeLists.txt` (`PHYBOT_ROOT` on walk targets). Test `tasks/walk/test/test_walk_config_policy.cpp`.

- [ ] **Step 1: Failing test** — `tasks/walk/test/test_walk_config_policy.cpp`:
```cpp
#include "walk/walk_config.h"
#include <cstdio>
#include <string>
int main(){ auto c=phybot::walk_config();
  if(c.policy_path.empty()||c.policy_path[0]!='/'||c.policy_path.find("cpg_policy.pt")==std::string::npos){
    std::printf("FAIL %s\n",c.policy_path.c_str());return 1;} std::printf("OK %s\n",c.policy_path.c_str());return 0; }
```

- [ ] **Step 2: Add target + `PHYBOT_ROOT` def, run → FAIL** (mirror `test_walk_numeric`'s defs +
`PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"`; `add_test(NAME walk_config_policy ...)`).

- [ ] **Step 3: Implement** — `walk_config.h` add `std::string policy_path;` + `std::string ResolveRoot(const std::string&);`
`walk_config.cpp`:
```cpp
#ifndef PHYBOT_ROOT
#error "PHYBOT_ROOT must be defined"
#endif
std::string phybot::ResolveRoot(const std::string& p){ return (!p.empty()&&p[0]=='/')?p:std::string(PHYBOT_ROOT)+"/"+p; }
// inside walk_config(): cfg.policy_path = ResolveRoot(walk["policy_path"].as<std::string>());
```
`tasks/walk/config/walk.yaml` add `policy_path: tasks/walk/test/golden/cpg_policy.pt`.
`walk_task.cpp` ctor: `std::string path = model_path.empty() ? cfg_.policy_path : model_path;` (use `path` for the runner).
Add `PHYBOT_ROOT` to `walk_task` + config-test targets' compile-defs.

- [ ] **Step 4: Run → PASS + verify golden bit-exact**
Run: `cmake --build build -j2 && (cd build && ctest -R 'walk_config_policy|walk_e2e|obs_alignment' --output-on-failure)` → PASS, `walk_e2e` 0.000e+00.

- [ ] **Step 5: Commit**
```bash
git add tasks/walk/walk_config.h tasks/walk/walk_config.cpp tasks/walk/config/walk.yaml \
        tasks/walk/walk_task.cpp tasks/walk/test/test_walk_config_policy.cpp CMakeLists.txt
git commit -m "feat(walk): policy_path from config; WalkTask() self-registers (name-only)"
```

---

### Task 5: phybot_tasks lib whole-archived + link guard

**Files:** Modify `CMakeLists.txt`; Test `core/test/test_task_registry_link.cpp`.

- [ ] **Step 1: Failing test** — `core/test/test_task_registry_link.cpp`:
```cpp
#include "phybot/task_registry.h"
#include <cstring>
#include <cstdio>
int main(){ auto& r=phybot::TaskRegistry::instance(); bool z=false,w=false;
  for(int i=0;i<r.size();++i){ if(!std::strcmp(r.name(i),"zero"))z=true; if(!std::strcmp(r.name(i),"walk"))w=true; }
  std::printf("size=%d zero=%d walk=%d\n",r.size(),z,w); return (z&&w)?0:1; }
```

- [ ] **Step 2: phybot_tasks lib + whole-archive expr + test target**
```cmake
add_library(phybot_tasks STATIC tasks/walk/walk_task.cpp tasks/zero/zero_task.cpp
    tasks/walk/walk_config.cpp tasks/zero/zero_config.cpp)
target_include_directories(phybot_tasks PUBLIC core/include tasks)
target_compile_definitions(phybot_tasks PRIVATE
    CONFIG_DIR="${CONFIG_DIR_PATH}" WALK_CONFIG_DIR="${WALK_CONFIG_DIR_PATH}"
    ZERO_CONFIG_DIR="${ZERO_CONFIG_DIR_PATH}" PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(phybot_tasks PUBLIC policy_runner "${YAML_CPP_LIB}")
set_property(TARGET phybot_tasks PROPERTY CXX_STANDARD 17)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  set(PHYBOT_TASKS_WHOLE "$<LINK_LIBRARY:WHOLE_ARCHIVE,phybot_tasks>")
else()
  set(PHYBOT_TASKS_WHOLE -Wl,--whole-archive phybot_tasks -Wl,--no-whole-archive)
endif()
add_executable(test_task_registry_link core/test/test_task_registry_link.cpp)
target_link_libraries(test_task_registry_link PRIVATE ${PHYBOT_TASKS_WHOLE})
add_test(NAME task_registry_link COMMAND test_task_registry_link)
```
Run: build the target → PASS (`zero=1 walk=1`). (Plain `phybot_tasks` without whole-archive FAILS.)

- [ ] **Step 3: Commit**
```bash
git add core/test/test_task_registry_link.cpp CMakeLists.txt
git commit -m "feat(build): phybot_tasks whole-archived so REGISTER_TASK registrars survive"
```

---

### Task 6: TaskRouter (route + default + shape)

**Files:** Create `core/include/phybot/task_router.h`, `core/src/task_router.cpp`; Test
`core/test/test_task_router.cpp`.

**Interfaces (produced):**
```cpp
class TaskRouter {
 public:
  void addTask(TaskCode code, const ControllerConfig& cc);   // dup code / bad range → throw; tracks default
  TaskCode route(TaskCode code) const;   // registered → itself; unknown/kNoTask → defaultCode()
  TaskCode defaultCode() const;
  void shape(TaskCode active, const DriveCmd& cmd, DataPackage& io) const;
};
```

- [ ] **Step 1: Failing test** — `core/test/test_task_router.cpp`:
```cpp
#include "phybot/task_router.h"
#include "phybot/input_protocol.h"
#include "phybot/data_package.h"
#include <cmath>
#include <cstdio>
static bool nr(double a,double b){return std::fabs(a-b)<1e-9;}
int main(){ using namespace phybot; TaskRouter r;
  ControllerConfig z; z.select_code=1; z.is_default=true;
  ControllerConfig w; w.select_code=2; w.vx={-0.4,1.0}; w.wz={-1.0,1.0};
  r.addTask(1,z); r.addTask(2,w);
  if(r.route(2)!=2||r.route(1)!=1||r.route(9)!=1||r.defaultCode()!=1){std::printf("FAIL route\n");return 1;}
  DriveCmd c; c.model=2; c.vx=1.0f; c.wz=-1.0f; DataPackage io; r.shape(2,c,io);
  if(!nr(io.js_vx_desire,1.0)||!nr(io.js_OmegaZ_desire,-1.0)||io.requested_task!=2){std::printf("FAIL shape\n");return 1;}
  bool t=false; try{ControllerConfig d;d.select_code=2;r.addTask(2,d);}catch(...){t=true;} if(!t){std::printf("FAIL dup\n");return 1;}
  std::printf("OK\n");return 0; }
```

- [ ] **Step 2: Add target (include lcm headers for DriveCmd), run → FAIL**
```cmake
add_executable(test_task_router core/test/test_task_router.cpp core/src/task_router.cpp)
target_include_directories(test_task_router PRIVATE core/include core/test "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcm/include")
add_test(NAME task_router COMMAND test_task_router)
```

- [ ] **Step 3: Implement** — `task_router.h`:
```cpp
#pragma once
#include <array>
#include "phybot/task_code.h"
#include "phybot/controller_config.h"
namespace phybot { struct DriveCmd; struct DataPackage;
class TaskRouter {
 public:
  void addTask(TaskCode code, const ControllerConfig& cc);
  TaskCode route(TaskCode code) const;
  TaskCode defaultCode() const { return default_; }
  void shape(TaskCode active, const DriveCmd& cmd, DataPackage& io) const;
 private:
  std::array<bool, kMaxTaskCode> has_{};
  std::array<ControllerConfig, kMaxTaskCode> cfg_{};
  TaskCode default_{kNoTask};
};
}
```
`task_router.cpp`:
```cpp
#include "phybot/task_router.h"
#include "phybot/input_protocol.h"
#include "phybot/data_package.h"
#include <stdexcept>
namespace phybot {
void TaskRouter::addTask(TaskCode code, const ControllerConfig& cc){
  if(code<=0||code>=kMaxTaskCode) throw std::runtime_error("select_code out of range");
  if(has_[code]) throw std::runtime_error("duplicate select_code");
  has_[code]=true; cfg_[code]=cc;
  if(cc.is_default){ if(default_!=kNoTask) throw std::runtime_error("more than one default task"); default_=code; }
}
TaskCode TaskRouter::route(TaskCode code) const {
  return (code>0&&code<kMaxTaskCode&&has_[code])?code:default_;
}
void TaskRouter::shape(TaskCode active, const DriveCmd& cmd, DataPackage& io) const {
  const ControllerConfig& c=cfg_[active];
  io.requested_task=active;
  io.js_vx_desire=c.vx.apply(cmd.vx); io.js_vy_desire=c.vy.apply(cmd.vy); io.js_OmegaZ_desire=c.wz.apply(cmd.wz);
}
}
```

- [ ] **Step 4: Run → PASS**, **Step 5: Commit**
```bash
git add core/include/phybot/task_router.h core/src/task_router.cpp core/test/test_task_router.cpp CMakeLists.txt
git commit -m "feat(router): TaskRouter (code identity, default fallback, axis shaping)"
```

---

### Task 7: SimHarness (glfw-free) + retarget teleop_smoke

**Files:** Create `backends/mujoco/sim_harness.h`/`.cpp`; Modify `core/test/test_teleop_smoke.cpp`
(→ moved? it's generic — keep in `core/test/`), `CMakeLists.txt`.

**Interfaces (produced):** see spec §7. Header at `backends/mujoco/sim_harness.h`, included as `"sim_harness.h"`.

- [ ] **Step 1: Rewrite test_teleop_smoke.cpp on SimHarness**
```cpp
#include "sim_harness.h"
#include "phybot/input_protocol.h"
#include <cstdio>
int main(){ phybot::SimHarness h({XML_PATH});
  phybot::DriveCmd zero; zero.model=1; for(int i=0;i<200;++i) h.tick(zero);
  double hz=h.backend().baseHeight();
  phybot::DriveCmd walk; walk.model=2; walk.vx=0.6f; for(int i=0;i<1500;++i) h.tick(walk);
  double hw=h.backend().baseHeight();
  bool ok=hz>0.4&&hw>0.4; std::printf("h_zero=%.3f h_walk=%.3f -> %s\n",hz,hw,ok?"OK":"FALL"); return ok?0:1; }
```

- [ ] **Step 2: Implement SimHarness** — `sim_harness.h`:
```cpp
#pragma once
#include <functional>
#include <string>
#include "mujoco_backend.h"
#include "phybot/task_manager.h"
#include "phybot/task_router.h"
#include "phybot/data_package.h"
namespace phybot { struct DriveCmd;
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
  MujocoBackend backend_; TaskManager manager_; TaskRouter router_; DataPackage io_{};
};
}
```
`sim_harness.cpp`:
```cpp
#include "sim_harness.h"
#include "phybot/task_registry.h"
#include "phybot/controller_config.h"
#include "phybot/input_protocol.h"
#include <ctime>
namespace phybot {
SimHarness::SimHarness(const Options& opt) : backend_(opt.xml_path) {
  io_.control_period=0.002;
  auto& reg=TaskRegistry::instance();
  for(int i=0;i<reg.size();++i){ ControllerConfig cc=load_controller_config(reg.name(i));
    manager_.add(reg.make(i), cc.select_code); router_.addTask(cc.select_code, cc); }
  manager_.start(router_.defaultCode(), io_);
}
void SimHarness::tick(const DriveCmd& cmd){
  TaskCode code=router_.route(cmd.model); router_.shape(code, cmd, io_);
  backend_.readToPackage(io_); manager_.tick(io_); backend_.writeFromPackage(io_); backend_.step();
}
void SimHarness::runHeadless(CommandSource src, double seconds, std::function<bool()> stop){
  const timespec period{0,2*1000*1000}; long total=seconds>0?(long)(seconds/0.002):-1;
  for(long t=0; total<0||t<total; ++t){ if(stop&&stop())break; tick(src()); nanosleep(&period,nullptr); }
}
}
```

- [ ] **Step 3: CMake — sim_harness lib + retarget teleop_smoke**
```cmake
add_library(sim_harness STATIC backends/mujoco/sim_harness.cpp core/src/task_router.cpp core/src/controller_config.cpp)
target_include_directories(sim_harness PUBLIC core/include backends/mujoco tasks
    PRIVATE "${MUJOCO_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcm/include")
target_compile_definitions(sim_harness PRIVATE PHYBOT_TASKS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tasks" PHYBOT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(sim_harness PUBLIC mujoco_backend ${PHYBOT_TASKS_WHOLE} "${YAML_CPP_LIB}")
set_property(TARGET sim_harness PROPERTY CXX_STANDARD 17)
# retarget test_teleop_smoke: link sim_harness only (drop walk_task/zero_task.cpp), keep XML_PATH/POLICY_PATH defs
```

- [ ] **Step 4: Run → PASS**, **Step 5: Commit**
```bash
git add backends/mujoco/sim_harness.h backends/mujoco/sim_harness.cpp core/test/test_teleop_smoke.cpp CMakeLists.txt
git commit -m "feat(harness): glfw-free SimHarness (registry↔controller.yaml pairing); retarget teleop_smoke"
```

---

### Task 8: GUI runner (sim_harness_gui) — apps only

**Files:** Create `backends/mujoco/sim_harness_gui.h`/`.cpp`; Modify `CMakeLists.txt`.

- [ ] **Step 1: Implement** — `sim_harness_gui.h`:
```cpp
#pragma once
#include "sim_harness.h"
namespace phybot { void RunGui(SimHarness& h, SimHarness::CommandSource src, const char* title); }
```
`sim_harness_gui.cpp`:
```cpp
#include "sim_harness_gui.h"
#include "mujoco_viewer.h"
#include <thread>
#include <mutex>
#include <ctime>
namespace phybot {
void RunGui(SimHarness& h, SimHarness::CommandSource src, const char* title){
  MujocoViewer viewer(title);
  std::thread physics([&]{ viewer.loadModel(h.backend().model(), h.backend().data());
    const timespec dt{0,2*1000*1000};
    while(!viewer.exitRequested()){ { std::lock_guard<std::recursive_mutex> lk(viewer.mutex());
      if(viewer.running()){ h.tick(src()); viewer.trackCamera(); } else { src(); } }
      nanosleep(&dt,nullptr); } });
  viewer.renderLoop(); physics.join();
}
}
```

- [ ] **Step 2: CMake**
```cmake
add_library(sim_harness_gui STATIC backends/mujoco/sim_harness_gui.cpp)
target_include_directories(sim_harness_gui PUBLIC backends/mujoco PRIVATE "${MUJOCO_INCLUDE_DIR}")
target_link_libraries(sim_harness_gui PUBLIC sim_harness mujoco_gui)
set_property(TARGET sim_harness_gui PROPERTY CXX_STANDARD 17)
```
Run: `cmake -S . -B build >/dev/null && cmake --build build --target sim_harness_gui -j2` → Built.

- [ ] **Step 3: Commit**
```bash
git add backends/mujoco/sim_harness_gui.h backends/mujoco/sim_harness_gui.cpp CMakeLists.txt
git commit -m "feat(harness): RunGui runner (physics thread + MujocoViewer), apps-only link"
```

---

### Task 9: Publisher — normalized axes + buttons.yaml

**Files:** Create `publisher/config/buttons.yaml`, `publisher/include/phybot/buttons_config.h`,
`publisher/src/buttons_config.cpp`; Modify `publisher/include/phybot/input_mapping.h`,
`publisher/test/test_input_mapping.cpp`, `CMakeLists.txt`.

- [ ] **Step 1: Failing test** — update `test_input_mapping.cpp`: left stick full forward → `d.vx==1.0f`
(not `0.6`); B press → `d.model==2` from buttons.yaml.

- [ ] **Step 2: Implement** — `buttons.yaml`:
```yaml
Y: 1
B: 2
# X: 3   # walk_amp (uncomment when added)
```
`buttons_config.h`:
```cpp
#pragma once
#include <map>
#include <string>
namespace phybot { struct ButtonsConfig { std::map<std::string,int> code;
  int code_for(const std::string& b,int fb) const { auto it=code.find(b); return it==code.end()?fb:it->second; } };
  ButtonsConfig load_buttons_config(); }
```
`buttons_config.cpp`:
```cpp
#include "phybot/buttons_config.h"
#include <yaml-cpp/yaml.h>
#ifndef PHYBOT_ROOT
#error "PHYBOT_ROOT must be defined"
#endif
namespace phybot { ButtonsConfig load_buttons_config(){ ButtonsConfig c;
  auto y=YAML::LoadFile(std::string(PHYBOT_ROOT)+"/publisher/config/buttons.yaml");
  for(auto it=y.begin();it!=y.end();++it) c.code[it->first.as<std::string>()]=it->second.as<int>(); return c; } }
```
`input_mapping.h`: change scales to normalized — `vx=-dz(left_y)`, `vy=dz(left_x)`, `wz=-dz(right_x)`
(drop `*0.6/*0.1/*1.0`); B rising → `model=buttons.code_for("B",2)`, Y → `buttons.code_for("Y",1)`.
Load `ButtonsConfig` once into `MappingState`.

- [ ] **Step 3: Run input_mapping + input_protocol (golden) → PASS**, **Step 4: Commit**
```bash
git add publisher/config/buttons.yaml publisher/include/phybot/buttons_config.h \
        publisher/src/buttons_config.cpp publisher/include/phybot/input_mapping.h \
        publisher/test/test_input_mapping.cpp CMakeLists.txt
git commit -m "feat(publisher): normalized axes + buttons.yaml (button→select_code)"
```

---

### Task 10: Thin apps + remove teleop_loop.h

**Files:** Rewrite `apps/sim_walk.cpp`, `apps/sim_teleop.cpp`; delete `backends/mujoco/teleop_loop.h`;
Modify `CMakeLists.txt`.

- [ ] **Step 1: sim_walk.cpp** (harness shell; fixed WALK source; `--gui`→RunGui) — see spec §7; key body:
```cpp
phybot::SimHarness h({xml});
auto src=[&]{ phybot::DriveCmd c; c.model=2; c.vx=(float)vx; return c; };  // WALK, normalized vx
if(gui) phybot::RunGui(h,src,"sim_walk"); else { h.runHeadless(src,sec);
  std::printf("[sim_walk] height=%.4f x=%.4f\n",h.backend().baseHeight(),h.backend().baseX()); }
```

- [ ] **Step 2: sim_teleop.cpp** (LCM source; `--gui`→RunGui):
```cpp
phybot::SimHarness h({xml});
auto src=[&]{ lcm.handleTimeout(0); return holder.latest; };
if(gui) phybot::RunGui(h,src,"sim_teleop"); else h.runHeadless(src,0);
```
(Keep `AnnounceLcmInit`, `lcm.good()`, `Holder`/`ip_decode` subscribe, `--gui` arg parse, empty-arg fallback.)

- [ ] **Step 3: Delete teleop_loop.h + wire CMake**
```bash
git rm backends/mujoco/teleop_loop.h
```
```cmake
target_link_libraries(sim_walk PRIVATE sim_harness sim_harness_gui)
target_link_libraries(sim_teleop PRIVATE sim_harness sim_harness_gui phybot_lcm)
# both: target_compile_definitions(... XML_PATH="${PHYBOT_XML_PATH}"); drop old walk_task/mujoco_gui direct links
```
Confirm nothing else includes `teleop_loop.h`.

- [ ] **Step 4: Full build + verify**
Run: `cmake -S . -B build >/dev/null && cmake --build build -j2 && (cd build && ctest --output-on-failure)`
Expected: all PASS (originals + `controller_config`, `task_registry_link`, `task_router`, `walk_config_policy`).
Run: `./build/sim_walk` → `height≈0.7`. `timeout 4 stdbuf -oL ./build/sim_walk --gui >/tmp/g.log 2>&1; grep -q sim_walk /tmp/g.log && echo GUI-STARTED`.

- [ ] **Step 5: Commit**
```bash
git add apps/sim_walk.cpp apps/sim_teleop.cpp CMakeLists.txt
git commit -m "refactor(apps): thin SimHarness shells; drop teleop_loop.h"
```

---

## Done-when
- `core/` has no `walk_*`/`zero_*` and no `task_id.h`; each task is self-contained in its folder.
- Adding `walk_amp` = folder (`walk_amp_task.*` + `REGISTER_TASK("walk_amp",…)` + `walk_amp.yaml` +
  `controller.yaml` `select_code:3`) + one `phybot_tasks` source line + `X: 3` in `buttons.yaml`.
- `ctest` green (incl. 4 new tests); `walk_e2e`/`obs_alignment` bit-exact; `sim_walk` headless
  height≈0.7; GUI apps link `mujoco_gui`, headless tests do not.
