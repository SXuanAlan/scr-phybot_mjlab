# Core Task FSM (A0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the dependency-free core task state machine (`Task` / `TaskManager` / `TaskRegistry` / `TaskId`) for the new deploy in `deploy/bdml_phybot_deploy_c2`, fully unit-tested without any robot, libtorch, Eigen, or LCM dependency.

**Architecture:** A header-only FSM living in `core/include/phybot/`. The unit of behavior is a `Task` (lifecycle `enter/run/exit`, plus `checkChange` self-transition and `canExit` interlock). A `TaskManager` owns all tasks in a `std::array` indexed by `enum class TaskId` and, each `tick()`, switches to the requested/auto task (interlocked by `canExit`) then runs the active one. Tasks self-register into a `TaskRegistry` via the `REGISTER_TASK` macro. The blackboard is a minimal `DataPackage` carrying only `requested_task`/`current_task` for now; later plans (A1+) grow it with sensor/command/action fields.

**Tech Stack:** C++17, CMake ≥ 3.12, CTest. No external libraries (no GoogleTest, Eigen, libtorch, LCM) — tests use a tiny in-repo `CHECK` macro and a test executable whose non-zero exit means failure.

## Global Constraints

- Target directory: `deploy/bdml_phybot_deploy_c2/` (currently empty). All paths below are relative to it unless stated otherwise.
- C++ standard: C++17. CMake minimum: 3.12 (matches official `deploy/phybot_deploy_c2/CMakeLists.txt`).
- Namespace: `phybot` for all new core types.
- No real-robot involvement in this plan (Group A0). No libtorch/Eigen/LCM includes in any A0 file.
- `enum class TaskId` is the single central list of tasks; `TaskId::COUNT` must always be the last enumerator (it sizes the `std::array`).
- `TaskManager`/`Task` touch only `DataPackage::requested_task` and `DataPackage::current_task`; they must not assume any other field exists.
- Commit after every task with a `feat:`/`test:` message.

---

## File Structure

- `core/include/phybot/task_id.h` — `enum class TaskId` (central task list).
- `core/include/phybot/data_package.h` — minimal `struct DataPackage` (blackboard; two `TaskId` fields for now).
- `core/include/phybot/task.h` — abstract base `class Task`.
- `core/include/phybot/task_manager.h` — `class TaskManager` (owns tasks, `tick()` switching).
- `core/include/phybot/task_registry.h` — `class TaskRegistry` singleton + `REGISTER_TASK` macro + `loadInto`.
- `core/test/check.h` — tiny assert macro `CHECK(cond)` used by tests.
- `core/test/test_task_manager.cpp` — all A0 unit tests (one `main`, returns non-zero on failure).
- `CMakeLists.txt` (target-dir root) — builds the test and registers it with CTest.

---

## Task 1: TaskId enum + minimal DataPackage + build/test scaffold

**Files:**
- Create: `deploy/bdml_phybot_deploy_c2/core/include/phybot/task_id.h`
- Create: `deploy/bdml_phybot_deploy_c2/core/include/phybot/data_package.h`
- Create: `deploy/bdml_phybot_deploy_c2/core/test/check.h`
- Create: `deploy/bdml_phybot_deploy_c2/core/test/test_task_manager.cpp`
- Create: `deploy/bdml_phybot_deploy_c2/CMakeLists.txt`

**Interfaces:**
- Produces: `enum class phybot::TaskId { ZERO, WALK_AMP, MIMIC, COUNT };` — `COUNT` is always last.
- Produces: `struct phybot::DataPackage { TaskId requested_task{TaskId::ZERO}; TaskId current_task{TaskId::ZERO}; };`
- Produces: `CHECK(cond)` macro (prints `FAIL: <expr> (file:line)` and sets a global failure flag).
- Produces: CTest test named `core_task_manager`.

- [ ] **Step 1: Write the failing test**

Create `core/test/check.h`:
```cpp
#pragma once
#include <cstdio>
inline int g_check_failures = 0;
#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      ++g_check_failures;                                               \
      std::printf("FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__);     \
    }                                                                   \
  } while (0)
```

Create `core/test/test_task_manager.cpp`:
```cpp
#include "phybot/task_id.h"
#include "phybot/data_package.h"
#include "check.h"

static void test_datapackage_defaults() {
  phybot::DataPackage io;
  CHECK(io.requested_task == phybot::TaskId::ZERO);
  CHECK(io.current_task == phybot::TaskId::ZERO);
}

int main() {
  test_datapackage_defaults();
  if (g_check_failures == 0) std::printf("ALL TESTS PASSED\n");
  return g_check_failures == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cmake -S deploy/bdml_phybot_deploy_c2 -B /tmp/a0_build && cmake --build /tmp/a0_build
```
Expected: FAIL — configure/compile error because `CMakeLists.txt`, `task_id.h`, and `data_package.h` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `core/include/phybot/task_id.h`:
```cpp
#pragma once
namespace phybot {
// 唯一中心任务清单。新增 task 在 COUNT 之前加一行;COUNT 必须始终最后。
enum class TaskId { ZERO, WALK_AMP, MIMIC, COUNT };
}  // namespace phybot
```

Create `core/include/phybot/data_package.h`:
```cpp
#pragma once
#include "phybot/task_id.h"
namespace phybot {
// 最小黑板。A1+ 阶段在此扩展传感器/命令/动作字段。
struct DataPackage {
  TaskId requested_task{TaskId::ZERO};  // 输入侧写: 想切到哪个 task
  TaskId current_task{TaskId::ZERO};    // TaskManager 写: 当前激活 task
};
}  // namespace phybot
```

Create `CMakeLists.txt` (at `deploy/bdml_phybot_deploy_c2/`):
```cmake
cmake_minimum_required(VERSION 3.12)
project(bdml_phybot_deploy_c2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_executable(test_task_manager core/test/test_task_manager.cpp)
target_include_directories(test_task_manager PRIVATE core/include core/test)

add_test(NAME core_task_manager COMMAND test_task_manager)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cmake -S deploy/bdml_phybot_deploy_c2 -B /tmp/a0_build && cmake --build /tmp/a0_build && ctest --test-dir /tmp/a0_build --output-on-failure
```
Expected: PASS — `ALL TESTS PASSED` and `100% tests passed`.

- [ ] **Step 5: Commit**

```bash
git add deploy/bdml_phybot_deploy_c2/core/include/phybot/task_id.h \
        deploy/bdml_phybot_deploy_c2/core/include/phybot/data_package.h \
        deploy/bdml_phybot_deploy_c2/core/test/check.h \
        deploy/bdml_phybot_deploy_c2/core/test/test_task_manager.cpp \
        deploy/bdml_phybot_deploy_c2/CMakeLists.txt
git commit -m "feat(core): TaskId enum + minimal DataPackage + test scaffold"
```

---

## Task 2: Task abstract base class

**Files:**
- Create: `deploy/bdml_phybot_deploy_c2/core/include/phybot/task.h`
- Modify: `deploy/bdml_phybot_deploy_c2/core/test/test_task_manager.cpp` (add a dummy task + test)

**Interfaces:**
- Consumes: `phybot::TaskId`, `phybot::DataPackage`.
- Produces: `class phybot::Task` with:
  - `Task(TaskId id, std::string name)`
  - `TaskId id() const`
  - `const std::string& name() const`
  - `virtual void enter(DataPackage&)` (default no-op)
  - `virtual void run(DataPackage&) = 0` (pure virtual)
  - `virtual void exit(DataPackage&)` (default no-op)
  - `virtual TaskId checkChange(DataPackage&)` (default returns own id = no change)
  - `virtual bool canExit(DataPackage&)` (default true)

- [ ] **Step 1: Write the failing test**

Add to `core/test/test_task_manager.cpp` (above `main`):
```cpp
#include "phybot/task.h"

namespace {
// 哑 task: 记录各生命周期被调用次数。
struct DummyTask : phybot::Task {
  int entered = 0, ran = 0, exited = 0;
  using phybot::Task::Task;
  void enter(phybot::DataPackage&) override { ++entered; }
  void run(phybot::DataPackage&) override { ++ran; }
  void exit(phybot::DataPackage&) override { ++exited; }
};
}  // namespace

static void test_task_basics() {
  DummyTask t{phybot::TaskId::WALK_AMP, "walk_amp"};
  phybot::DataPackage io;
  CHECK(t.id() == phybot::TaskId::WALK_AMP);
  CHECK(t.name() == "walk_amp");
  CHECK(t.checkChange(io) == phybot::TaskId::WALK_AMP);  // 默认: 返回自身=不变
  CHECK(t.canExit(io) == true);                          // 默认: 允许退出
  t.enter(io); t.run(io); t.exit(io);
  CHECK(t.entered == 1 && t.ran == 1 && t.exited == 1);
}
```

Add `test_task_basics();` as the first line inside `main()` (before the existing call).

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cmake --build /tmp/a0_build 2>&1 | head -20
```
Expected: FAIL — compile error: `phybot/task.h` not found / `phybot::Task` undefined.

- [ ] **Step 3: Write minimal implementation**

Create `core/include/phybot/task.h`:
```cpp
#pragma once
#include <string>
#include <utility>
#include "phybot/task_id.h"

namespace phybot {
struct DataPackage;  // 前置声明: Task 接口只按引用使用黑板

class Task {
 public:
  Task(TaskId id, std::string name) : id_(id), name_(std::move(name)) {}
  virtual ~Task() = default;

  TaskId id() const { return id_; }
  const std::string& name() const { return name_; }

  virtual void enter(DataPackage&) {}                 // 进入(复位/加载)
  virtual void run(DataPackage&) = 0;                  // 每拍执行
  virtual void exit(DataPackage&) {}                   // 退出(清零)
  virtual TaskId checkChange(DataPackage&) { return id_; }  // 自跳转, 返回自身=不变
  virtual bool canExit(DataPackage&) { return true; }       // 互锁

 protected:
  TaskId id_;
  std::string name_;
};
}  // namespace phybot
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cmake --build /tmp/a0_build && ctest --test-dir /tmp/a0_build --output-on-failure
```
Expected: PASS — `ALL TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add deploy/bdml_phybot_deploy_c2/core/include/phybot/task.h \
        deploy/bdml_phybot_deploy_c2/core/test/test_task_manager.cpp
git commit -m "feat(core): add Task abstract base class"
```

---

## Task 3: TaskManager switching + tick logic

**Files:**
- Create: `deploy/bdml_phybot_deploy_c2/core/include/phybot/task_manager.h`
- Modify: `deploy/bdml_phybot_deploy_c2/core/test/test_task_manager.cpp` (add switching tests)

**Interfaces:**
- Consumes: `phybot::Task`, `phybot::TaskId`, `phybot::DataPackage`.
- Produces: `class phybot::TaskManager` with:
  - `void add(std::unique_ptr<Task> t)` — stores at index `(size_t)t->id()`.
  - `void start(TaskId id, DataPackage& io)` — sets active, writes `io.current_task`, calls `enter`.
  - `TaskId current() const`.
  - `void tick(DataPackage& io)` — order: `checkChange` (auto) else `io.requested_task` (external); if different, target exists, and `canExit()` true → `exit→swap→enter`; then write `io.current_task` and call active `run`.

- [ ] **Step 1: Write the failing test**

Add to `core/test/test_task_manager.cpp` (above `main`):
```cpp
#include "phybot/task_manager.h"
#include <memory>

namespace {
// 可配置哑 task: 控制 checkChange / canExit。
struct CfgTask : phybot::Task {
  using phybot::Task::Task;
  phybot::TaskId force_change;     // 设为非自身则强制自跳转
  bool block_exit = false;
  int ran = 0, entered = 0, exited = 0;
  CfgTask(phybot::TaskId id, std::string n)
      : phybot::Task(id, std::move(n)), force_change(id) {}
  void enter(phybot::DataPackage&) override { ++entered; }
  void run(phybot::DataPackage&) override { ++ran; }
  void exit(phybot::DataPackage&) override { ++exited; }
  phybot::TaskId checkChange(phybot::DataPackage&) override { return force_change; }
  bool canExit(phybot::DataPackage&) override { return !block_exit; }
};

std::unique_ptr<CfgTask> make(phybot::TaskId id, const char* n,
                              CfgTask** out) {
  auto t = std::make_unique<CfgTask>(id, n);
  *out = t.get();
  return t;
}
}  // namespace

static void test_manager_runs_active_each_tick() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero;
  m.add(make(TaskId::ZERO, "zero", &zero));
  DataPackage io;
  m.start(TaskId::ZERO, io);
  CHECK(zero->entered == 1);
  CHECK(io.current_task == TaskId::ZERO);
  m.tick(io);
  m.tick(io);
  CHECK(zero->ran == 2);          // 每拍都跑当前
  CHECK(m.current() == TaskId::ZERO);
}

static void test_manager_switches_on_request() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero, *walk;
  m.add(make(TaskId::ZERO, "zero", &zero));
  m.add(make(TaskId::WALK_AMP, "walk_amp", &walk));
  DataPackage io;
  m.start(TaskId::ZERO, io);
  io.requested_task = TaskId::WALK_AMP;
  m.tick(io);
  CHECK(zero->exited == 1);        // 退出旧
  CHECK(walk->entered == 1);       // 进入新
  CHECK(walk->ran == 1);           // 切换当拍跑新 task
  CHECK(zero->ran == 0);           // 切换当拍不跑旧 task
  CHECK(io.current_task == TaskId::WALK_AMP);
  CHECK(m.current() == TaskId::WALK_AMP);
}

static void test_manager_no_switch_when_same() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero;
  m.add(make(TaskId::ZERO, "zero", &zero));
  DataPackage io;
  m.start(TaskId::ZERO, io);
  io.requested_task = TaskId::ZERO;   // 与当前相同
  m.tick(io);
  CHECK(zero->exited == 0);           // 不切换
  CHECK(zero->ran == 1);
}

static void test_manager_canExit_blocks_switch() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero, *walk;
  m.add(make(TaskId::ZERO, "zero", &zero));
  m.add(make(TaskId::WALK_AMP, "walk_amp", &walk));
  DataPackage io;
  m.start(TaskId::ZERO, io);
  zero->block_exit = true;             // 互锁: 拒绝离开
  io.requested_task = TaskId::WALK_AMP;
  m.tick(io);
  CHECK(zero->exited == 0);
  CHECK(walk->entered == 0);
  CHECK(m.current() == TaskId::ZERO);  // 维持当前
}

static void test_manager_checkChange_auto_transition() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero, *walk;
  m.add(make(TaskId::ZERO, "zero", &zero));
  m.add(make(TaskId::WALK_AMP, "walk_amp", &walk));
  DataPackage io;
  m.start(TaskId::ZERO, io);
  zero->force_change = TaskId::WALK_AMP;   // 状态自决跳转, 优先于 requested
  io.requested_task = TaskId::ZERO;
  m.tick(io);
  CHECK(m.current() == TaskId::WALK_AMP);
  CHECK(walk->entered == 1);
}

static void test_manager_ignores_unregistered_target() {
  using namespace phybot;
  TaskManager m;
  CfgTask *zero;
  m.add(make(TaskId::ZERO, "zero", &zero));   // MIMIC 未注册
  DataPackage io;
  m.start(TaskId::ZERO, io);
  io.requested_task = TaskId::MIMIC;          // 目标不存在 → 安全忽略
  m.tick(io);
  CHECK(zero->exited == 0);
  CHECK(m.current() == TaskId::ZERO);
}
```

Add these calls inside `main()` before the success print:
```cpp
  test_manager_runs_active_each_tick();
  test_manager_switches_on_request();
  test_manager_no_switch_when_same();
  test_manager_canExit_blocks_switch();
  test_manager_checkChange_auto_transition();
  test_manager_ignores_unregistered_target();
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cmake --build /tmp/a0_build 2>&1 | head -20
```
Expected: FAIL — compile error: `phybot/task_manager.h` not found.

- [ ] **Step 3: Write minimal implementation**

Create `core/include/phybot/task_manager.h`:
```cpp
#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include "phybot/task_id.h"
#include "phybot/task.h"
#include "phybot/data_package.h"

namespace phybot {
class TaskManager {
 public:
  void add(std::unique_ptr<Task> t) { tasks_[(std::size_t)t->id()] = std::move(t); }

  void start(TaskId id, DataPackage& io) {
    current_ = tasks_[(std::size_t)id].get();
    io.current_task = id;
    current_->enter(io);
  }

  TaskId current() const { return current_ ? current_->id() : TaskId::ZERO; }

  void tick(DataPackage& io) {
    TaskId next = current_->checkChange(io);            // ① 自决(返回自身=不变)
    if (next == current_->id()) next = io.requested_task;  // ② 否则外部请求
    if (next != current_->id() && tasks_[(std::size_t)next]) {
      if (current_->canExit(io)) {                      // ③ 互锁
        current_->exit(io);
        current_ = tasks_[(std::size_t)next].get();
        current_->enter(io);
      }
    }
    io.current_task = current_->id();
    current_->run(io);                                  // ④ 跑当前
  }

 private:
  std::array<std::unique_ptr<Task>, (std::size_t)TaskId::COUNT> tasks_{};
  Task* current_ = nullptr;
};
}  // namespace phybot
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cmake --build /tmp/a0_build && ctest --test-dir /tmp/a0_build --output-on-failure
```
Expected: PASS — `ALL TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add deploy/bdml_phybot_deploy_c2/core/include/phybot/task_manager.h \
        deploy/bdml_phybot_deploy_c2/core/test/test_task_manager.cpp
git commit -m "feat(core): add TaskManager switching + tick logic"
```

---

## Task 4: TaskRegistry + REGISTER_TASK + loadInto

**Files:**
- Create: `deploy/bdml_phybot_deploy_c2/core/include/phybot/task_registry.h`
- Modify: `deploy/bdml_phybot_deploy_c2/core/test/test_task_manager.cpp` (add registry tests)

**Interfaces:**
- Consumes: `phybot::Task`, `phybot::TaskId`, `phybot::TaskManager`.
- Produces: `class phybot::TaskRegistry` singleton:
  - `static TaskRegistry& instance()`
  - `void add(TaskId id, const char* name, Factory f)` where `using Factory = std::function<std::unique_ptr<Task>()>`
  - `void loadInto(TaskManager& m) const` — adds one instance of every registered task.
  - `bool has(TaskId id) const`
  - `const char* name(TaskId id) const` (nullptr if unregistered)
- Produces: macro `REGISTER_TASK(ID, NAME, CLASS)` — registers `CLASS` (default-constructible, derives `Task`) at static-init time.

- [ ] **Step 1: Write the failing test**

Add to `core/test/test_task_manager.cpp` (above `main`):
```cpp
#include "phybot/task_registry.h"

namespace {
struct RegZeroTask : phybot::Task {
  RegZeroTask() : phybot::Task(phybot::TaskId::ZERO, "zero") {}
  void run(phybot::DataPackage&) override {}
};
struct RegWalkTask : phybot::Task {
  RegWalkTask() : phybot::Task(phybot::TaskId::WALK_AMP, "walk_amp") {}
  void run(phybot::DataPackage&) override {}
};
}  // namespace
REGISTER_TASK(phybot::TaskId::ZERO, "zero", RegZeroTask)
REGISTER_TASK(phybot::TaskId::WALK_AMP, "walk_amp", RegWalkTask)

static void test_registry_self_registration() {
  using namespace phybot;
  auto& reg = TaskRegistry::instance();
  CHECK(reg.has(TaskId::ZERO));
  CHECK(reg.has(TaskId::WALK_AMP));
  CHECK(reg.has(TaskId::MIMIC) == false);
  CHECK(std::string(reg.name(TaskId::WALK_AMP)) == "walk_amp");
}

static void test_registry_loadInto() {
  using namespace phybot;
  TaskManager m;
  TaskRegistry::instance().loadInto(m);
  DataPackage io;
  m.start(TaskId::ZERO, io);
  io.requested_task = TaskId::WALK_AMP;
  m.tick(io);
  CHECK(m.current() == TaskId::WALK_AMP);   // 注册的 task 可被切换
}
```

Add inside `main()` before the success print:
```cpp
  test_registry_self_registration();
  test_registry_loadInto();
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cmake --build /tmp/a0_build 2>&1 | head -20
```
Expected: FAIL — compile error: `phybot/task_registry.h` not found / `REGISTER_TASK` undefined.

- [ ] **Step 3: Write minimal implementation**

Create `core/include/phybot/task_registry.h`:
```cpp
#pragma once
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include "phybot/task_id.h"
#include "phybot/task.h"
#include "phybot/task_manager.h"

namespace phybot {
class TaskRegistry {
 public:
  using Factory = std::function<std::unique_ptr<Task>()>;

  static TaskRegistry& instance() {
    static TaskRegistry r;
    return r;
  }

  void add(TaskId id, const char* name, Factory f) {
    factories_[(std::size_t)id] = std::move(f);
    names_[(std::size_t)id] = name;
  }

  void loadInto(TaskManager& m) const {
    for (std::size_t i = 0; i < (std::size_t)TaskId::COUNT; ++i)
      if (factories_[i]) m.add(factories_[i]());
  }

  bool has(TaskId id) const { return static_cast<bool>(factories_[(std::size_t)id]); }
  const char* name(TaskId id) const { return names_[(std::size_t)id]; }

 private:
  std::array<Factory, (std::size_t)TaskId::COUNT> factories_{};
  std::array<const char*, (std::size_t)TaskId::COUNT> names_{};
};
}  // namespace phybot

// 自注册宏: 在静态初始化期把 CLASS 工厂登记到注册表。
// 注意: 当 task 编进静态库时, 需用 CMake OBJECT 库/alwayslink 防止链接器丢弃
// (见 A1 plan 的构建任务);本 A0 测试中 task 与测试同 TU, 注册必然生效。
#define REGISTER_TASK(ID, NAME, CLASS)                                       \
  namespace {                                                                \
  struct CLASS##_Reg {                                                       \
    CLASS##_Reg() {                                                          \
      ::phybot::TaskRegistry::instance().add(                               \
          (ID), (NAME),                                                      \
          [] { return std::unique_ptr<::phybot::Task>(new CLASS()); });     \
    }                                                                        \
  } CLASS##_reg_instance;                                                    \
  }
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cmake --build /tmp/a0_build && ctest --test-dir /tmp/a0_build --output-on-failure
```
Expected: PASS — `ALL TESTS PASSED`, `100% tests passed`.

- [ ] **Step 5: Commit**

```bash
git add deploy/bdml_phybot_deploy_c2/core/include/phybot/task_registry.h \
        deploy/bdml_phybot_deploy_c2/core/test/test_task_manager.cpp
git commit -m "feat(core): add TaskRegistry + REGISTER_TASK + loadInto"
```

---

## Done criteria for A0

- `ctest --test-dir /tmp/a0_build` reports 100% passed.
- Headers `task_id.h`, `data_package.h`, `task.h`, `task_manager.h`, `task_registry.h` exist under `core/include/phybot/` and compile with `-std=c++17` and no external dependency.
- All five behaviors are covered by tests: per-tick run, switch-on-request, no-switch-when-same, `canExit` interlock, `checkChange` auto-transition, unregistered-target ignored, self-registration, `loadInto`.

## Roadmap (subsequent plans — authored when reached, each needs codebase investigation first)

- **A1** PolicyTask + PolicyRunner + HistoryBuffers + ObsBuilder + InputProtocol + mujoco backend + zero/walk_amp + dry-run + **alignment bit-exactness vs official** (requires studying official `rl_deploy_walk_amp` and `DataPackage`).
- **A2** Main-loop integration (TaskManager + InputProtocol replace server/manager), validated on mujoco backend (requires studying official `run_mini.cpp`).
- **A3** SDL2 publisher input layer + bindings.yaml + N1/N2 (debounce, enum cross-thread).
- **A4** mimic migration + _template.
- **A5** cleanup + CI/regression.
- **B1–B5** real-robot bring-up (gated on A all-green + §11.4 checklist).
