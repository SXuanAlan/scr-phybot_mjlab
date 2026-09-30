# A1-num: CPG-walk numeric core (obs / history / action) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce, dependency-free and unit-tested, the exact numeric pipeline of the official CPG walk (`deploy/phybot_deploy_c2/RL_deploy_cpg`) inside the new deploy `deploy/bdml_phybot_deploy_c2`: the 283-D observation recipe (with its quirks), the pos/vel history buffers, the ZYX→gravity function, and the action→joint-target mapping.

**Architecture:** Pure C++17 functions/structs operating on `std::array`/`std::vector` of `double` (intermediate math in `double`, observation emitted as `float` exactly like the official, which builds a `torch::kFloat` tensor). No libtorch, no Eigen, no yaml, no mujoco — those arrive in the next plan (A1-policy). The cpg constants (scales, default pose, gains) are hardcoded here as `walk_config()` from the investigated official `rl_params.yaml`; A1-policy will load them from yaml and the golden comparison will catch any mismatch.

**Tech Stack:** C++17, CMake ≥ 3.12, CTest, in-repo `CHECK` macro (from A0). Builds on the A0 core (`task_id.h`, `data_package.h`, `check.h`, `test_task_manager.cpp`, `CMakeLists.txt`).

## Global Constraints

- Target directory: `deploy/bdml_phybot_deploy_c2/`. All paths below are relative to it unless stated.
- C++17, namespace `phybot`, no external libraries in this plan (no Eigen/libtorch/yaml).
- Robot constants: `kNumDofs = 21`. Observation dim `kObsDim = 283`. Proprio block `kProprio = 73`. History frames used in obs `kHistFramesInObs = 5` (→ 5×21 = 105 each for pos and vel). Full history length `kHistLen = 20`. Decimation = 10.
- **Bit-exactness intent:** all values, scales, ordering, and the documented *quirks* of official `rl_deploy_cpg` must be reproduced verbatim. Do NOT "clean up" any quirk. The quirks (from the spec §13.1 / investigation):
  1. pos-history stores `q × dof_pos_scale` with **NO** default-pose subtraction (while the live proprio block at obs[10:31] subtracts default).
  2. obs index 9 is a binary walk flag `(|commands(0)| < 0.3) ? 0.0 : 1.0`, where `commands(0) = js_vx_desire × lin_vel_scale` and `lin_vel_scale = 1.0`.
  3. no obs clipping, no action clipping, no torque-limit clamping.
  4. history is newest-at-tail; obs reads the newest 5 frames (`tail(105)`); history starts all-zero (first inference sees zero history).
  5. `last_actions` stored in obs[52:73] is the **raw** network output (un-scaled, un-offset).
- cpg config values (from official `RL_deploy_cpg/config/rl_params.yaml`, verbatim):
  - `ang_vel_scale = 0.25`, `dof_pos_scale = 1.0`, `dof_vel_scale = 0.05`, `lin_vel_scale = 1.0`.
  - `action_scale[21]` = all `0.2`.
  - `default_dof_pos[21]` = `[-0.242,0.029,0.007,0.49,-0.295,0.0, -0.242,-0.029,-0.007,0.49,-0.295,0.0, 0.0, 0.0,0.0,0.0,-0.17, 0.0,-0.0,0.0,-0.17]`.
  - `rl_p[21]` = `[100,100,100,100,50,50, 100,100,100,100,50,50, 20, 50,50,50,50, 50,50,50,50]`.
  - `rl_d[21]` = `[5,5,5,5,5,5, 5,5,5,5,5,5, 10, 5,5,5,5, 5,5,5,5]`.
- ZYX→gravity (from official `Euler_ZYXToGravityVec`, `R = Rz(a0)·Ry(a1)·Rx(a2)`, return `-R.transpose().col(2)`; closed form, independent of yaw a0): `g = [ sin(a1), -cos(a1)·sin(a2), -cos(a1)·cos(a2) ]` with `a1 = zyx[1]`, `a2 = zyx[2]`. (A1-policy must confirm this matches the official's actual float arithmetic; if a ULP diff appears, reproduce the official matrix-multiply order instead.)
- Commit after every task.

## File Structure

- `core/include/phybot/data_package.h` — **extend** A0's struct with the cpg sensor/command/output fields.
- `core/include/phybot/robot_consts.h` — `kNumDofs`, `kObsDim`, etc.
- `core/include/phybot/walk_config.h` — `WalkConfig` struct + `walk_config()` returning the cpg values.
- `core/include/phybot/gravity.h` — `GravityFromEulerZYX`.
- `core/include/phybot/walk_obs.h` — `BuildWalkObs(...)` producing the 283 obs.
- `core/include/phybot/walk_history.h` — `WalkHistory` (pos/vel buffers).
- `core/include/phybot/walk_action.h` — `ApplyWalkAction(...)` writing joint targets to DataPackage.
- `core/test/test_walk_numeric.cpp` — new test executable for all of the above.
- `CMakeLists.txt` — add a second test executable + ctest entry.

---

## Task 1: robot_consts + extend DataPackage with cpg fields + second test exe

**Files:**
- Create: `core/include/phybot/robot_consts.h`
- Modify: `core/include/phybot/data_package.h`
- Create: `core/test/test_walk_numeric.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `namespace phybot { constexpr int kNumDofs = 21; constexpr int kObsDim = 283; constexpr int kProprio = 73; constexpr int kHistLen = 20; constexpr int kHistFramesInObs = 5; constexpr int kDecimation = 10; }`
- Produces (extends `DataPackage`): readable fields `std::array<double,3> imu_angular_vel{}; std::array<double,3> imu_zyx{}; std::array<double,kNumDofs> motor_pos{}; std::array<double,kNumDofs> motor_vel{}; double js_vx_desire{0}, js_vy_desire{0}, js_OmegaZ_desire{0};` and writable fields `std::array<double,kNumDofs> motor_Pos_desire{}, motor_Vel_desire{}, motor_Torque_desire{}, torq_desire{}, sim_P{}, sim_D{};` (all zero-initialized). Keeps existing `requested_task`/`current_task`.
- Produces: CTest test `walk_numeric`.

- [ ] **Step 1: Write the failing test**

Create `core/test/test_walk_numeric.cpp`:
```cpp
#include "phybot/robot_consts.h"
#include "phybot/data_package.h"
#include "check.h"
#include <cstdio>

static void test_datapackage_cpg_fields_zeroed() {
  phybot::DataPackage io;
  CHECK(io.motor_pos.size() == (size_t)phybot::kNumDofs);
  CHECK(io.motor_Pos_desire.size() == (size_t)phybot::kNumDofs);
  CHECK(io.imu_zyx.size() == 3);
  CHECK(io.js_vx_desire == 0.0);
  CHECK(io.motor_pos[0] == 0.0 && io.motor_pos[20] == 0.0);
  CHECK(io.sim_P[10] == 0.0);
}

int main() {
  test_datapackage_cpg_fields_zeroed();
  if (g_check_failures == 0) std::printf("ALL WALK NUMERIC TESTS PASSED\n");
  return g_check_failures == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cmake -S deploy/bdml_phybot_deploy_c2 -B /tmp/a1_build && cmake --build /tmp/a1_build
```
Expected: FAIL — `robot_consts.h` missing and `DataPackage` has no `motor_pos`; also `CMakeLists.txt` has no `walk_numeric` target yet.

- [ ] **Step 3: Write minimal implementation**

Create `core/include/phybot/robot_consts.h`:
```cpp
#pragma once
namespace phybot {
constexpr int kNumDofs        = 21;
constexpr int kObsDim         = 283;   // 73 proprio + 5*21 pos-hist + 5*21 vel-hist
constexpr int kProprio        = 73;
constexpr int kHistLen        = 20;    // full history depth
constexpr int kHistFramesInObs = 5;    // newest 5 frames enter the obs
constexpr int kDecimation     = 10;
}  // namespace phybot
```

Modify `core/include/phybot/data_package.h` — add the includes and fields (keep the existing two task fields):
```cpp
#pragma once
#include <array>
#include "phybot/task_id.h"
#include "phybot/robot_consts.h"
namespace phybot {
// 黑板。A0 起始最小, A1-num 扩展 cpg 走路所需字段。
struct DataPackage {
  TaskId requested_task{TaskId::ZERO};
  TaskId current_task{TaskId::ZERO};

  // —— 传感器 (task 读) ——
  std::array<double, 3>         imu_angular_vel{};
  std::array<double, 3>         imu_zyx{};            // ZYX euler
  std::array<double, kNumDofs>  motor_pos{};
  std::array<double, kNumDofs>  motor_vel{};
  double js_vx_desire{0.0}, js_vy_desire{0.0}, js_OmegaZ_desire{0.0};

  // —— 控制输出 (task 写) ——
  std::array<double, kNumDofs>  motor_Pos_desire{};
  std::array<double, kNumDofs>  motor_Vel_desire{};
  std::array<double, kNumDofs>  motor_Torque_desire{};
  std::array<double, kNumDofs>  torq_desire{};
  std::array<double, kNumDofs>  sim_P{};
  std::array<double, kNumDofs>  sim_D{};
};
}  // namespace phybot
```

Modify `CMakeLists.txt` — append after the existing `test_task_manager` block:
```cmake
add_executable(test_walk_numeric core/test/test_walk_numeric.cpp)
target_include_directories(test_walk_numeric PRIVATE core/include core/test)
add_test(NAME walk_numeric COMMAND test_walk_numeric)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cmake -S deploy/bdml_phybot_deploy_c2 -B /tmp/a1_build && cmake --build /tmp/a1_build && (cd /tmp/a1_build && ctest --output-on-failure)
```
Expected: PASS — `walk_numeric` passes and the existing `core_task_manager` still passes (2/2). Note: run ctest from inside the build dir (installed ctest 3.16 lacks `--test-dir`).

- [ ] **Step 5: Commit**

```bash
git add deploy/bdml_phybot_deploy_c2/core/include/phybot/robot_consts.h \
        deploy/bdml_phybot_deploy_c2/core/include/phybot/data_package.h \
        deploy/bdml_phybot_deploy_c2/core/test/test_walk_numeric.cpp \
        deploy/bdml_phybot_deploy_c2/CMakeLists.txt
git commit -m "feat(core): robot consts + cpg DataPackage fields + walk_numeric test"
```

---

## Task 2: WalkConfig (cpg values) + ZYX→gravity

**Files:**
- Create: `core/include/phybot/walk_config.h`
- Create: `core/include/phybot/gravity.h`
- Modify: `core/test/test_walk_numeric.cpp`

**Interfaces:**
- Produces: `struct phybot::WalkConfig { double ang_vel_scale, dof_pos_scale, dof_vel_scale, lin_vel_scale, walk_flag_threshold; std::array<double,kNumDofs> action_scale, default_dof_pos, rl_p, rl_d; };` and `WalkConfig walk_config();` returning the cpg values.
- Produces: `std::array<double,3> phybot::GravityFromEulerZYX(const std::array<double,3>& zyx);`

- [ ] **Step 1: Write the failing test**

Add to `test_walk_numeric.cpp` (above `main`, add includes at top):
```cpp
#include "phybot/walk_config.h"
#include "phybot/gravity.h"
#include <cmath>

static bool approx(double a, double b) { return std::fabs(a - b) < 1e-12; }
static bool fapprox(float a, float b) { return std::fabs(a - b) < 1e-6f; }  // 缩放浮点比较(避免脆性 ==)

static void test_walk_config_values() {
  auto c = phybot::walk_config();
  CHECK(approx(c.ang_vel_scale, 0.25));
  CHECK(approx(c.dof_vel_scale, 0.05));
  CHECK(approx(c.dof_pos_scale, 1.0));
  CHECK(approx(c.lin_vel_scale, 1.0));
  CHECK(approx(c.walk_flag_threshold, 0.3));
  CHECK(approx(c.action_scale[0], 0.2) && approx(c.action_scale[20], 0.2));
  CHECK(approx(c.default_dof_pos[4], -0.295) && approx(c.default_dof_pos[10], -0.295));
  CHECK(approx(c.default_dof_pos[0], -0.242));
  CHECK(approx(c.rl_p[12], 20.0) && approx(c.rl_d[12], 10.0));
}

static void test_gravity() {
  using phybot::GravityFromEulerZYX;
  auto g0 = GravityFromEulerZYX({0.0, 0.0, 0.0});
  CHECK(approx(g0[0], 0.0) && approx(g0[1], 0.0) && approx(g0[2], -1.0));
  auto gy = GravityFromEulerZYX({0.7, 0.0, 0.0});   // yaw-only → gravity unchanged
  CHECK(approx(gy[0], 0.0) && approx(gy[1], 0.0) && approx(gy[2], -1.0));
  auto gp = GravityFromEulerZYX({0.0, M_PI / 2, 0.0});  // pitch 90° → [1,0,0]
  CHECK(approx(gp[0], 1.0) && approx(gp[1], 0.0) && approx(gp[2], 0.0));
}
```
Add `test_walk_config_values();` and `test_gravity();` inside `main()` before the success print.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build /tmp/a1_build 2>&1 | head -20`
Expected: FAIL — `walk_config.h` / `gravity.h` not found.

- [ ] **Step 3: Write minimal implementation**

Create `core/include/phybot/walk_config.h`:
```cpp
#pragma once
#include <array>
#include "phybot/robot_consts.h"
namespace phybot {
struct WalkConfig {
  double ang_vel_scale, dof_pos_scale, dof_vel_scale, lin_vel_scale, walk_flag_threshold;
  std::array<double, kNumDofs> action_scale;
  std::array<double, kNumDofs> default_dof_pos;
  std::array<double, kNumDofs> rl_p;
  std::array<double, kNumDofs> rl_d;
};
// 官方 RL_deploy_cpg/config/rl_params.yaml 的值 (硬编码; A1-policy 改为从 yaml 载入)。
inline WalkConfig walk_config() {
  WalkConfig c;
  c.ang_vel_scale = 0.25; c.dof_pos_scale = 1.0; c.dof_vel_scale = 0.05;
  c.lin_vel_scale = 1.0;  c.walk_flag_threshold = 0.3;
  c.action_scale = {0.2,0.2,0.2,0.2,0.2,0.2, 0.2,0.2,0.2,0.2,0.2,0.2, 0.2,
                    0.2,0.2,0.2,0.2, 0.2,0.2,0.2,0.2};
  c.default_dof_pos = {-0.242,0.029,0.007,0.49,-0.295,0.0,
                       -0.242,-0.029,-0.007,0.49,-0.295,0.0,
                       0.0,
                       0.0,0.0,0.0,-0.17,
                       0.0,-0.0,0.0,-0.17};
  c.rl_p = {100,100,100,100,50,50, 100,100,100,100,50,50, 20,
            50,50,50,50, 50,50,50,50};
  c.rl_d = {5,5,5,5,5,5, 5,5,5,5,5,5, 10, 5,5,5,5, 5,5,5,5};
  return c;
}
}  // namespace phybot
```

Create `core/include/phybot/gravity.h`:
```cpp
#pragma once
#include <array>
#include <cmath>
namespace phybot {
// 复现官方 Euler_ZYXToGravityVec: R = Rz(a0)Ry(a1)Rx(a2), 返回 -R^T.col(2)。
// 闭式: g = [sin(a1), -cos(a1)sin(a2), -cos(a1)cos(a2)] (与 yaw a0 无关)。
inline std::array<double, 3> GravityFromEulerZYX(const std::array<double, 3>& zyx) {
  const double a1 = zyx[1], a2 = zyx[2];
  return { std::sin(a1), -std::cos(a1) * std::sin(a2), -std::cos(a1) * std::cos(a2) };
}
}  // namespace phybot
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build /tmp/a1_build && (cd /tmp/a1_build && ctest --output-on-failure)`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deploy/bdml_phybot_deploy_c2/core/include/phybot/walk_config.h \
        deploy/bdml_phybot_deploy_c2/core/include/phybot/gravity.h \
        deploy/bdml_phybot_deploy_c2/core/test/test_walk_numeric.cpp
git commit -m "feat(core): WalkConfig (cpg values) + ZYX->gravity"
```

---

## Task 3: WalkHistory (pos/vel buffers, newest-at-tail, quirk reproduction)

**Files:**
- Create: `core/include/phybot/walk_history.h`
- Modify: `core/test/test_walk_numeric.cpp`

**Interfaces:**
- Consumes: `WalkConfig`, `kNumDofs`, `kHistLen`, `kHistFramesInObs`.
- Produces: `class phybot::WalkHistory` with:
  - ctor: zero-initializes both buffers (size `kHistLen*kNumDofs = 420` each).
  - `void Push(const std::array<double,kNumDofs>& q, const std::array<double,kNumDofs>& dq, const WalkConfig& c);` — shifts each buffer left by `kNumDofs`, appends at tail: pos tail = `q[i]*dof_pos_scale` (**NO default subtraction — quirk #1**), vel tail = `dq[i]*dof_vel_scale`.
  - `void NewestPos(float* out105) const;` / `void NewestVel(float* out105) const;` — copy the newest `kHistFramesInObs*kNumDofs = 105` entries (the tail) into `out` as `float`.
  - `void Reset();` — zero both buffers.

- [ ] **Step 1: Write the failing test**

Add to `test_walk_numeric.cpp`:
```cpp
#include "phybot/walk_history.h"

static void test_history_zero_init_and_push() {
  using namespace phybot;
  WalkConfig c = walk_config();
  WalkHistory h;
  float pos105[105], vel105[105];
  h.NewestPos(pos105); h.NewestVel(vel105);
  for (int i = 0; i < 105; ++i) { CHECK(pos105[i] == 0.0f); CHECK(vel105[i] == 0.0f); }

  std::array<double,kNumDofs> q{}, dq{};
  for (int i = 0; i < kNumDofs; ++i) { q[i] = 1.0; dq[i] = 2.0; }  // pos→1.0*1.0, vel→2.0*0.05=0.1
  h.Push(q, dq, c);
  h.NewestPos(pos105); h.NewestVel(vel105);
  // newest frame is the tail block [84..104]; quirk: pos stores q*scale, NO default sub
  CHECK(pos105[104] == 1.0f);        // q=1.0 * dof_pos_scale 1.0 (exact)
  CHECK(fapprox(vel105[104], 0.1f)); // dq=2.0 * dof_vel_scale 0.05
  CHECK(pos105[83] == 0.0f);         // still-zero older frame just before newest
}

static void test_history_ordering() {
  using namespace phybot;
  WalkConfig c = walk_config();
  WalkHistory h;
  std::array<double,kNumDofs> dq{};
  for (int f = 1; f <= 6; ++f) {           // push 6 frames with q[0]=f
    std::array<double,kNumDofs> q{}; q[0] = (double)f;
    h.Push(q, dq, c);
  }
  float pos105[105]; h.NewestPos(pos105);
  // newest 5 frames are f=2,3,4,5,6 (oldest f=1 shifted out of the 5-frame window)
  CHECK(pos105[0]  == 2.0f);    // frame f=2, dof 0
  CHECK(pos105[84] == 6.0f);    // newest frame f=6, dof 0 (start of tail block)
}
```
Add the two test calls inside `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build /tmp/a1_build 2>&1 | head -20`
Expected: FAIL — `walk_history.h` not found.

- [ ] **Step 3: Write minimal implementation**

Create `core/include/phybot/walk_history.h`:
```cpp
#pragma once
#include <array>
#include <vector>
#include "phybot/robot_consts.h"
#include "phybot/walk_config.h"
namespace phybot {
class WalkHistory {
 public:
  WalkHistory() : pos_(kHistLen * kNumDofs, 0.0), vel_(kHistLen * kNumDofs, 0.0) {}

  void Reset() {
    std::fill(pos_.begin(), pos_.end(), 0.0);
    std::fill(vel_.begin(), vel_.end(), 0.0);
  }

  void Push(const std::array<double, kNumDofs>& q,
            const std::array<double, kNumDofs>& dq, const WalkConfig& c) {
    const int n = kNumDofs, total = kHistLen * kNumDofs;
    for (int i = 0; i < total - n; ++i) { pos_[i] = pos_[i + n]; vel_[i] = vel_[i + n]; }
    for (int i = 0; i < n; ++i) {
      pos_[total - n + i] = q[i] * c.dof_pos_scale;   // quirk #1: NO default subtraction
      vel_[total - n + i] = dq[i] * c.dof_vel_scale;
    }
  }

  void NewestPos(float* out105) const { CopyTail(pos_, out105); }
  void NewestVel(float* out105) const { CopyTail(vel_, out105); }

 private:
  static void CopyTail(const std::vector<double>& buf, float* out) {
    const int k = kHistFramesInObs * kNumDofs;        // 105
    const int start = (int)buf.size() - k;
    for (int i = 0; i < k; ++i) out[i] = (float)buf[start + i];
  }
  std::vector<double> pos_, vel_;
};
}  // namespace phybot
```
(Add `#include <algorithm>` for `std::fill` if your toolchain needs it — include it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build /tmp/a1_build && (cd /tmp/a1_build && ctest --output-on-failure)`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deploy/bdml_phybot_deploy_c2/core/include/phybot/walk_history.h \
        deploy/bdml_phybot_deploy_c2/core/test/test_walk_numeric.cpp
git commit -m "feat(core): WalkHistory buffers (newest-at-tail, no-default quirk)"
```

---

## Task 4: BuildWalkObs — the 283-D observation recipe (the crux)

**Files:**
- Create: `core/include/phybot/walk_obs.h`
- Modify: `core/test/test_walk_numeric.cpp`

**Interfaces:**
- Consumes: `WalkConfig`, `kObsDim`, `kNumDofs`, `kProprio`, and the newest-105 pos/vel arrays from `WalkHistory`.
- Produces:
  ```cpp
  struct WalkObsInputs {
    std::array<double,3>        gravity;       // from GravityFromEulerZYX
    std::array<double,3>        ang_vel;       // imu_angular_vel
    std::array<double,kNumDofs> q;             // motor_pos
    std::array<double,kNumDofs> dq;            // motor_vel
    std::array<double,kNumDofs> last_action;   // raw previous net output
    double js_vx, js_vy, js_wz;
  };
  void BuildWalkObs(const WalkObsInputs& in, const WalkConfig& c,
                    const float* pos_newest105, const float* vel_newest105,
                    float* obs283);   // writes exactly kObsDim floats
  ```
  Layout (exact): `[0:3]=ang_vel*ang_vel_scale; [3:6]=gravity; [6:9]=cmd=[js_vx*lin_vel_scale, js_vy*lin_vel_scale, js_wz*ang_vel_scale]; [9]=(|cmd0|<walk_flag_threshold)?0:1; [10:31]=(q-default)*dof_pos_scale; [31:52]=dq*dof_vel_scale; [52:73]=last_action; [73:178]=pos_newest105; [178:283]=vel_newest105`.

- [ ] **Step 1: Write the failing test**

Add to `test_walk_numeric.cpp`:
```cpp
#include "phybot/walk_obs.h"

static void test_build_walk_obs() {
  using namespace phybot;
  WalkConfig c = walk_config();
  WalkObsInputs in{};
  in.gravity = {0.0, 0.0, -1.0};
  in.ang_vel = {0.4, 0.8, -0.2};               // *0.25 → 0.1,0.2,-0.05
  in.q = c.default_dof_pos;                     // q-default = 0
  for (int i = 0; i < kNumDofs; ++i) in.dq[i] = 0.2;   // *0.05 → 0.01
  for (int i = 0; i < kNumDofs; ++i) in.last_action[i] = 0.3;
  in.js_vx = 0.5; in.js_vy = -0.1; in.js_wz = 0.4;     // cmd=0.5,-0.1,0.1
  float zeros105[105] = {0};
  float obs[kObsDim];
  BuildWalkObs(in, c, zeros105, zeros105, obs);

  CHECK(fapprox(obs[0], 0.1f) && fapprox(obs[1], 0.2f) && fapprox(obs[2], -0.05f));  // ang_vel*0.25
  CHECK(obs[3] == 0.0f && obs[4] == 0.0f && obs[5] == -1.0f);    // gravity (exact)
  CHECK(obs[6] == 0.5f && obs[7] == -0.1f && fapprox(obs[8], 0.1f));  // commands (cmd2=wz*0.25)
  CHECK(obs[9] == 1.0f);                                         // |0.5|>=0.3 → walk
  CHECK(obs[10] == 0.0f && obs[30] == 0.0f);                     // (q-default)=0
  CHECK(fapprox(obs[31], 0.01f) && fapprox(obs[51], 0.01f));     // dq*0.05
  CHECK(obs[52] == 0.3f && obs[72] == 0.3f);                     // last_action raw (direct copy)
  CHECK(obs[73] == 0.0f && obs[177] == 0.0f);                    // zero pos history
  CHECK(obs[178] == 0.0f && obs[282] == 0.0f);                   // zero vel history
}

static void test_walk_flag_threshold() {
  using namespace phybot;
  WalkConfig c = walk_config();
  WalkObsInputs in{}; in.q = c.default_dof_pos;
  float z[105] = {0}; float obs[kObsDim];
  in.js_vx = 0.2; BuildWalkObs(in, c, z, z, obs); CHECK(obs[9] == 0.0f);  // |0.2|<0.3 → stand
  in.js_vx = 0.3; BuildWalkObs(in, c, z, z, obs); CHECK(obs[9] == 1.0f);  // 0.3 not < 0.3 → walk
  in.js_vx = -0.5; BuildWalkObs(in, c, z, z, obs); CHECK(obs[9] == 1.0f); // abs
}

static void test_pos_minus_default() {
  using namespace phybot;
  WalkConfig c = walk_config();
  WalkObsInputs in{}; in.q = c.default_dof_pos; in.q[0] += 0.1; in.q[20] -= 0.05;
  float z[105] = {0}; float obs[kObsDim];
  BuildWalkObs(in, c, z, z, obs);
  CHECK(std::fabs(obs[10] - 0.1f) < 1e-6f);     // (q0-default0)*1.0
  CHECK(std::fabs(obs[30] - (-0.05f)) < 1e-6f); // (q20-default20)*1.0
}
```
Add the three test calls inside `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build /tmp/a1_build 2>&1 | head -20`
Expected: FAIL — `walk_obs.h` not found.

- [ ] **Step 3: Write minimal implementation**

Create `core/include/phybot/walk_obs.h`:
```cpp
#pragma once
#include <array>
#include <cmath>
#include <cstring>
#include "phybot/robot_consts.h"
#include "phybot/walk_config.h"
namespace phybot {
struct WalkObsInputs {
  std::array<double, 3>        gravity{};
  std::array<double, 3>        ang_vel{};
  std::array<double, kNumDofs> q{};
  std::array<double, kNumDofs> dq{};
  std::array<double, kNumDofs> last_action{};
  double js_vx{0}, js_vy{0}, js_wz{0};
};

inline void BuildWalkObs(const WalkObsInputs& in, const WalkConfig& c,
                         const float* pos_newest105, const float* vel_newest105,
                         float* obs) {
  const double cmd0 = in.js_vx * c.lin_vel_scale;
  const double cmd1 = in.js_vy * c.lin_vel_scale;
  const double cmd2 = in.js_wz * c.ang_vel_scale;
  for (int i = 0; i < 3; ++i) obs[i] = (float)(in.ang_vel[i] * c.ang_vel_scale);  // [0:3]
  for (int i = 0; i < 3; ++i) obs[3 + i] = (float)in.gravity[i];                   // [3:6]
  obs[6] = (float)cmd0; obs[7] = (float)cmd1; obs[8] = (float)cmd2;                // [6:9]
  obs[9] = (std::fabs(cmd0) < c.walk_flag_threshold) ? 0.0f : 1.0f;               // [9]
  for (int i = 0; i < kNumDofs; ++i)
    obs[10 + i] = (float)((in.q[i] - c.default_dof_pos[i]) * c.dof_pos_scale);     // [10:31]
  for (int i = 0; i < kNumDofs; ++i)
    obs[10 + kNumDofs + i] = (float)(in.dq[i] * c.dof_vel_scale);                  // [31:52]
  for (int i = 0; i < kNumDofs; ++i)
    obs[10 + 2 * kNumDofs + i] = (float)in.last_action[i];                         // [52:73]
  std::memcpy(obs + kProprio, pos_newest105, sizeof(float) * 105);                 // [73:178]
  std::memcpy(obs + kProprio + 105, vel_newest105, sizeof(float) * 105);           // [178:283]
}
}  // namespace phybot
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build /tmp/a1_build && (cd /tmp/a1_build && ctest --output-on-failure)`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deploy/bdml_phybot_deploy_c2/core/include/phybot/walk_obs.h \
        deploy/bdml_phybot_deploy_c2/core/test/test_walk_numeric.cpp
git commit -m "feat(core): BuildWalkObs 283-D recipe (cpg-exact incl. quirks)"
```

---

## Task 5: ApplyWalkAction — action → joint targets

**Files:**
- Create: `core/include/phybot/walk_action.h`
- Modify: `core/test/test_walk_numeric.cpp`

**Interfaces:**
- Consumes: `WalkConfig`, `DataPackage`, `kNumDofs`.
- Produces: `void phybot::ApplyWalkAction(const std::array<double,kNumDofs>& action_raw, const WalkConfig& c, DataPackage& io);`
  - `io.motor_Pos_desire[i] = action_raw[i]*action_scale[i] + default_dof_pos[i]`.
  - `io.motor_Vel_desire`, `io.motor_Torque_desire`, `io.torq_desire` → all zero.
  - `io.sim_P = rl_p`, `io.sim_D = rl_d`. (No torque-limit clamp — quirk #3.)

- [ ] **Step 1: Write the failing test**

Add to `test_walk_numeric.cpp`:
```cpp
#include "phybot/walk_action.h"

static void test_apply_walk_action() {
  using namespace phybot;
  WalkConfig c = walk_config();
  DataPackage io;
  std::array<double,kNumDofs> act{};
  act[0] = 1.0;    // → 1.0*0.2 + default[0]
  act[4] = 2.0;    // → 2.0*0.2 + (-0.295)
  ApplyWalkAction(act, c, io);
  CHECK(std::fabs(io.motor_Pos_desire[0] - (0.2 + c.default_dof_pos[0])) < 1e-12);
  CHECK(std::fabs(io.motor_Pos_desire[4] - (0.4 + (-0.295))) < 1e-12);
  CHECK(io.motor_Vel_desire[0] == 0.0 && io.motor_Torque_desire[0] == 0.0 && io.torq_desire[0] == 0.0);
  CHECK(io.sim_P[12] == 20.0 && io.sim_D[12] == 10.0);   // rl_p/rl_d at waist joint
}
```
Add the call inside `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build /tmp/a1_build 2>&1 | head -20`
Expected: FAIL — `walk_action.h` not found.

- [ ] **Step 3: Write minimal implementation**

Create `core/include/phybot/walk_action.h`:
```cpp
#pragma once
#include <array>
#include "phybot/robot_consts.h"
#include "phybot/walk_config.h"
#include "phybot/data_package.h"
namespace phybot {
inline void ApplyWalkAction(const std::array<double, kNumDofs>& action_raw,
                            const WalkConfig& c, DataPackage& io) {
  for (int i = 0; i < kNumDofs; ++i) {
    io.motor_Pos_desire[i]     = action_raw[i] * c.action_scale[i] + c.default_dof_pos[i];
    io.motor_Vel_desire[i]     = 0.0;
    io.motor_Torque_desire[i]  = 0.0;
    io.torq_desire[i]          = 0.0;
    io.sim_P[i]                = c.rl_p[i];
    io.sim_D[i]                = c.rl_d[i];
  }
}
}  // namespace phybot
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build /tmp/a1_build && (cd /tmp/a1_build && ctest --output-on-failure)`
Expected: PASS — all tests (`core_task_manager`, `walk_numeric`) green.

- [ ] **Step 5: Commit**

```bash
git add deploy/bdml_phybot_deploy_c2/core/include/phybot/walk_action.h \
        deploy/bdml_phybot_deploy_c2/core/test/test_walk_numeric.cpp
git commit -m "feat(core): ApplyWalkAction (action*scale+default, gains, no clamp)"
```

---

## Done criteria for A1-num

- `ctest` from `/tmp/a1_build` reports both `core_task_manager` and `walk_numeric` passing.
- The cpg numeric pipeline (gravity, 283-obs recipe with all quirks, pos/vel history, action→target) exists as dependency-free headers and is unit-tested against hand-computed values.
- All five cpg quirks (Global Constraints) are reproduced and covered by a test.

## Next plan (A1-policy — needs libtorch + official integration, authored after A1-num lands)

- `PolicyRunner` (libtorch load of `cpg_policy.pt` + forward), `PolicyTask` base, `WalkTask` (TaskId::WALK) wiring obs→runner→action + history-after-inference + decimation gate, generic `task_dry_run` app.
- **Bit-exact alignment gate:** generate a golden file from the actual official `rl_deploy_cpg` (in `deploy/phybot_deploy_c2`, via an added `cpg_golden_gen` target that drives a fixed synthetic frame sequence and dumps inputs + `motor_Pos_desire`/`sim_P`/`sim_D`); the new tree's alignment test replays the same frames through `WalkTask` and asserts bit/tight-float match. This is where the "vs official" gate truly lives; if it reveals a divergence (e.g. the gravity ULP caveat), fix the new code to match official.
- Decide libtorch/Eigen wiring for the new tree's CMake (mirror official `ThirdParty/libtorch`).
