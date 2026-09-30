# Xbox input layer — mapping + LCM transport + teleop wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Xbox input layer so the controller drives the live mujoco sim end-to-end: controller state → command mapping → LCM (two-process) → main decodes → TaskManager switches task + feeds the walk velocity command. Builds on the already-done device layer (`JsController`) and the bit-exact `WalkTask`.

**Architecture:** Two processes over LCM (real-robot-faithful, same wire format as official `0xeb90` packet). Publisher: `JsController` → mapping(`ControllerState`→`DriveCmd`) → `InputProtocol::encode` → LCM publish on `Robot_ControlData`. Main (sim): LCM subscribe → `InputProtocol::decode` → clamp → `io.requested_task` (model→TaskId) + `io.js_*` → `TaskManager.tick` (ZERO + WALK) → `MujocoBackend` step. Pure pieces (codec, mapping, ZeroState) are unit-tested; transport is verified with an automated localhost LCM loopback; the full chain is verified with a headless teleop smoke test that injects commands over LCM; the live Xbox-in-hand demo is human-run.

**Tech Stack:** C++17, CMake, CTest. LCM (`liblcm`, `lcm-cpp.hpp`), glib-2.0, libtorch, mujoco — all via interim paths into the official `deploy/phybot_deploy_c2` ThirdParty (already how torch/mujoco are wired; comment to vendor later). The LCM message type reuses the official generated `system2Algorithm_packet.hpp` (copied into the new tree).

## Global Constraints

- Target directory: `deploy/bdml_phybot_deploy_c2/`. Paths relative to it unless stated. C++17, namespace `phybot`. Keep all existing ctests passing.
- ctest 3.16 → run from inside the build dir (no `--test-dir`).
- **Wire format (byte-identical to official, do NOT change):** `system2Algorithm_packet { int16 header; int16 frame_type; int8 data[59]; int8 checksum; }`. `header=0xeb90`, `frame_type=0x0001`. `data[0]=model(int8)`, `data[1..4]=vx(float32 memcpy)`, `data[5..8]=vy`, `data[9..12]=wz`. `checksum = (sum of raw struct bytes [2,63)) % 255` cast to int8. LCM channel `"Robot_ControlData"`.
- **Command limits (clamp on the MAIN side, matching official `get_cmd_vel`):** `vx∈[-0.4,1.0]`, `vy∈[-0.1,0.1]`, `wz∈[-1.0,1.0]`.
- **model→TaskId map (main):** `1→TaskId::ZERO`, `2→TaskId::WALK`. (Matches official `model` ids: 1=zero, 2=RL_walk=cpg.)
- LCM interim include/lib: `deploy/phybot_deploy_c2/lcm_server/lcm/release/{include,lib}` (has `lcm-cpp.hpp`, `liblcm.so`, `libglib-2.0.so.0`). Generated message header copied from `deploy/phybot_deploy_c2/lcm_server/custom_lcm/system2Algorithm_packet.hpp`.
- Do NOT touch the official tree. Commit only new-tree files. Commit after every task.

## File Structure

- `core/include/custom_lcm/system2Algorithm_packet.hpp` — copied generated LCM message (verbatim from official).
- `core/include/phybot/input_protocol.h` (+ `core/src/input_protocol.cpp`) — `DriveCmd` + `encode`/`decode`/`checksum`.
- `publisher/include/phybot/input_mapping.h` — `MapControllerToDrive(ControllerState, MappingState&) -> DriveCmd` (deadzone/scale/invert + button edge-detect).
- `tasks/zero/zero_task.{h,cpp}` — `ZeroState : phybot::Task` (PD hold to default pose), registered `TaskId::ZERO`.
- `publisher/apps/main_publisher.cpp` — the publisher process.
- `apps/sim_teleop.cpp` — the teleop sim main.
- `core/test/test_input_protocol.cpp`, `publisher/test/test_input_mapping.cpp`, `core/test/test_zero_task.cpp`, `core/test/test_lcm_loopback.cpp`, `core/test/test_teleop_smoke.cpp` — ctests.
- `CMakeLists.txt` — targets + ctests.

---

## Task 1: InputProtocol (wire codec) + round-trip test

**Files:**
- Create: `core/include/custom_lcm/system2Algorithm_packet.hpp` (copy of official)
- Create: `core/include/phybot/input_protocol.h`, `core/src/input_protocol.cpp`
- Create: `core/test/test_input_protocol.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `struct phybot::DriveCmd { int8_t model{0}; float vx{0}, vy{0}, wz{0}; };`
- Produces: `int8_t phybot::ip_checksum(const custom_lcm::system2Algorithm_packet&);`
- Produces: `void phybot::ip_encode(const DriveCmd&, custom_lcm::system2Algorithm_packet&);`
- Produces: `bool phybot::ip_decode(const custom_lcm::system2Algorithm_packet&, DriveCmd&);` (validates header, frame_type, checksum)
- Produces: CTest `input_protocol`.

- [ ] **Step 1: Copy the generated LCM header**

```bash
mkdir -p deploy/bdml_phybot_deploy_c2/core/include/custom_lcm
cp deploy/phybot_deploy_c2/lcm_server/custom_lcm/system2Algorithm_packet.hpp \
   deploy/bdml_phybot_deploy_c2/core/include/custom_lcm/system2Algorithm_packet.hpp
```

- [ ] **Step 2: Write the failing test**

Create `core/test/test_input_protocol.cpp`:
```cpp
#include "phybot/input_protocol.h"
#include "check.h"
#include <cmath>
#include <cstdio>
static bool fa(float a, float b){ return std::fabs(a-b) < 1e-6f; }

static void test_roundtrip() {
  using namespace phybot;
  DriveCmd in; in.model = 2; in.vx = 0.37f; in.vy = -0.08f; in.wz = 0.5f;
  custom_lcm::system2Algorithm_packet pkt;
  ip_encode(in, pkt);
  CHECK((uint16_t)pkt.header == 0xeb90);
  CHECK((uint16_t)pkt.frame_type == 0x0001);
  DriveCmd out;
  CHECK(ip_decode(pkt, out));
  CHECK(out.model == 2);
  CHECK(fa(out.vx, 0.37f) && fa(out.vy, -0.08f) && fa(out.wz, 0.5f));
}
static void test_decode_rejects_bad() {
  using namespace phybot;
  DriveCmd in; in.model = 1;
  custom_lcm::system2Algorithm_packet pkt; ip_encode(in, pkt);
  DriveCmd out;
  pkt.checksum = pkt.checksum ^ 0x7F;      // corrupt checksum
  CHECK(!ip_decode(pkt, out));
  ip_encode(in, pkt); pkt.header = 0x1234;  // bad header
  CHECK(!ip_decode(pkt, out));
}
int main(){ test_roundtrip(); test_decode_rejects_bad();
  if(g_check_failures==0) std::printf("ALL INPUT PROTOCOL TESTS PASSED\n");
  return g_check_failures==0?0:1; }
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cmake -S deploy/bdml_phybot_deploy_c2 -B /tmp/in_build && cmake --build /tmp/in_build 2>&1 | head -15`
Expected: FAIL — `input_protocol.h` missing.

- [ ] **Step 4: Write minimal implementation**

Create `core/include/phybot/input_protocol.h`:
```cpp
#pragma once
#include <cstdint>
#include "custom_lcm/system2Algorithm_packet.hpp"
namespace phybot {
struct DriveCmd { int8_t model{0}; float vx{0}, vy{0}, wz{0}; };
int8_t ip_checksum(const custom_lcm::system2Algorithm_packet& p);
void   ip_encode(const DriveCmd& c, custom_lcm::system2Algorithm_packet& p);
bool   ip_decode(const custom_lcm::system2Algorithm_packet& p, DriveCmd& out);
}  // namespace phybot
```

Create `core/src/input_protocol.cpp`:
```cpp
#include "phybot/input_protocol.h"
#include <cstring>
namespace phybot {
static constexpr uint16_t kHeader = 0xeb90, kFrame = 0x0001;
int8_t ip_checksum(const custom_lcm::system2Algorithm_packet& p) {
  const uint8_t* b = reinterpret_cast<const uint8_t*>(&p);
  int32_t s = 0; for (int i = 2; i < 63; ++i) s += b[i];
  return (int8_t)(s % 255);
}
void ip_encode(const DriveCmd& c, custom_lcm::system2Algorithm_packet& p) {
  p.header = (int16_t)kHeader; p.frame_type = (int16_t)kFrame;
  std::memset(p.data, 0, sizeof(p.data));
  p.data[0] = c.model;
  std::memcpy(p.data + 1, &c.vx, 4);
  std::memcpy(p.data + 5, &c.vy, 4);
  std::memcpy(p.data + 9, &c.wz, 4);
  p.checksum = ip_checksum(p);
}
bool ip_decode(const custom_lcm::system2Algorithm_packet& p, DriveCmd& out) {
  if ((uint16_t)p.header != kHeader) return false;
  if ((uint16_t)p.frame_type != kFrame) return false;
  if (ip_checksum(p) != p.checksum) return false;
  out.model = p.data[0];
  std::memcpy(&out.vx, p.data + 1, 4);
  std::memcpy(&out.vy, p.data + 5, 4);
  std::memcpy(&out.wz, p.data + 9, 4);
  return true;
}
}  // namespace phybot
```

Modify `CMakeLists.txt` — append (note: `core/include` is already an include dir for the core lib; the codec needs no LCM link — the generated header is standalone):
```cmake
add_executable(test_input_protocol core/test/test_input_protocol.cpp core/src/input_protocol.cpp)
target_include_directories(test_input_protocol PRIVATE core/include core/test)
add_test(NAME input_protocol COMMAND test_input_protocol)
```

- [ ] **Step 5: Run + Commit**

Run: `cmake --build /tmp/in_build && (cd /tmp/in_build && ctest --output-on-failure)` → all pass (8/8).
```bash
git add deploy/bdml_phybot_deploy_c2/core/include/custom_lcm/system2Algorithm_packet.hpp \
        deploy/bdml_phybot_deploy_c2/core/include/phybot/input_protocol.h \
        deploy/bdml_phybot_deploy_c2/core/src/input_protocol.cpp \
        deploy/bdml_phybot_deploy_c2/core/test/test_input_protocol.cpp \
        deploy/bdml_phybot_deploy_c2/CMakeLists.txt
git commit -m "feat(core): InputProtocol wire codec (0xeb90) + round-trip test"
```

---

## Task 2: Input mapping (ControllerState → DriveCmd) + test

**Files:**
- Create: `publisher/include/phybot/input_mapping.h`
- Create: `publisher/test/test_input_mapping.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `struct phybot::MappingState { bool prev_b=false, prev_y=false; int8_t model=1; };` (carries button edge-detect state + last selected model; starts model=1=ZERO).
- Produces: `phybot::DriveCmd phybot::MapControllerToDrive(const ControllerState& s, MappingState& m);`
  - Buttons (rising-edge select; natural debounce): on B rising edge → `m.model=2`; on Y rising edge → `m.model=1`. Otherwise keep `m.model`.
  - Axes (deadzone 0.08 applied to raw, then scale; main re-clamps): `vx = -dz(s.left_y)*0.6` (stick up = forward), `vy = dz(s.left_x)*0.1`, `wz = -dz(s.right_x)*1.0`.
  - Returns `DriveCmd{ m.model, vx, vy, wz }`.
  - `dz(v)` = `(|v|<0.08)?0:v`.

- [ ] **Step 1: Write the failing test**

Create `publisher/test/test_input_mapping.cpp`:
```cpp
#include "phybot/controller_state.h"
#include "phybot/input_mapping.h"
#include "check.h"
#include <cmath>
#include <cstdio>
static bool fa(float a, float b){ return std::fabs(a-b) < 1e-5f; }

static void test_button_edge_switch() {
  using namespace phybot;
  MappingState m;  ControllerState s{};
  CHECK(MapControllerToDrive(s, m).model == 1);   // default zero
  s.b = true;  CHECK(MapControllerToDrive(s, m).model == 2);  // B rising -> walk
  s.b = true;  CHECK(MapControllerToDrive(s, m).model == 2);  // held -> stays
  s.b = false; CHECK(MapControllerToDrive(s, m).model == 2);  // release -> stays walk
  s.y = true;  CHECK(MapControllerToDrive(s, m).model == 1);  // Y rising -> zero
}
static void test_axes_deadzone_scale() {
  using namespace phybot;
  MappingState m;  ControllerState s{};
  s.left_y = 0.05f;  CHECK(fa(MapControllerToDrive(s,m).vx, 0.0f));   // inside deadzone
  s.left_y = -1.0f;  CHECK(fa(MapControllerToDrive(s,m).vx, 0.6f));   // up -> +vx*0.6
  s.left_x = 1.0f;   CHECK(fa(MapControllerToDrive(s,m).vy, 0.1f));
  s.right_x = 1.0f;  CHECK(fa(MapControllerToDrive(s,m).wz, -1.0f));
}
int main(){ test_button_edge_switch(); test_axes_deadzone_scale();
  if(g_check_failures==0) std::printf("ALL INPUT MAPPING TESTS PASSED\n");
  return g_check_failures==0?0:1; }
```

- [ ] **Step 2: Verify fail** — `cmake --build /tmp/in_build 2>&1 | head` → `input_mapping.h` missing.

- [ ] **Step 3: Write minimal implementation**

Create `publisher/include/phybot/input_mapping.h`:
```cpp
#pragma once
#include <cstdint>
#include <cmath>
#include "phybot/controller_state.h"
#include "phybot/input_protocol.h"
namespace phybot {
struct MappingState { bool prev_b=false, prev_y=false; int8_t model=1; };
inline float ip_dz(float v) { return (std::fabs(v) < 0.08f) ? 0.0f : v; }
inline DriveCmd MapControllerToDrive(const ControllerState& s, MappingState& m) {
  if (s.b && !m.prev_b) m.model = 2;   // B rising edge -> walk
  if (s.y && !m.prev_y) m.model = 1;   // Y rising edge -> zero
  m.prev_b = s.b; m.prev_y = s.y;
  DriveCmd c;
  c.model = m.model;
  c.vx = -ip_dz(s.left_y) * 0.6f;   // stick up (-y) -> forward
  c.vy =  ip_dz(s.left_x) * 0.1f;
  c.wz = -ip_dz(s.right_x) * 1.0f;
  return c;
}
}  // namespace phybot
```

Modify `CMakeLists.txt` — append:
```cmake
add_executable(test_input_mapping publisher/test/test_input_mapping.cpp)
target_include_directories(test_input_mapping PRIVATE publisher/include core/include core/test)
add_test(NAME input_mapping COMMAND test_input_mapping)
```

- [ ] **Step 4: Run + Commit**

`cmake --build /tmp/in_build && (cd /tmp/in_build && ctest --output-on-failure)` → pass.
```bash
git add deploy/bdml_phybot_deploy_c2/publisher/include/phybot/input_mapping.h \
        deploy/bdml_phybot_deploy_c2/publisher/test/test_input_mapping.cpp \
        deploy/bdml_phybot_deploy_c2/CMakeLists.txt
git commit -m "feat(publisher): controller->DriveCmd mapping (edge-switch + deadzone)"
```

---

## Task 3: ZeroState task — FAITHFUL port of official (captured-pose cubic ramp, stiff gains)

> **CRITICAL — safety/alignment.** The official `ZeroState` does NOT snap to a fixed pose. On first entry it CAPTURES `init_pose = current actual motor_pos`, then **cubic-interpolates** (3rd-order polynomial, zero boundary velocities) from `init_pose` to `zero_pose` over `zero_totalTime = 0.5 s`, holding with STIFF gains `zero_p = 1000`, `zero_d = 100`. `zero_pose` equals the cpg `default_dof_pos` exactly, so the subsequent ZERO→WALK handoff is smooth (WalkTask holds default for its first 9 ticks). Reproduce this verbatim; a static snap with soft walk gains would command a violent jump on a real robot powered up in an arbitrary pose.

**Files:**
- Modify: `core/include/phybot/data_package.h` (add `double control_period{0.002};`)
- Create: `core/include/phybot/cubic_traj.h` (verbatim port of official `ThirdpolyVector`)
- Create: `core/include/phybot/zero_config.h` (`ZeroConfig` + `zero_config()`)
- Create: `core/include/phybot/zero_task.h`, `tasks/zero/zero_task.cpp`
- Create: `core/test/test_zero_task.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- `DataPackage` gains `double control_period{0.002};` (the dt the ramp integrates; set by backend/main).
- `void phybot::CubicTraj(const double* p0, const double* p0d, const double* p1, const double* p1d, double T, double t, double* pos, double* vel, int n)` — verbatim port of official `ThirdpolyVector`: `t<0`→`p0`,vel0; `t<=T`→cubic; `t>T`→`p1`,vel0.
- `struct phybot::ZeroConfig { std::array<double,kNumDofs> zero_pose, zero_p, zero_d; double zero_totalTime; };` and `ZeroConfig zero_config()` — `zero_pose = walk_config().default_dof_pos`; `zero_p = {1000×21}`; `zero_d = {100×21}`; `zero_totalTime = 0.5`. (Official ZERO.yaml values; 2 neck joints are not in our 21-dof model.)
- `class phybot::ZeroState : public phybot::Task` — `enter(io)` captures `init_pose_ = io.motor_pos`, `t_ = 0`; `run(io)` advances `t_ += io.control_period`, computes the cubic ramp init_pose_→zero_pose, writes `io.motor_Pos_desire`/`io.motor_Vel_desire`, `io.motor_Torque_desire=0`, `io.sim_P=zero_p`, `io.sim_D=zero_d`. Registered `REGISTER_TASK(TaskId::ZERO, "zero", ZeroState)`.
- CTest `zero_task`.

- [ ] **Step 1: Add `control_period` to DataPackage**

In `core/include/phybot/data_package.h`, add to the struct (near the sensor fields):
```cpp
  double control_period{0.002};   // dt (s); set by backend/main; used by ZeroState ramp
```

- [ ] **Step 2: Write the failing test**

Create `core/test/test_zero_task.cpp`:
```cpp
#include "phybot/zero_task.h"
#include "phybot/zero_config.h"
#include "phybot/cubic_traj.h"
#include "phybot/walk_config.h"
#include "phybot/data_package.h"
#include "check.h"
#include <cmath>
#include <cstdio>

static void test_cubic_boundaries() {
  using namespace phybot;
  double p0[2]={0.0,1.0}, z[2]={0,0}, p1[2]={1.0,-1.0}, pos[2], vel[2];
  CubicTraj(p0,z,p1,z, 0.5, 0.0, pos, vel, 2);
  CHECK(std::fabs(pos[0]-0.0)<1e-12 && std::fabs(pos[1]-1.0)<1e-12);   // t=0 -> p0
  CHECK(std::fabs(vel[0])<1e-12 && std::fabs(vel[1])<1e-12);           // zero boundary vel
  CubicTraj(p0,z,p1,z, 0.5, 0.5, pos, vel, 2);
  CHECK(std::fabs(pos[0]-1.0)<1e-9 && std::fabs(pos[1]-(-1.0))<1e-9);  // t=T -> p1
  CubicTraj(p0,z,p1,z, 0.5, 0.25, pos, vel, 2);
  CHECK(std::fabs(pos[0]-0.5)<1e-9);                                   // symmetric midpoint
  CubicTraj(p0,z,p1,z, 0.5, 1.0, pos, vel, 2);
  CHECK(std::fabs(pos[0]-1.0)<1e-12);                                  // t>T -> p1 hold
}

static void test_zero_ramps_from_captured_pose() {
  using namespace phybot;
  ZeroState zs; DataPackage io;
  io.control_period = 0.002;
  for (int i=0;i<kNumDofs;++i) io.motor_pos[i] = 0.7;   // arbitrary power-on pose
  zs.enter(io);                                          // capture init_pose
  zs.run(io);                                            // first tick: t=0.002, near captured pose
  auto zc = zero_config();
  CHECK(std::fabs(io.motor_Pos_desire[0] - 0.7) < 0.05);          // NOT snapped to zero_pose
  CHECK(io.sim_P[0] == zc.zero_p[0] && io.sim_D[0] == zc.zero_d[0]);  // stiff gains 1000/100
  for (int k=0;k<300;++k) zs.run(io);                   // 300*0.002=0.6s > totalTime
  for (int i=0;i<kNumDofs;++i)
    CHECK(std::fabs(io.motor_Pos_desire[i] - zc.zero_pose[i]) < 1e-9);  // settled at zero_pose
  CHECK(std::fabs(zc.zero_pose[4] - (-0.295)) < 1e-12);            // zero_pose == default_dof_pos
}

int main(){ test_cubic_boundaries(); test_zero_ramps_from_captured_pose();
  if(g_check_failures==0) std::printf("ALL ZERO TASK TESTS PASSED\n");
  return g_check_failures==0?0:1; }
```

- [ ] **Step 3: Verify fail** — headers missing.

- [ ] **Step 4: Write minimal implementation**

Create `core/include/phybot/cubic_traj.h` (verbatim port of official `ThirdpolyVector`):
```cpp
#pragma once
namespace phybot {
// 官方 ZeroState::ThirdpolyVector 逐行移植(三次多项式插值, 边界速度可非零)。
inline void CubicTraj(const double* p0, const double* p0d,
                      const double* p1, const double* p1d,
                      double T, double t, double* pos, double* vel, int n) {
  if (t < 0.0) { for (int i=0;i<n;++i){ pos[i]=p0[i]; vel[i]=0.0; } return; }
  if (t <= T) {
    for (int i=0;i<n;++i) {
      double a0 = p0[i], a1 = p0d[i];
      double m = p1[i] - p0[i] - p0d[i]*T;
      double nn = p1d[i] - p0d[i];
      double a2 = 3.0*m/(T*T) - nn/T;
      double a3 = -2.0*m/(T*T*T) + nn/(T*T);
      pos[i] = a3*t*t*t + a2*t*t + a1*t + a0;
      vel[i] = 3.0*a3*t*t + 2.0*a2*t + a1;
    }
  } else { for (int i=0;i<n;++i){ pos[i]=p1[i]; vel[i]=0.0; } }
}
}  // namespace phybot
```

Create `core/include/phybot/zero_config.h`:
```cpp
#pragma once
#include <array>
#include "phybot/robot_consts.h"
#include "phybot/walk_config.h"
namespace phybot {
struct ZeroConfig {
  std::array<double, kNumDofs> zero_pose;  // == walk default_dof_pos
  std::array<double, kNumDofs> zero_p;     // 1000
  std::array<double, kNumDofs> zero_d;     // 100
  double zero_totalTime;                   // 0.5 s
};
inline ZeroConfig zero_config() {
  ZeroConfig z;
  z.zero_pose = walk_config().default_dof_pos;   // 官方 zero_joint_position[0:21] == cpg default
  z.zero_p.fill(1000.0);
  z.zero_d.fill(100.0);
  z.zero_totalTime = 0.5;
  return z;
}
}  // namespace phybot
```

Create `core/include/phybot/zero_task.h`:
```cpp
#pragma once
#include <array>
#include "phybot/task.h"
#include "phybot/robot_consts.h"
namespace phybot {
class ZeroState : public Task {
 public:
  ZeroState() : Task(TaskId::ZERO, "zero") {}
  void enter(DataPackage& io) override;   // arm first-run capture, reset t
  void run(DataPackage& io) override;     // first run captures init_pose; then cubic ramp
 private:
  std::array<double, kNumDofs> init_pose_{};
  double t_{0.0};
  bool first_{true};   // capture init_pose on first run AFTER pose is read (官方在读位后捕获)
};
}  // namespace phybot
```

Create `tasks/zero/zero_task.cpp`:
```cpp
#include "phybot/zero_task.h"
#include "phybot/zero_config.h"
#include "phybot/cubic_traj.h"
#include "phybot/data_package.h"
#include "phybot/task_registry.h"
namespace phybot {
void ZeroState::enter(DataPackage&) {           // 武装首次捕获(不在此读位姿——start() 早于 backend 读取)
  first_ = true;
  t_ = 0.0;
}
void ZeroState::run(DataPackage& io) {
  static const ZeroConfig z = zero_config();
  static const std::array<double,kNumDofs> zerov{};  // 边界速度 0
  if (first_) {                                  // 官方 If_first_Zero: 读到实际位姿后捕获
    for (int i=0;i<kNumDofs;++i) init_pose_[i] = io.motor_pos[i];
    first_ = false;
  }
  t_ += io.control_period;
  std::array<double,kNumDofs> pos, vel;
  CubicTraj(init_pose_.data(), zerov.data(), z.zero_pose.data(), zerov.data(),
            z.zero_totalTime, t_, pos.data(), vel.data(), kNumDofs);
  for (int i=0;i<kNumDofs;++i) {
    io.motor_Pos_desire[i]    = pos[i];
    io.motor_Vel_desire[i]    = vel[i];
    io.motor_Torque_desire[i] = 0.0;
    io.torq_desire[i]         = 0.0;
    io.sim_P[i]               = z.zero_p[i];
    io.sim_D[i]               = z.zero_d[i];
  }
}
REGISTER_TASK(TaskId::ZERO, "zero", ZeroState)
}  // namespace phybot
```

Modify `CMakeLists.txt` — append:
```cmake
add_executable(test_zero_task core/test/test_zero_task.cpp tasks/zero/zero_task.cpp)
target_include_directories(test_zero_task PRIVATE core/include core/test)
add_test(NAME zero_task COMMAND test_zero_task)
```

- [ ] **Step 5: Run + Commit**

`cmake --build /tmp/in_build && (cd /tmp/in_build && ctest --output-on-failure)` → pass.
```bash
git add deploy/bdml_phybot_deploy_c2/core/include/phybot/data_package.h \
        deploy/bdml_phybot_deploy_c2/core/include/phybot/cubic_traj.h \
        deploy/bdml_phybot_deploy_c2/core/include/phybot/zero_config.h \
        deploy/bdml_phybot_deploy_c2/core/include/phybot/zero_task.h \
        deploy/bdml_phybot_deploy_c2/tasks/zero/zero_task.cpp \
        deploy/bdml_phybot_deploy_c2/core/test/test_zero_task.cpp \
        deploy/bdml_phybot_deploy_c2/CMakeLists.txt
git commit -m "feat(tasks): ZeroState faithful official port (captured-pose cubic ramp, stiff gains)"
```

---

## Task 4: Publisher process + LCM loopback test

**Files:**
- Create: `publisher/apps/main_publisher.cpp`
- Create: `core/test/test_lcm_loopback.cpp`
- Modify: `CMakeLists.txt` (add an interim LCM cmake block + the publisher target + loopback ctest)

**Interfaces:**
- Consumes: `JsController`, `MapControllerToDrive`, `ip_encode`, the LCM lib.
- Produces: executable `main_publisher` (loops `JsController.discover/poll → MapControllerToDrive → ip_encode → lcm.publish("Robot_ControlData", &pkt)` at ~100 Hz; re-discovers on disconnect).
- Produces: CTest `lcm_loopback` — publishes an encoded `DriveCmd` on a real `lcm::LCM`, `handleTimeout`s it back through a subscriber, `ip_decode`s, asserts round-trip. (Pure localhost; no Xbox.)

- [ ] **Step 1: Add the interim LCM cmake block**

In `CMakeLists.txt`, near the existing interim torch/mujoco wiring, add:
```cmake
# --- interim LCM (point at official ThirdParty; vendor later) ---
set(PHYBOT_LCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../phybot_deploy_c2/lcm_server/lcm/release")
add_library(phybot_lcm INTERFACE)
target_include_directories(phybot_lcm INTERFACE "${PHYBOT_LCM_DIR}/include")
target_link_libraries(phybot_lcm INTERFACE
    "${PHYBOT_LCM_DIR}/lib/liblcm.so" "${PHYBOT_LCM_DIR}/lib/libglib-2.0.so.0")
```

- [ ] **Step 2: Write the loopback test (the transport gate)**

Create `core/test/test_lcm_loopback.cpp`:
```cpp
#include "phybot/input_protocol.h"
#include "check.h"
#include <lcm/lcm-cpp.hpp>
#include <cstdio>
struct Handler {
  bool got=false; phybot::DriveCmd cmd;
  void on(const lcm::ReceiveBuffer*, const std::string&,
          const custom_lcm::system2Algorithm_packet* p) {
    got = phybot::ip_decode(*p, cmd);
  }
};
int main() {
  lcm::LCM lcm;
  if (!lcm.good()) { std::printf("LCM init failed (no multicast?)\n"); return 1; }
  Handler h;
  lcm.subscribe("Robot_ControlData", &Handler::on, &h);
  phybot::DriveCmd in; in.model=2; in.vx=0.42f; in.vy=-0.05f; in.wz=0.3f;
  custom_lcm::system2Algorithm_packet pkt; phybot::ip_encode(in, pkt);
  lcm.publish("Robot_ControlData", &pkt);
  // pump for up to ~500ms
  for (int i=0;i<50 && !h.got;++i) lcm.handleTimeout(10);
  CHECK(h.got);
  CHECK(h.cmd.model==2);
  CHECK(std::abs(h.cmd.vx-0.42f)<1e-6 && std::abs(h.cmd.wz-0.3f)<1e-6);
  if (g_check_failures==0) std::printf("LCM LOOPBACK PASSED\n");
  return g_check_failures==0?0:1;
}
```

- [ ] **Step 3: Write the publisher**

Create `publisher/apps/main_publisher.cpp`:
```cpp
#include "phybot/js_controller.h"
#include "phybot/input_mapping.h"
#include "phybot/input_protocol.h"
#include <lcm/lcm-cpp.hpp>
#include <cstdio>
#include <ctime>
int main() {
  lcm::LCM lcm;
  if (!lcm.good()) { std::printf("LCM init failed\n"); return 1; }
  phybot::JsController jc; phybot::MappingState ms;
  timespec ts{0, 10*1000*1000};  // ~100 Hz
  std::printf("[publisher] running; discovering controller...\n");
  while (true) {
    if (!jc.connected()) { if (!jc.discover()) { nanosleep(&ts,nullptr); continue; }
                           std::printf("[publisher] connected: %s\n", jc.name().c_str()); }
    phybot::ControllerState s;
    if (!jc.poll(s)) { nanosleep(&ts,nullptr); continue; }
    phybot::DriveCmd c = phybot::MapControllerToDrive(s, ms);
    custom_lcm::system2Algorithm_packet pkt; phybot::ip_encode(c, pkt);
    lcm.publish("Robot_ControlData", &pkt);
    nanosleep(&ts, nullptr);
  }
}
```

Modify `CMakeLists.txt` — append:
```cmake
add_executable(test_lcm_loopback core/test/test_lcm_loopback.cpp core/src/input_protocol.cpp)
target_include_directories(test_lcm_loopback PRIVATE core/include core/test)
target_link_libraries(test_lcm_loopback PRIVATE phybot_lcm)
add_test(NAME lcm_loopback COMMAND test_lcm_loopback)

add_executable(main_publisher publisher/apps/main_publisher.cpp core/src/input_protocol.cpp)
target_include_directories(main_publisher PRIVATE publisher/include core/include)
target_link_libraries(main_publisher PRIVATE phybot_publisher phybot_lcm)
```

- [ ] **Step 4: Build + run loopback + commit**

Run: `cmake -S deploy/bdml_phybot_deploy_c2 -B /tmp/in_build && cmake --build /tmp/in_build && (cd /tmp/in_build && ctest --output-on-failure)`.
Expected: `lcm_loopback` passes (LCM publish+subscribe+decode round-trips on localhost). If LCM multicast is unavailable in the environment, the test prints "LCM init failed" and returns non-zero — report that as an environment limitation (the codec is still proven by `input_protocol`).
Also build `main_publisher` and run `timeout 1 ./main_publisher` — expect `[publisher] connected: Microsoft Xbox One X pad`.
```bash
git add deploy/bdml_phybot_deploy_c2/publisher/apps/main_publisher.cpp \
        deploy/bdml_phybot_deploy_c2/core/test/test_lcm_loopback.cpp \
        deploy/bdml_phybot_deploy_c2/CMakeLists.txt
git commit -m "feat(publisher): LCM publisher process + localhost loopback test"
```

---

## Task 5: Teleop sim main + headless smoke test

**Files:**
- Create: `apps/sim_teleop.cpp`
- Create: `core/test/test_teleop_smoke.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `MujocoBackend`, `TaskManager`, `TaskRegistry` (ZERO + WALK registered), `ip_decode`, LCM.
- Produces: executable `sim_teleop` — LCM subscribe `Robot_ControlData`; control loop: `handleTimeout(0)` to drain LCM; map latest `model`→`TaskId` into `io.requested_task`; set `io.js_vx_desire/js_vy_desire/js_OmegaZ_desire` = clamped vx/vy/wz; `backend.readToPackage(io)`; `manager.tick(io)`; `backend.writeFromPackage(io)`; `backend.step()`. Prints base height periodically.
- Produces: CTest `teleop_smoke` — drives the same pipeline but injects commands by publishing over LCM in-process (zero for ~0.5 s, then walk with vx=0.4 for ~3 s); asserts the active task switches ZERO→WALK and base height stays > 0.4 m (robot doesn't fall) and advances in x. This is the automated end-to-end gate (LCM transport + decode + task switch + live sim) without the Xbox.

- [ ] **Step 1: Write the smoke test** (full end-to-end, LCM-injected)

Create `core/test/test_teleop_smoke.cpp`:
```cpp
#include "phybot/task_manager.h"
#include "phybot/task_registry.h"
#include "phybot/input_protocol.h"
#include "phybot/mujoco_backend.h"   // adjust include to the backend header's actual path
#include "phybot/data_package.h"
#include <lcm/lcm-cpp.hpp>
#include <cstdio>
#include <cmath>
// model->TaskId helper mirrored from sim_teleop
static phybot::TaskId model_to_task(int8_t m){ return m==2 ? phybot::TaskId::WALK : phybot::TaskId::ZERO; }
struct Cmd { phybot::DriveCmd c; bool any=false; };
struct H { Cmd* cur; void on(const lcm::ReceiveBuffer*, const std::string&,
        const custom_lcm::system2Algorithm_packet* p){ phybot::DriveCmd d; if(phybot::ip_decode(*p,d)){cur->c=d;cur->any=true;} } };
int main(){
  using namespace phybot;
  lcm::LCM lcm; if(!lcm.good()){ std::printf("LCM init failed\n"); return 1; }
  Cmd cur; H h{&cur}; lcm.subscribe("Robot_ControlData", &H::on, &h);
  MujocoBackend backend(/* model + cpg policy paths per the backend's ctor */);
  TaskManager mgr; TaskRegistry::instance().loadInto(mgr);   // ZERO + WALK both registered
  DataPackage io; io.control_period = 0.002;                 // dt for the ZeroState ramp
  mgr.start(TaskId::ZERO, io);                               // start in ZERO (power-on); captures pose on 1st run
  // sequence: ticks 0..249 stay in ZERO (0.5s cubic ramp to default), tick 250 -> WALK
  auto publish=[&](int8_t model,float vx){ DriveCmd d; d.model=model; d.vx=vx;
    custom_lcm::system2Algorithm_packet pkt; ip_encode(d,pkt); lcm.publish("Robot_ControlData",&pkt); };
  double min_h=1e9; bool saw_walk=false;
  for(int tick=0; tick<2000; ++tick){               // 2000*0.002 = 4 s
    if(tick==0) publish(1,0.0f);                     // zero
    if(tick==250) publish(2,0.4f);                   // -> walk forward
    lcm.handleTimeout(0);
    io.requested_task = model_to_task(cur.c.model);
    io.js_vx_desire = std::clamp((double)cur.c.vx,-0.4,1.0);
    io.js_vy_desire = std::clamp((double)cur.c.vy,-0.1,0.1);
    io.js_OmegaZ_desire = std::clamp((double)cur.c.wz,-1.0,1.0);
    backend.readToPackage(io); mgr.tick(io); backend.writeFromPackage(io); backend.step();
    if(mgr.current()==TaskId::WALK) saw_walk=true;
    if(tick>300){ double hgt=backend.baseHeight(); if(hgt<min_h) min_h=hgt; }
  }
  CHECK(saw_walk);            // task switched to walk via LCM command
  CHECK(min_h > 0.4);         // robot did not fall while walking
  std::printf("teleop_smoke: saw_walk=%d min_h=%.3f\n", saw_walk, min_h);
  if(g_check_failures==0) std::printf("TELEOP SMOKE PASSED\n");
  return g_check_failures==0?0:1;
}
```
> The implementer must match the real `MujocoBackend` ctor signature, `baseHeight()`, and the registry/start API as they exist in the tree (read `backends/mujoco/mujoco_backend.h` and `tasks/walk/walk_task.*`), and ensure BOTH ZeroState and WalkTask are registered (WalkTask currently takes a model path — confirm how it is constructed/registered and that `TaskRegistry::loadInto` provides it; if WalkTask needs the policy path, register a factory that supplies `core/test/golden/cpg_policy.pt`). Adjust includes/paths accordingly. Use `-DGOLDEN_DIR` like the other torch tests for the model path.

- [ ] **Step 2: Write `sim_teleop.cpp`** — same loop as the smoke test but without the in-process `publish()` calls (commands arrive from the external `main_publisher`), running indefinitely and printing base height + current task name each ~0.5 s.

- [ ] **Step 3: CMake + build + run smoke**

Add `sim_teleop` and `test_teleop_smoke` targets linking `phybot_lcm`, the torch/mujoco libs, the core, walk_task, zero_task sources (mirror how `walk_sim`/`sim_walk` link torch+mujoco; add `phybot_lcm`). Build; run `(cd /tmp/in_build && ctest -R teleop_smoke --output-on-failure)`; expect `saw_walk=1`, `min_h>0.4`. All other ctests still pass.

- [ ] **Step 4: Commit**

```bash
git add deploy/bdml_phybot_deploy_c2/apps/sim_teleop.cpp \
        deploy/bdml_phybot_deploy_c2/core/test/test_teleop_smoke.cpp \
        deploy/bdml_phybot_deploy_c2/CMakeLists.txt
git commit -m "feat(sim): LCM teleop main + headless end-to-end smoke (switch+walk)"
```

---

## Done criteria

- ctests: `input_protocol`, `input_mapping`, `zero_task`, `lcm_loopback`, `teleop_smoke` all pass (plus all prior). `teleop_smoke` proves the full chain LCM→decode→task-switch→live-sim works headless.
- `main_publisher` auto-connects to the Xbox and publishes; `sim_teleop` subscribes and drives the sim.
- **Live two-process demo (human-run):** terminal 1 `./main_publisher`; terminal 2 `./sim_teleop`. Press **B** → robot walks; left stick forward → speed; **Y** → back to stand. (If LCM multicast isn't available headless, note it; the codec + mapping + task-switch are still proven by the unit/loopback/smoke tests, and the demo runs on the real robot/host with multicast.)

## Roadmap (later)
- bindings.yaml (replace hardcoded mapping); the existing official `lcm_model_map` for App back-compat; vendor LCM/torch/mujoco/model into the new tree's own `third_party/`; the `_template` task.
