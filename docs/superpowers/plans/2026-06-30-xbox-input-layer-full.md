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

## Task 3: ZeroState task (PD hold to default pose)

**Files:**
- Create: `tasks/zero/zero_task.h`, `tasks/zero/zero_task.cpp`
- Create: `core/test/test_zero_task.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `class phybot::ZeroState : public phybot::Task` — ctor id `TaskId::ZERO` name `"zero"`; `run(DataPackage& io)` sets `io.motor_Pos_desire = walk_config().default_dof_pos`, `io.motor_Vel_desire=0`, `io.motor_Torque_desire=0`, `io.sim_P = rl_p`, `io.sim_D = rl_d`. Registered `REGISTER_TASK(TaskId::ZERO, "zero", ZeroState)`.
- Produces: CTest `zero_task`.

- [ ] **Step 1: Write the failing test**

Create `core/test/test_zero_task.cpp`:
```cpp
#include "phybot/zero_task.h"
#include "phybot/walk_config.h"
#include "phybot/data_package.h"
#include "check.h"
#include <cmath>
#include <cstdio>
static void test_zero_holds_default() {
  using namespace phybot;
  ZeroState z; DataPackage io;
  z.run(io);
  auto c = walk_config();
  for (int i = 0; i < kNumDofs; ++i) {
    CHECK(std::fabs(io.motor_Pos_desire[i] - c.default_dof_pos[i]) < 1e-12);
    CHECK(io.motor_Vel_desire[i] == 0.0);
    CHECK(io.sim_P[i] == c.rl_p[i] && io.sim_D[i] == c.rl_d[i]);
  }
}
int main(){ test_zero_holds_default();
  if(g_check_failures==0) std::printf("ALL ZERO TASK TESTS PASSED\n");
  return g_check_failures==0?0:1; }
```

- [ ] **Step 2: Verify fail** — `zero_task.h` missing.

- [ ] **Step 3: Write minimal implementation**

Create `tasks/zero/zero_task.h`:
```cpp
#pragma once
#include "phybot/task.h"
namespace phybot {
class ZeroState : public Task {
 public:
  ZeroState() : Task(TaskId::ZERO, "zero") {}
  void run(DataPackage& io) override;
};
}  // namespace phybot
```

Create `tasks/zero/zero_task.cpp`:
```cpp
#include "phybot/zero_task.h"
#include "phybot/walk_config.h"
#include "phybot/walk_action.h"
#include "phybot/data_package.h"
#include "phybot/task_registry.h"
namespace phybot {
void ZeroState::run(DataPackage& io) {
  static const WalkConfig c = walk_config();
  for (int i = 0; i < kNumDofs; ++i) {
    io.motor_Pos_desire[i]    = c.default_dof_pos[i];
    io.motor_Vel_desire[i]    = 0.0;
    io.motor_Torque_desire[i] = 0.0;
    io.torq_desire[i]         = 0.0;
    io.sim_P[i]               = c.rl_p[i];
    io.sim_D[i]               = c.rl_d[i];
  }
}
REGISTER_TASK(TaskId::ZERO, "zero", ZeroState)
}  // namespace phybot
```
(Place `zero_task.h` so `#include "phybot/zero_task.h"` resolves — add `tasks/zero` to the include path of the test/targets, or put the header under `core/include/phybot/`. Simplest: create it at `core/include/phybot/zero_task.h` and the .cpp at `tasks/zero/zero_task.cpp`. Adjust the create path accordingly.)

Modify `CMakeLists.txt` — append:
```cmake
add_executable(test_zero_task core/test/test_zero_task.cpp tasks/zero/zero_task.cpp)
target_include_directories(test_zero_task PRIVATE core/include core/test)
add_test(NAME zero_task COMMAND test_zero_task)
```

- [ ] **Step 4: Run + Commit**

```bash
git add deploy/bdml_phybot_deploy_c2/core/include/phybot/zero_task.h \
        deploy/bdml_phybot_deploy_c2/tasks/zero/zero_task.cpp \
        deploy/bdml_phybot_deploy_c2/core/test/test_zero_task.cpp \
        deploy/bdml_phybot_deploy_c2/CMakeLists.txt
git commit -m "feat(tasks): ZeroState PD-hold-default-pose task (TaskId::ZERO)"
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
  TaskManager mgr; TaskRegistry::instance().loadInto(mgr);
  DataPackage io; mgr.start(TaskId::ZERO, io);
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
