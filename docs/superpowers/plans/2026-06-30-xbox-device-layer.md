# Xbox controller device layer (auto-discovery, hotplug) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A robust, dependency-free controller device layer in `deploy/bdml_phybot_deploy_c2/` that AUTO-DISCOVERS the Xbox gamepad by name (never hardcodes `js0`), reads it into a normalized `ControllerState`, and recovers from unplug/replug — so it survives migration to the real robot where the device node index differs.

**Architecture:** Linux joystick (`js`) API via the kernel header `<linux/joystick.h>` only — no SDL2/libevdev/libudev (their dev headers are absent; kernel headers are always present, and `/dev/input/js*` is world-readable so no `input`-group/ACL/sudo is needed). Discovery scans `/dev/input/js0..js31`, reads each device's name via `JSIOCGNAME`, and opens the first whose name matches Xbox-like substrings. Pure mapping/normalization logic is factored into a header-only, unit-tested function; the thin device I/O (open/read/hotplug) is verified live with a `controller_probe` app against the connected pad.

**Tech Stack:** C++17, CMake ≥ 3.12, CTest, in-repo `CHECK` macro. Kernel headers `<linux/joystick.h>`, `<sys/ioctl.h>`, `<fcntl.h>`, `<unistd.h>`, `<dirent.h>`. No external libraries.

## Global Constraints

- Target directory: `deploy/bdml_phybot_deploy_c2/`. Paths below are relative to it unless stated.
- C++17, namespace `phybot`, no external libraries (kernel headers only). Must keep the existing 5 ctests passing.
- Build out-of-tree; the installed ctest is 3.16 — run ctest from INSIDE the build dir (`(cd <build> && ctest --output-on-failure)`), never `--test-dir`.
- NEVER hardcode a device index. Discovery is by NAME match, case-insensitive, against substrings: `xbox`, `x-box`, `microsoft`. (The connected pad reports `Microsoft Xbox One X pad`.)
- Device layer reports RAW normalized axes in `[-1,1]` (`value/32767`, clamped). It does NOT apply deadzone/scale/invert/semantics — those belong to the later mapping layer. Triggers follow the xpad raw convention (rest ≈ -1, pressed ≈ +1); the mapping layer remaps to 0..1 if desired.
- Xbox `js` mapping (Linux `xpad` default) used as the documented default — axes: 0=LX,1=LY,2=LT,3=RX,4=RY,5=RT,6=DPAD_X,7=DPAD_Y; buttons: 0=A,1=B,2=X,3=Y,4=LB,5=RB,6=BACK,7=START,8=GUIDE,9=LS,10=RS. This is verified live with `controller_probe` (layouts can vary by pad/kernel); unknown numbers are ignored, not fatal.
- Commit after every task.

## File Structure

- `publisher/include/phybot/controller_state.h` — `struct ControllerState` (normalized buttons + axes + `connected`).
- `publisher/include/phybot/js_mapping.h` — pure, header-only: `bool NameMatchesController(string)`, `void ApplyJsEvent(uint8_t type, uint8_t number, int16_t value, ControllerState&)`. (Unit-tested.)
- `publisher/include/phybot/js_controller.h` + `publisher/src/js_controller.cpp` — `class JsController`: `discover()`, `connected()`, `poll(ControllerState&)`, `close()`. (Device I/O; verified via probe.)
- `publisher/apps/controller_probe.cpp` — interactive app: discover + print live state.
- `publisher/test/test_js_mapping.cpp` — unit tests for the pure logic. ctest `js_mapping`.
- `CMakeLists.txt` — add the `js_mapping` ctest, the `controller_probe` app.

---

## Task 1: ControllerState + pure js mapping/discovery logic (unit-tested)

**Files:**
- Create: `publisher/include/phybot/controller_state.h`
- Create: `publisher/include/phybot/js_mapping.h`
- Create: `publisher/test/test_js_mapping.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `struct phybot::ControllerState { bool a,b,x,y,lb,rb,back,start,guide,ls,rs; float left_x,left_y,right_x,right_y,lt,rt,dpad_x,dpad_y; bool connected; };` all default-false/0.
- Produces: `bool phybot::NameMatchesController(const std::string& name)` — true iff lowercased name contains "xbox" or "x-box" or "microsoft".
- Produces: `void phybot::ApplyJsEvent(uint8_t type, uint8_t number, int16_t value, ControllerState& st)` — strips the `JS_EVENT_INIT` (0x80) flag from `type`; for `JS_EVENT_BUTTON` sets the mapped bool to `value != 0`; for `JS_EVENT_AXIS` sets the mapped axis to `clamp(value/32767.0f, -1, 1)`; unmapped numbers are ignored.
- Produces: CTest `js_mapping`.

- [ ] **Step 1: Write the failing test**

Create `publisher/test/test_js_mapping.cpp`:
```cpp
#include "phybot/controller_state.h"
#include "phybot/js_mapping.h"
#include "check.h"
#include <linux/joystick.h>   // JS_EVENT_BUTTON / JS_EVENT_AXIS / JS_EVENT_INIT
#include <cmath>
#include <cstdio>

static bool fapprox(float a, float b) { return std::fabs(a - b) < 1e-4f; }

static void test_name_match() {
  using phybot::NameMatchesController;
  CHECK(NameMatchesController("Microsoft Xbox One X pad"));
  CHECK(NameMatchesController("Xbox Wireless Controller"));
  CHECK(NameMatchesController("X-Box 360 pad"));
  CHECK(NameMatchesController("microsoft x-box one"));     // case-insensitive
  CHECK(!NameMatchesController("Logitech M500s Mouse"));
  CHECK(!NameMatchesController("3Dconnexion SpaceMouse"));
}

static void test_buttons() {
  using namespace phybot;
  ControllerState st{};
  ApplyJsEvent(JS_EVENT_BUTTON, 0, 1, st); CHECK(st.a);
  ApplyJsEvent(JS_EVENT_BUTTON, 1, 1, st); CHECK(st.b);
  ApplyJsEvent(JS_EVENT_BUTTON, 3, 1, st); CHECK(st.y);
  ApplyJsEvent(JS_EVENT_BUTTON, 0, 0, st); CHECK(!st.a);              // release
  ApplyJsEvent(JS_EVENT_BUTTON | JS_EVENT_INIT, 7, 1, st); CHECK(st.start);  // INIT flag stripped
  ApplyJsEvent(JS_EVENT_BUTTON, 99, 1, st);                          // unknown -> ignored, no crash
}

static void test_axes() {
  using namespace phybot;
  ControllerState st{};
  ApplyJsEvent(JS_EVENT_AXIS, 0,  32767, st); CHECK(fapprox(st.left_x, 1.0f));
  ApplyJsEvent(JS_EVENT_AXIS, 1, -32767, st); CHECK(fapprox(st.left_y, -1.0f));
  ApplyJsEvent(JS_EVENT_AXIS, 3,  16384, st); CHECK(fapprox(st.right_x, 0.5001f) || st.right_x > 0.49f);
  ApplyJsEvent(JS_EVENT_AXIS, 2,  32767, st); CHECK(fapprox(st.lt, 1.0f));   // trigger raw
  ApplyJsEvent(JS_EVENT_AXIS, 6, -32767, st); CHECK(fapprox(st.dpad_x, -1.0f));
}

int main() {
  test_name_match();
  test_buttons();
  test_axes();
  if (g_check_failures == 0) std::printf("ALL JS MAPPING TESTS PASSED\n");
  return g_check_failures == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cmake -S deploy/bdml_phybot_deploy_c2 -B /tmp/xbox_build && cmake --build /tmp/xbox_build 2>&1 | head -20
```
Expected: FAIL — `controller_state.h` / `js_mapping.h` not found and no `js_mapping` target.

- [ ] **Step 3: Write minimal implementation**

Create `publisher/include/phybot/controller_state.h`:
```cpp
#pragma once
namespace phybot {
// 归一化手柄状态。设备层只填原始归一化值([-1,1]); 死区/缩放/反向在 mapping 层。
struct ControllerState {
  bool a=false, b=false, x=false, y=false;
  bool lb=false, rb=false;
  bool back=false, start=false, guide=false;
  bool ls=false, rs=false;                       // 摇杆按下
  float left_x=0, left_y=0, right_x=0, right_y=0; // [-1,1]
  float lt=0, rt=0;                              // 扳机, xpad 原始约定(松≈-1, 满≈+1)
  float dpad_x=0, dpad_y=0;                      // 方向键(轴形式)
  bool connected=false;
};
}  // namespace phybot
```

Create `publisher/include/phybot/js_mapping.h`:
```cpp
#pragma once
#include <cstdint>
#include <string>
#include <algorithm>
#include <cctype>
#include "phybot/controller_state.h"
namespace phybot {

inline bool NameMatchesController(const std::string& name) {
  std::string n = name;
  std::transform(n.begin(), n.end(), n.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return n.find("xbox") != std::string::npos ||
         n.find("x-box") != std::string::npos ||
         n.find("microsoft") != std::string::npos;
}

// type: JS_EVENT_BUTTON(0x01) / JS_EVENT_AXIS(0x02), possibly OR'd with JS_EVENT_INIT(0x80).
inline void ApplyJsEvent(uint8_t type, uint8_t number, int16_t value, ControllerState& st) {
  type &= ~0x80;  // strip JS_EVENT_INIT
  if (type == 0x01) {                       // BUTTON
    const bool on = value != 0;
    switch (number) {
      case 0: st.a = on; break;     case 1: st.b = on; break;
      case 2: st.x = on; break;     case 3: st.y = on; break;
      case 4: st.lb = on; break;    case 5: st.rb = on; break;
      case 6: st.back = on; break;  case 7: st.start = on; break;
      case 8: st.guide = on; break; case 9: st.ls = on; break;
      case 10: st.rs = on; break;   default: break;  // unknown -> ignore
    }
  } else if (type == 0x02) {                // AXIS
    float v = (float)value / 32767.0f;
    if (v > 1.0f) v = 1.0f; else if (v < -1.0f) v = -1.0f;
    switch (number) {
      case 0: st.left_x = v; break;   case 1: st.left_y = v; break;
      case 2: st.lt = v; break;       case 3: st.right_x = v; break;
      case 4: st.right_y = v; break;  case 5: st.rt = v; break;
      case 6: st.dpad_x = v; break;   case 7: st.dpad_y = v; break;
      default: break;
    }
  }
}
}  // namespace phybot
```

Modify `CMakeLists.txt` — append:
```cmake
add_executable(test_js_mapping publisher/test/test_js_mapping.cpp)
target_include_directories(test_js_mapping PRIVATE publisher/include core/test)
add_test(NAME js_mapping COMMAND test_js_mapping)
```
(`core/test/check.h` is reused for the `CHECK` macro — hence the `core/test` include.)

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cmake -S deploy/bdml_phybot_deploy_c2 -B /tmp/xbox_build && cmake --build /tmp/xbox_build && (cd /tmp/xbox_build && ctest --output-on-failure)
```
Expected: PASS — `js_mapping` passes; the existing 5 ctests still pass (6/6).

- [ ] **Step 5: Commit**

```bash
git add deploy/bdml_phybot_deploy_c2/publisher/include/phybot/controller_state.h \
        deploy/bdml_phybot_deploy_c2/publisher/include/phybot/js_mapping.h \
        deploy/bdml_phybot_deploy_c2/publisher/test/test_js_mapping.cpp \
        deploy/bdml_phybot_deploy_c2/CMakeLists.txt
git commit -m "feat(publisher): ControllerState + pure js mapping/name-match (unit-tested)"
```

> Note: the `publisher/` subtree is new; ensure `.gitignore` tracks it. The repo `.gitignore` ignores `deploy/*` with explicit `!deploy/bdml_phybot_deploy_c2/...` exceptions only for `core/`. Add exceptions for the new subtree if `git add` reports the files are ignored: append to `.gitignore`:
> ```
> !deploy/bdml_phybot_deploy_c2/publisher/
> !deploy/bdml_phybot_deploy_c2/publisher/**
> ```
> (Verify with `git check-ignore -v <file>`; only add what's needed.)

---

## Task 2: JsController (discover/poll/hotplug) + controller_probe app

**Files:**
- Create: `publisher/include/phybot/js_controller.h`
- Create: `publisher/src/js_controller.cpp`
- Create: `publisher/apps/controller_probe.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `ControllerState`, `NameMatchesController`, `ApplyJsEvent`.
- Produces: `class phybot::JsController` with:
  - `bool discover();` — scan `/dev/input/js0..js31`; for each, `open(O_RDONLY|O_NONBLOCK)`, `ioctl(fd, JSIOCGNAME(sizeof buf), buf)`; if `NameMatchesController(name)` keep it (store fd + name, set `state_.connected=true`, return true), else `close`. Return false if none matched. Closes any previously-open fd first.
  - `bool connected() const;`
  - `bool poll(ControllerState& out);` — if not connected return false; non-blocking-drain all pending `struct js_event` via `read()`; for each call `ApplyJsEvent(ev.type, ev.number, ev.value, state_)`; on `read()` error with `errno==ENODEV` (or `EBADF`) mark disconnected (`close`, `state_.connected=false`) and return false; otherwise copy `state_` to `out` and return true.
  - `void close();` and destructor closing the fd.
  - Stores `int fd_=-1`, `std::string name_`, `ControllerState state_`.
- Produces: app `controller_probe` (target), printing discovered device name and live state at ~50 Hz; on disconnect, retries `discover()` every ~1 s.

- [ ] **Step 1: Write the implementation (no ctest — needs live hardware; verified via probe)**

Create `publisher/include/phybot/js_controller.h`:
```cpp
#pragma once
#include <string>
#include "phybot/controller_state.h"
namespace phybot {
class JsController {
 public:
  ~JsController();
  bool discover();                  // scan /dev/input/js*, open first Xbox-like by name
  bool connected() const { return fd_ >= 0; }
  bool poll(ControllerState& out);  // drain events; false if disconnected
  void close();
  const std::string& name() const { return name_; }
 private:
  int fd_ = -1;
  std::string name_;
  ControllerState state_{};
};
}  // namespace phybot
```

Create `publisher/src/js_controller.cpp`:
```cpp
#include "phybot/js_controller.h"
#include "phybot/js_mapping.h"
#include <linux/joystick.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cerrno>
#include <cstdio>

namespace phybot {

JsController::~JsController() { close(); }

void JsController::close() {
  if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
  state_.connected = false;
}

bool JsController::discover() {
  close();
  char path[32], nbuf[256];
  for (int i = 0; i < 32; ++i) {
    std::snprintf(path, sizeof(path), "/dev/input/js%d", i);
    int fd = ::open(path, O_RDONLY | O_NONBLOCK);
    if (fd < 0) continue;
    nbuf[0] = '\0';
    if (ioctl(fd, JSIOCGNAME(sizeof(nbuf)), nbuf) < 0) { ::close(fd); continue; }
    std::string name(nbuf);
    if (NameMatchesController(name)) {
      fd_ = fd; name_ = name; state_ = ControllerState{}; state_.connected = true;
      return true;
    }
    ::close(fd);
  }
  return false;
}

bool JsController::poll(ControllerState& out) {
  if (fd_ < 0) return false;
  struct js_event ev;
  for (;;) {
    ssize_t n = ::read(fd_, &ev, sizeof(ev));
    if (n == sizeof(ev)) { ApplyJsEvent(ev.type, ev.number, ev.value, state_); continue; }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;  // no more events now
    if (n < 0 && (errno == ENODEV || errno == EBADF)) { close(); return false; }  // unplugged
    break;  // partial/other -> stop draining this cycle
  }
  out = state_;
  return true;
}

}  // namespace phybot
```

Create `publisher/apps/controller_probe.cpp`:
```cpp
#include "phybot/js_controller.h"
#include <cstdio>
#include <ctime>
int main() {
  phybot::JsController jc;
  std::printf("Discovering Xbox controller (move sticks / press buttons; Ctrl-C to quit)...\n");
  timespec ts{0, 20 * 1000 * 1000};  // 20 ms ~ 50 Hz
  int since_scan = 0;
  while (true) {
    if (!jc.connected()) {
      if (jc.discover()) std::printf("CONNECTED: \"%s\"\n", jc.name().c_str());
      else { if (since_scan++ % 50 == 0) std::printf("...no controller; retrying\n"); nanosleep(&ts, nullptr); continue; }
    }
    phybot::ControllerState s;
    if (!jc.poll(s)) { std::printf("DISCONNECTED; rediscovering...\n"); nanosleep(&ts, nullptr); continue; }
    std::printf("\rA%d B%d X%d Y%d LB%d RB%d | LX%+.2f LY%+.2f RX%+.2f RY%+.2f LT%+.2f RT%+.2f DX%+.0f DY%+.0f   ",
                s.a,s.b,s.x,s.y,s.lb,s.rb, s.left_x,s.left_y,s.right_x,s.right_y,s.lt,s.rt,s.dpad_x,s.dpad_y);
    std::fflush(stdout);
    nanosleep(&ts, nullptr);
  }
  return 0;
}
```

Modify `CMakeLists.txt` — append:
```cmake
add_library(phybot_publisher STATIC publisher/src/js_controller.cpp)
target_include_directories(phybot_publisher PUBLIC publisher/include)

add_executable(controller_probe publisher/apps/controller_probe.cpp)
target_link_libraries(controller_probe PRIVATE phybot_publisher)
```

- [ ] **Step 2: Build**

Run:
```bash
cmake -S deploy/bdml_phybot_deploy_c2 -B /tmp/xbox_build && cmake --build /tmp/xbox_build --target controller_probe
```
Expected: compiles cleanly; `controller_probe` binary produced.

- [ ] **Step 3: Verify discovery + name match against the live device (no input needed)**

Run for ~1 s and confirm it discovers the Xbox by name (not by hardcoded index):
```bash
timeout 1 /tmp/xbox_build/controller_probe | head -3
```
Expected: prints `CONNECTED: "Microsoft Xbox One X pad"` (the connected pad), proving name-based auto-discovery works among the host's many input devices.

- [ ] **Step 4: (Manual, by the human) live mapping check**

The human runs `/tmp/xbox_build/controller_probe`, presses each button and moves each stick/trigger/dpad, and confirms the printed fields match (A→A, left stick→LX/LY, etc.). If any number is mismatched on this specific pad, adjust the `switch` numbers in `js_mapping.h` and re-run `js_mapping` unit test + probe. (This step is human-verified; it is not an automated ctest.)

- [ ] **Step 5: Commit**

```bash
git add deploy/bdml_phybot_deploy_c2/publisher/include/phybot/js_controller.h \
        deploy/bdml_phybot_deploy_c2/publisher/src/js_controller.cpp \
        deploy/bdml_phybot_deploy_c2/publisher/apps/controller_probe.cpp \
        deploy/bdml_phybot_deploy_c2/CMakeLists.txt
git commit -m "feat(publisher): JsController auto-discovery + hotplug + controller_probe"
```

---

## Done criteria for the device layer

- `js_mapping` ctest passes; existing 5 ctests still pass.
- `controller_probe` auto-discovers the Xbox by NAME (prints `Microsoft Xbox One X pad`) without any hardcoded index, among the host's ~27 input devices.
- Unplugging/replugging the pad while `controller_probe` runs shows `DISCONNECTED` then re-`CONNECTED` (human-verified).
- Live button/axis mapping confirmed against the actual pad (human-verified); any pad-specific number tweaks folded back into `js_mapping.h` + unit test.

## Roadmap (subsequent input-layer plans — authored when reached)

- **Mapping layer**: `ControllerState` + `config/bindings.yaml` → `Command{requested_model/task, vx, vy, wz}` with deadzone/scale/invert/slew and button→task (debounce N2). (Needs a tiny YAML reader or hand-rolled parser; dependency-light.)
- **Transport**: `InputProtocol` (encode/decode the official 0xeb90 wire format, shared) + the LCM publisher process (reads JsController → mapping → InputProtocol::encode → LCM publish). Brings in the official LCM lib (interim ThirdParty path).
- **Main-side**: LCM subscribe + `InputProtocol::decode` → set `io.requested_task` (model→TaskId) + `io.js_*` continuous command (replacing the fixed command currently in `sim_walk`); enum cross-thread (N1) + main-side clamp.
- Then the sim/main runs the walk task under live Xbox control end-to-end.
