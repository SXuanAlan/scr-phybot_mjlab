// test_teleop_smoke: headless end-to-end gate — command -> task-switch (ZERO->WALK)
// -> live MuJoCo sim, with NO LCM in the loop.
//
// This sandbox has no multicast route and the vendored liblcm ABORTS (buffer overflow)
// inside handleTimeout() when used headless (see the test_lcm_loopback note in
// CMakeLists.txt). So instead of publishing over LCM, this test sets a DriveCmd
// directly in-process and feeds it through the exact same ApplyCmdAndTick() helper
// that the real LCM-driven apps/sim_teleop.cpp uses (see backends/mujoco/teleop_loop.h).
// That helper is the only thing that would differ between "commanded via LCM" and
// "commanded directly" -- everything downstream (task switch + live sim) is identical.
#include "phybot/data_package.h"
#include "phybot/task_manager.h"
#include "phybot/task_id.h"
#include "phybot/zero_task.h"
#include "phybot/input_protocol.h"
#include "walk/walk_task.h"
#include "mujoco_backend.h"
#include "teleop_loop.h"
#include "check.h"
#include <cstdio>
#include <memory>

// Paths injected by CMake (mirrors test_walk_sim's POLICY_PATH/XML_PATH pattern).
#ifndef POLICY_PATH
#define POLICY_PATH "core/test/golden/cpg_policy.pt"
#endif
#ifndef XML_PATH
#define XML_PATH "assets/robot/phybot_c2/xml/phybot_c2.xml"
#endif

int main() {
  using namespace phybot;

  // tasks/zero/config/zero.yaml: ramp_total_time=0.5s -> 250 ticks @ control_period=0.002.
  constexpr int    ZERO_TICKS   = 250;
  constexpr int    TOTAL_TICKS  = 1500;   // 250 ZERO + 1250 WALK, 3.0s sim time @ dt=0.002
  constexpr int    SETTLE_TICKS = 10;     // skip first few ticks before recording min height
  constexpr double HEIGHT_FLOOR = 0.4;
  constexpr double WALK_VX      = 0.4;

  std::printf("[teleop_smoke] policy=%s\n", POLICY_PATH);
  std::printf("[teleop_smoke] xml=%s\n",    XML_PATH);

  MujocoBackend backend(XML_PATH);   // keyframe-0 standing pose

  TaskManager manager;
  manager.add(std::make_unique<ZeroState>());
  manager.add(std::make_unique<WalkTask>(POLICY_PATH));

  DataPackage io;
  io.control_period = 0.002;
  manager.start(TaskId::ZERO, io);   // power-on: captures pose on 1st run()

  const double start_x = backend.baseX();
  double min_height = 1e9;
  bool   saw_walk    = false;

  for (int tick = 0; tick < TOTAL_TICKS; ++tick) {
    DriveCmd cmd;
    if (tick < ZERO_TICKS) {
      cmd.model = 1;                 // ZERO / stand
    } else {
      cmd.model = 2;                 // WALK
      cmd.vx    = static_cast<float>(WALK_VX);
    }

    ApplyCmdAndTick(cmd, io, manager, backend);

    if (manager.current() == TaskId::WALK) saw_walk = true;

    if (tick >= SETTLE_TICKS) {
      double h = backend.baseHeight();
      if (h < min_height) min_height = h;
    }
  }

  double x_disp = backend.baseX() - start_x;

  std::printf("[teleop_smoke] saw_walk=%d final_task=%s min_height=%.4f x_disp=%.4f\n",
              saw_walk, manager.current() == TaskId::WALK ? "WALK" : "other",
              min_height, x_disp);

  CHECK(saw_walk);                          // task switched to WALK via the command
  CHECK(manager.current() == TaskId::WALK);
  CHECK(min_height > HEIGHT_FLOOR);         // robot did not fall
  CHECK(x_disp > 0.0);                      // base advanced in +x while walking

  if (g_check_failures == 0) std::printf("TELEOP SMOKE PASSED\n");
  return g_check_failures == 0 ? 0 : 1;
}
