// test_teleop_smoke: headless end-to-end gate — command -> task-switch (ZERO->WALK)
// -> live MuJoCo sim, driven entirely through SimHarness (see
// backends/mujoco/sim_harness.h). NO LCM in the loop: this sandbox has no
// multicast route and the vendored liblcm ABORTS (buffer overflow) inside
// handleTimeout() when used headless (see the test_lcm_loopback note in
// CMakeLists.txt). So instead of publishing over LCM, this test sets a DriveCmd
// directly in-process and feeds it to SimHarness::tick(), the exact same
// route->shape->read->manager.tick->write->step loop apps/sim_teleop.cpp drives
// from LCM-decoded commands.
//
// This is also the first end-to-end exercise of config-driven self-registration:
// SimHarness's ctor iterates TaskRegistry and pairs each self-registered task
// (constructed via its name-only factory, e.g. WalkTask()) with its own
// tasks/<name>/config/controller.yaml. WalkTask::enter() then resolves its policy
// path from tasks/walk/config/walk.yaml (via WalkConfig) -- nobody here injects a
// concrete model_path. If the config-driven wiring is broken, WalkTask either
// fails to load its policy (throws) or the robot falls once WALK engages.
#include "sim_harness.h"
#include "phybot/input_protocol.h"
#include <cstdio>

// Path injected by CMake (mirrors test_walk_sim's XML_PATH pattern). SimHarness
// resolves each task's policy path itself (from that task's own config), so no
// POLICY_PATH is needed here.
#ifndef XML_PATH
#define XML_PATH "assets/robot/phybot_c2/xml/phybot_c2.xml"
#endif

int main() {
  using namespace phybot;

  SimHarness h({XML_PATH});

  // ZERO / stand (select_code=1, tasks/zero/config/controller.yaml's default task).
  DriveCmd zero;
  zero.model = 1;
  for (int i = 0; i < 200; ++i) h.tick(zero);
  double h_zero = h.backend().baseHeight();

  // WALK (select_code=2, tasks/walk/config/controller.yaml) with forward vx.
  DriveCmd walk;
  walk.model = 2;
  walk.vx    = 0.6f;
  for (int i = 0; i < 1500; ++i) h.tick(walk);
  double h_walk = h.backend().baseHeight();

  bool ok = h_zero > 0.4 && h_walk > 0.4;
  std::printf("h_zero=%.3f h_walk=%.3f -> %s\n", h_zero, h_walk, ok ? "OK" : "FALL");
  return ok ? 0 : 1;
}
