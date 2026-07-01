// sim_teleop: real-host teleop main. Subscribes Robot_ControlData over LCM and
// drives a live MuJoCo sim through SimHarness::tick() (see
// backends/mujoco/sim_harness.h) -- the exact same route->shape->read->tick->
// write->step loop core/test/test_teleop_smoke.cpp exercises without LCM.
//
// NOT run in ctest: requires a working LCM multicast route on the host (see the
// test_lcm_loopback note in CMakeLists.txt -- the vendored liblcm aborts inside
// handleTimeout() in a headless sandbox with no multicast route). This is a
// real-host tool, like main_publisher.
//
// Live two-process demo:
//   terminal 1: ./main_publisher     (Xbox controller -> Robot_ControlData over LCM)
//   terminal 2: ./sim_teleop         (subscribes -> drives the live sim)
//   B -> WALK, left stick forward -> vx, Y -> back to ZERO/stand (button->select_code
//   is config-driven; see publisher/config/buttons.yaml).
//
// Usage: ./sim_teleop [xml_path] [--gui]
#include "sim_harness.h"
#include "sim_harness_gui.h"
#include "phybot/input_protocol.h"
#include "phybot/lcm_env.h"
#include <lcm/lcm-cpp.hpp>
#include <cstdio>
#include <string>
#include <vector>

// Asset path is baked in by CMake (absolute, vendored -- CWD-independent). This
// #ifndef fallback is relative to the repo root, for a manual build without CMake;
// run such a build from deploy/bdml_phybot_deploy_c2/.
#ifndef XML_PATH
#define XML_PATH "assets/robot/phybot_c2/xml/phybot_c2.xml"
#endif

namespace {

// Holds the most recently decoded DriveCmd. handleTimeout(0) is polled from the
// single control-loop thread (never blocks), so the LCM callback and
// SimHarness::tick() never run concurrently -- no locking needed.
struct CmdHolder {
  phybot::DriveCmd latest;  // default: model=0 -> TaskRouter falls back to the default task (ZERO)
  void onControlData(const lcm::ReceiveBuffer*, const std::string&,
                      const custom_lcm::system2Algorithm_packet* p) {
    phybot::DriveCmd decoded;
    if (phybot::ip_decode(*p, decoded)) latest = decoded;
  }
};

}  // namespace

int main(int argc, char** argv) {
  using namespace phybot;

  // --gui may appear anywhere; the rest are positional [xml_path].
  bool gui = false;
  std::vector<const char*> pos;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--gui") gui = true;
    else pos.push_back(argv[i]);
  }
  // Empty positional ("") falls back to the default, so `--gui` alone works
  // without having to spell out the xml path.
  const char* xml_path = (pos.size() > 0 && pos[0][0]) ? pos[0] : XML_PATH;

  AnnounceLcmInit("sim_teleop");
  lcm::LCM lcm;
  if (!lcm.good()) {
    std::printf("[sim_teleop] LCM init failed (no multicast route?)\n");
    return 1;
  }

  CmdHolder holder;
  lcm.subscribe("Robot_ControlData", &CmdHolder::onControlData, &holder);

  std::printf("[sim_teleop] xml=%s\n", xml_path);

  SimHarness h({xml_path});
  auto src = [&]{ lcm.handleTimeout(0); return holder.latest; };

  if (gui) {
    std::printf("[sim_teleop] GUI: buttons.yaml selects task, sticks=vx/vy/wz; "
                "space=pause, ESC/close=quit\n");
    RunGui(h, src, "sim_teleop");
  } else {
    std::printf("[sim_teleop] running; waiting for Robot_ControlData "
                "(buttons.yaml: button -> DriveCmd.model select_code)...\n");
    h.runHeadless(src, 0);
  }
  return 0;
}
