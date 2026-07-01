// sim_teleop: real-host teleop main. Subscribes Robot_ControlData over LCM and drives
// a live headless MuJoCo simulation via the shared ApplyCmdAndTick() loop (see
// backends/mujoco/teleop_loop.h) -- the exact same loop body that
// core/test/test_teleop_smoke.cpp exercises without LCM.
//
// NOT run in ctest: requires a working LCM multicast route on the host (see the
// test_lcm_loopback note in CMakeLists.txt -- the vendored liblcm aborts inside
// handleTimeout() in a headless sandbox with no multicast route). This is a
// real-host tool, like main_publisher.
//
// Live two-process demo:
//   terminal 1: ./main_publisher     (Xbox controller -> Robot_ControlData over LCM)
//   terminal 2: ./sim_teleop         (subscribes -> drives the live sim)
//   B -> WALK, left stick forward -> vx, Y -> back to ZERO/stand.
//
// Usage: ./sim_teleop [cpg_policy.pt] [phybot_c2.xml]
#include "phybot/data_package.h"
#include "phybot/task_manager.h"
#include "phybot/task_id.h"
#include "phybot/zero_task.h"
#include "phybot/input_protocol.h"
#include "phybot/lcm_env.h"
#include "walk/walk_task.h"
#include "mujoco_backend.h"
#include "teleop_loop.h"
#include <lcm/lcm-cpp.hpp>
#include <cstdio>
#include <ctime>
#include <memory>
#include <string>
#include <vector>

#ifdef PHYBOT_ENABLE_GUI
#include "mujoco_viewer.h"
#include <thread>
#include <mutex>
#endif

// Paths injected by CMake (mirrors test_walk_sim's POLICY_PATH/XML_PATH pattern);
// overridable via argv for ad hoc runs.
#ifndef POLICY_PATH
#define POLICY_PATH "core/test/golden/cpg_policy.pt"
#endif
#ifndef XML_PATH
#define XML_PATH "assets/robot/phybot_c2/xml/phybot_c2.xml"
#endif

namespace {

// Holds the most recently decoded DriveCmd. handleTimeout(0) is polled from the
// single control-loop thread (never blocks), so the LCM callback and the loop body
// below never run concurrently -- no locking needed.
struct CmdHolder {
  phybot::DriveCmd latest;  // default: model=0 -> ZERO (safe power-on state)
  void onControlData(const lcm::ReceiveBuffer*, const std::string&,
                      const custom_lcm::system2Algorithm_packet* p) {
    phybot::DriveCmd decoded;
    if (phybot::ip_decode(*p, decoded)) latest = decoded;
  }
};

const char* TaskName(phybot::TaskId id) {
  switch (id) {
    case phybot::TaskId::ZERO: return "ZERO";
    case phybot::TaskId::WALK: return "WALK";
    default:                   return "?";
  }
}

}  // namespace

int main(int argc, char** argv) {
  using namespace phybot;

  bool gui = false;
  std::vector<const char*> pos;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--gui") gui = true;
    else pos.push_back(argv[i]);
  }
  const char* policy_path = (pos.size() > 0 && pos[0][0]) ? pos[0] : POLICY_PATH;
  const char* xml_path    = (pos.size() > 1 && pos[1][0]) ? pos[1] : XML_PATH;
#ifndef PHYBOT_ENABLE_GUI
  if (gui) std::printf("[sim_teleop] (built without GUI; running headless)\n");
  gui = false;
#endif

  AnnounceLcmInit("sim_teleop");
  lcm::LCM lcm;
  if (!lcm.good()) {
    std::printf("[sim_teleop] LCM init failed (no multicast route?)\n");
    return 1;
  }

  CmdHolder holder;
  lcm.subscribe("Robot_ControlData", &CmdHolder::onControlData, &holder);

  std::printf("[sim_teleop] policy=%s\n", policy_path);
  std::printf("[sim_teleop] xml=%s\n",    xml_path);

  MujocoBackend backend(xml_path);   // keyframe-0 standing pose

  TaskManager manager;
  manager.add(std::make_unique<ZeroState>());
  manager.add(std::make_unique<WalkTask>(policy_path));

  DataPackage io;
  io.control_period = 0.002;
  manager.start(TaskId::ZERO, io);   // power-on: captures pose on 1st run()

  // One iteration: drain LCM (updates holder.latest) then advance the FSM + sim.
  auto control_tick = [&]{
    lcm.handleTimeout(0);
    ApplyCmdAndTick(holder.latest, io, manager, backend);
  };

#ifdef PHYBOT_ENABLE_GUI
  if (gui) {
    MujocoViewer viewer("sim_teleop");
    std::thread physics([&]{
      viewer.loadModel(backend.model(), backend.data());
      const timespec dt{0, 2 * 1000 * 1000};
      while (!viewer.exitRequested()) {
        {
          std::lock_guard<std::recursive_mutex> lk(viewer.mutex());
          if (viewer.running()) { control_tick(); viewer.trackCamera(); }
          else { lcm.handleTimeout(0); mj_forward(backend.model(), backend.data()); }
        }
        nanosleep(&dt, nullptr);
      }
    });
    std::printf("[sim_teleop] GUI: B=WALK, Y=ZERO, sticks=vx/vy/wz; space=pause, ESC/close=quit\n");
    viewer.renderLoop();   // blocks on the main thread until the window closes
    physics.join();
    return 0;
  }
#endif

  const timespec period{0, 2 * 1000 * 1000};  // 2ms -> real-time pacing at control_period
  const long kPrintEveryTicks = 250;          // 250 * 0.002s = 0.5s of sim time

  std::printf("[sim_teleop] running; waiting for Robot_ControlData (model 1=ZERO, 2=WALK)...\n");
  for (long tick = 0;; ++tick) {
    control_tick();

    if (tick % kPrintEveryTicks == 0) {
      std::printf("[sim_teleop] t=%.3fs task=%s height=%.4f x=%.4f\n",
                  backend.simTime(), TaskName(manager.current()),
                  backend.baseHeight(), backend.baseX());
    }

    nanosleep(&period, nullptr);
  }
}
