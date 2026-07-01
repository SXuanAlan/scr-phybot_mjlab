#pragma once
// SimHarness: glfw-free composition root that pairs the (self-registering) task
// registry with each task's own controller.yaml, then drives a headless MuJoCo
// sim through TaskRouter (command shaping) + TaskManager (task switch/tick).
//
// This is the FIRST place config-driven self-registration is exercised end-to-end:
// the ctor iterates TaskRegistry, constructs each task via its name-only factory
// (e.g. WalkTask() -- see tasks/walk/walk_task.cpp), and registers it under the
// select_code read from that task's own tasks/<name>/config/controller.yaml. No
// caller here ever names a concrete Task subclass or injects a policy path.
#include <functional>
#include <string>
#include "mujoco_backend.h"
#include "phybot/task_manager.h"
#include "phybot/task_router.h"
#include "phybot/data_package.h"

namespace phybot {

struct DriveCmd;

class SimHarness {
 public:
  struct Options {
    std::string xml_path;
  };

  explicit SimHarness(const Options& opt);

  // One control-loop step: route the command -> shape axes into io_ -> read
  // sensors -> tick the active task -> write actuator targets -> step physics.
  void tick(const DriveCmd& cmd);

  // Headless real-time-paced run loop (opt-in; tick() alone is enough for tests).
  using CommandSource = std::function<DriveCmd()>;
  void runHeadless(CommandSource src, double seconds, std::function<bool()> stop = {});

  MujocoBackend& backend() { return backend_; }
  TaskManager&   manager() { return manager_; }

 private:
  MujocoBackend backend_;
  TaskManager   manager_;
  TaskRouter    router_;
  DataPackage   io_{};
};

}  // namespace phybot
