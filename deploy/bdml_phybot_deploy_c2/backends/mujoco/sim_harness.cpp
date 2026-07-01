#include "sim_harness.h"
#include "phybot/task_registry.h"
#include "phybot/controller_config.h"
#include "phybot/input_protocol.h"
#include <ctime>

namespace phybot {

SimHarness::SimHarness(const Options& opt) : backend_(opt.xml_path) {
  io_.control_period = 0.002;

  // Pair every self-registered task (TaskRegistry, populated by REGISTER_TASK
  // static-init side effects inside the whole-archived phybot_tasks) with its own
  // tasks/<name>/config/controller.yaml: select_code decides where TaskManager
  // files it, and the full ControllerConfig (incl. axis ranges + default flag)
  // goes to TaskRouter for command routing/shaping. Neither this ctor nor any
  // caller names a concrete Task subclass -- reg.make(i) is the only construction
  // point, driven entirely by what's in the registry + that task's yaml.
  auto& reg = TaskRegistry::instance();
  for (int i = 0; i < reg.size(); ++i) {
    ControllerConfig cc = load_controller_config(reg.name(i));
    manager_.add(reg.make(i), cc.select_code);
    router_.addTask(cc.select_code, cc);
  }
  manager_.start(router_.defaultCode(), io_);
}

void SimHarness::tick(const DriveCmd& cmd) {
  TaskCode code = router_.route(cmd.model);
  router_.shape(code, cmd, io_);
  backend_.readToPackage(io_);
  manager_.tick(io_);
  backend_.writeFromPackage(io_);
  backend_.step();
}

void SimHarness::runHeadless(CommandSource src, double seconds, std::function<bool()> stop) {
  const timespec period{0, 2 * 1000 * 1000};  // 2ms -> real-time pacing at control_period
  const long total = seconds > 0 ? static_cast<long>(seconds / 0.002) : -1;
  for (long t = 0; total < 0 || t < total; ++t) {
    if (stop && stop()) break;
    tick(src());
    nanosleep(&period, nullptr);
  }
}

}  // namespace phybot
