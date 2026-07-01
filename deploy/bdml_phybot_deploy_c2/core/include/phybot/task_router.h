#pragma once
#include <array>
#include "phybot/task_code.h"
#include "phybot/controller_config.h"
namespace phybot { struct DriveCmd; struct DataPackage;
class TaskRouter {
 public:
  void addTask(TaskCode code, const ControllerConfig& cc);
  TaskCode route(TaskCode code) const;
  TaskCode defaultCode() const { return default_; }
  void shape(TaskCode active, const DriveCmd& cmd, DataPackage& io) const;
 private:
  std::array<bool, kMaxTaskCode> has_{};
  std::array<ControllerConfig, kMaxTaskCode> cfg_{};
  TaskCode default_{kNoTask};
};
}
