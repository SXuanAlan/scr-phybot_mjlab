#pragma once
#include <array>
#include <memory>
#include <string>
#include "phybot/task.h"
#include "phybot/task_id.h"
#include "phybot/walk_config.h"
#include "phybot/walk_history.h"
#include "phybot/robot_consts.h"

namespace phybot {
class PolicyRunner;

class WalkTask : public Task {
 public:
  // model_path: path to cpg_policy.pt (injectable for tests)
  explicit WalkTask(const std::string& model_path = "");
  ~WalkTask() override;

  void enter(DataPackage& io) override;
  void run(DataPackage& io) override;
  void exit(DataPackage& io) override;

 private:
  std::string model_path_;
  std::unique_ptr<PolicyRunner> runner_;
  WalkHistory history_;
  WalkConfig cfg_;
  std::array<double, kNumDofs> last_action_{};
  int step_num_{0};
};
}  // namespace phybot
