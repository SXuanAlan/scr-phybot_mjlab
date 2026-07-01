#pragma once
#include <array>
#include "phybot/task.h"
#include "phybot/robot_consts.h"
namespace phybot {
// 官方 ZeroState 逐行移植:首次 run() 捕获实际位姿 init_pose_,再用三次多项式(CubicTraj)
// 从 init_pose_ 平滑坡到 zero_config().zero_pose,以 zero_p/zero_d 刚性保持。
class ZeroState : public Task {
 public:
  ZeroState() : Task(TaskId::ZERO, "zero") {}
  void enter(DataPackage& io) override;   // arm first-run capture, reset t
  void run(DataPackage& io) override;     // first run captures init_pose; then cubic ramp
 private:
  std::array<double, kNumDofs> init_pose_{};
  double t_{0.0};
  bool first_{true};   // capture init_pose on first run AFTER pose is read (官方在读位后捕获)
};
}  // namespace phybot
