#pragma once
#include <array>
#include "phybot/task_id.h"
#include "phybot/robot_consts.h"
namespace phybot {
// 黑板。A0 起始最小, A1-num 扩展 cpg 走路所需字段。
struct DataPackage {
  TaskId requested_task{TaskId::ZERO};
  TaskId current_task{TaskId::ZERO};

  // —— 传感器 (task 读) ——
  std::array<double, 3>         imu_angular_vel{};
  std::array<double, 3>         imu_zyx{};            // ZYX euler
  std::array<double, kNumDofs>  motor_pos{};
  std::array<double, kNumDofs>  motor_vel{};
  double js_vx_desire{0.0}, js_vy_desire{0.0}, js_OmegaZ_desire{0.0};
  double control_period{0.002};   // dt (s); set by backend/main; used by ZeroState ramp

  // —— 控制输出 (task 写) ——
  std::array<double, kNumDofs>  motor_Pos_desire{};
  std::array<double, kNumDofs>  motor_Vel_desire{};
  std::array<double, kNumDofs>  motor_Torque_desire{};
  std::array<double, kNumDofs>  torq_desire{};
  std::array<double, kNumDofs>  sim_P{};
  std::array<double, kNumDofs>  sim_D{};
};
}  // namespace phybot
