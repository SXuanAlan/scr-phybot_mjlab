#include "zero/zero_task.h"
#include "zero/zero_config.h"
#include "phybot/cubic_traj.h"
#include "phybot/data_package.h"
#include "phybot/task_registry.h"
namespace phybot {
void ZeroState::enter(DataPackage&) {           // 武装首次捕获(不在此读位姿——start() 早于 backend 读取)
  first_ = true;
  t_ = 0.0;
}
void ZeroState::run(DataPackage& io) {
  static const ZeroConfig z = zero_config();
  static const std::array<double,kNumDofs> zerov{};  // 边界速度 0
  if (first_) {                                  // 官方 If_first_Zero: 读到实际位姿后捕获
    for (int i=0;i<kNumDofs;++i) init_pose_[i] = io.motor_pos[i];
    first_ = false;
  }
  t_ += io.control_period;
  std::array<double,kNumDofs> pos, vel;
  CubicTraj(init_pose_.data(), zerov.data(), z.zero_pose.data(), zerov.data(),
            z.zero_totalTime, t_, pos.data(), vel.data(), kNumDofs);
  for (int i=0;i<kNumDofs;++i) {
    io.motor_Pos_desire[i]    = pos[i];
    io.motor_Vel_desire[i]    = vel[i];
    io.motor_Torque_desire[i] = 0.0;
    io.torq_desire[i]         = 0.0;
    io.sim_P[i]               = z.zero_p[i];
    io.sim_D[i]               = z.zero_d[i];
  }
}
REGISTER_TASK("zero", ZeroState)
}  // namespace phybot
