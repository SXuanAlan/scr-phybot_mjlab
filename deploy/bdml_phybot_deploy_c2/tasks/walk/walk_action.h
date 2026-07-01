#pragma once
#include <array>
#include "phybot/robot_consts.h"
#include "walk/walk_config.h"
#include "phybot/data_package.h"
namespace phybot {
inline void ApplyWalkAction(const std::array<double, kNumDofs>& action_raw,
                            const WalkConfig& c, DataPackage& io) {
  for (int i = 0; i < kNumDofs; ++i) {
    io.motor_Pos_desire[i]     = action_raw[i] * c.action_scale[i] + c.default_dof_pos[i];
    io.motor_Vel_desire[i]     = 0.0;
    io.motor_Torque_desire[i]  = 0.0;
    io.torq_desire[i]          = 0.0;
    io.sim_P[i]                = c.rl_p[i];
    io.sim_D[i]                = c.rl_d[i];
  }
}
}  // namespace phybot
