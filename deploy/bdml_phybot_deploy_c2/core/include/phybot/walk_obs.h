#pragma once
#include <array>
#include <cmath>
#include <cstring>
#include "phybot/robot_consts.h"
#include "phybot/walk_config.h"
namespace phybot {
struct WalkObsInputs {
  std::array<double, 3>        gravity{};
  std::array<double, 3>        ang_vel{};
  std::array<double, kNumDofs> q{};
  std::array<double, kNumDofs> dq{};
  std::array<double, kNumDofs> last_action{};
  double js_vx{0}, js_vy{0}, js_wz{0};
};

inline void BuildWalkObs(const WalkObsInputs& in, const WalkConfig& c,
                         const float* pos_newest105, const float* vel_newest105,
                         float* obs) {
  const double cmd0 = in.js_vx * c.lin_vel_scale;
  const double cmd1 = in.js_vy * c.lin_vel_scale;
  const double cmd2 = in.js_wz * c.ang_vel_scale;
  for (int i = 0; i < 3; ++i) obs[i] = (float)(in.ang_vel[i] * c.ang_vel_scale);  // [0:3]
  for (int i = 0; i < 3; ++i) obs[3 + i] = (float)in.gravity[i];                   // [3:6]
  obs[6] = (float)cmd0; obs[7] = (float)cmd1; obs[8] = (float)cmd2;                // [6:9]
  obs[9] = (std::fabs(cmd0) < c.walk_flag_threshold) ? 0.0f : 1.0f;               // [9]
  for (int i = 0; i < kNumDofs; ++i)
    obs[10 + i] = (float)((in.q[i] - c.default_dof_pos[i]) * c.dof_pos_scale);     // [10:31]
  for (int i = 0; i < kNumDofs; ++i)
    obs[10 + kNumDofs + i] = (float)(in.dq[i] * c.dof_vel_scale);                  // [31:52]
  for (int i = 0; i < kNumDofs; ++i)
    obs[10 + 2 * kNumDofs + i] = (float)in.last_action[i];                         // [52:73]
  std::memcpy(obs + kProprio, pos_newest105, sizeof(float) * kHistFramesInObs * kNumDofs);                 // [73:178]
  std::memcpy(obs + kProprio + kHistFramesInObs * kNumDofs, vel_newest105, sizeof(float) * kHistFramesInObs * kNumDofs);           // [178:283]
}
}  // namespace phybot
