#pragma once
#include <array>
#include "phybot/robot_consts.h"
namespace phybot {
struct WalkConfig {
  double ang_vel_scale, dof_pos_scale, dof_vel_scale, lin_vel_scale, walk_flag_threshold;
  std::array<double, kNumDofs> action_scale;
  std::array<double, kNumDofs> default_dof_pos;
  std::array<double, kNumDofs> rl_p;
  std::array<double, kNumDofs> rl_d;
};
// 官方 RL_deploy_cpg/config/rl_params.yaml 的值 (硬编码; A1-policy 改为从 yaml 载入)。
inline WalkConfig walk_config() {
  WalkConfig c;
  c.ang_vel_scale = 0.25; c.dof_pos_scale = 1.0; c.dof_vel_scale = 0.05;
  c.lin_vel_scale = 1.0;  c.walk_flag_threshold = 0.3;
  c.action_scale = {0.2,0.2,0.2,0.2,0.2,0.2, 0.2,0.2,0.2,0.2,0.2,0.2, 0.2,
                    0.2,0.2,0.2,0.2, 0.2,0.2,0.2,0.2};
  c.default_dof_pos = {-0.242,0.029,0.007,0.49,-0.295,0.0,
                       -0.242,-0.029,-0.007,0.49,-0.295,0.0,
                       0.0,
                       0.0,0.0,0.0,-0.17,
                       0.0,-0.0,0.0,-0.17};
  c.rl_p = {100,100,100,100,50,50, 100,100,100,100,50,50, 20,
            50,50,50,50, 50,50,50,50};
  c.rl_d = {5,5,5,5,5,5, 5,5,5,5,5,5, 10, 5,5,5,5, 5,5,5,5};
  return c;
}
}  // namespace phybot
