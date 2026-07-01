#pragma once
#include <array>
#include "phybot/robot_consts.h"
namespace phybot {
// [SIM/policy config] This struct is populated from clean YAML, NOT hardcoded here:
//   config/robot.yaml               -> default_dof_pos (default_pose; single source of truth)
//   tasks/walk/config/walk.yaml     -> scales, action_scale, rl_p (sim_kp), rl_d (sim_kd)
// rl_p/rl_d are the SIM actuator PD gains fed to mujoco (DataPackage.sim_P/sim_D). The REAL
// robot uses the motor-driver PD (official MotorList config) and is NOT configured here.
struct WalkConfig {
  double ang_vel_scale, dof_pos_scale, dof_vel_scale, lin_vel_scale, walk_flag_threshold;
  std::array<double, kNumDofs> action_scale;
  std::array<double, kNumDofs> default_dof_pos;
  std::array<double, kNumDofs> rl_p;
  std::array<double, kNumDofs> rl_d;
};
// Values equal 官方 RL_deploy_cpg/config/rl_params.yaml, now loaded at runtime from this
// tree's own config/robot.yaml + tasks/walk/config/walk.yaml — see walk_config.cpp.
// yaml-cpp is intentionally kept OUT of this header (only walk_config.cpp includes it),
// so downstream headers (walk_obs.h, walk_action.h, walk_history.h, walk_task.h, ...) and
// everything that merely needs the WalkConfig struct stay free of the yaml-cpp dependency.
WalkConfig walk_config();
}  // namespace phybot
