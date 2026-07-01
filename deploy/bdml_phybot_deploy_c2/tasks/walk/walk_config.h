#pragma once
#include <array>
#include <string>
#include "phybot/robot_consts.h"
namespace phybot {
// [SIM/policy config] This struct is populated from clean YAML, NOT hardcoded here:
//   config/robot.yaml               -> default_dof_pos (default_pose; single source of truth)
//   tasks/walk/config/walk.yaml     -> scales, action_scale, rl_p (sim_kp), rl_d (sim_kd),
//                                       policy_path (resolved to an absolute path)
// rl_p/rl_d are the SIM actuator PD gains fed to mujoco (DataPackage.sim_P/sim_D). The REAL
// robot uses the motor-driver PD (official MotorList config) and is NOT configured here.
struct WalkConfig {
  double ang_vel_scale, dof_pos_scale, dof_vel_scale, lin_vel_scale, walk_flag_threshold;
  std::array<double, kNumDofs> action_scale;
  std::array<double, kNumDofs> default_dof_pos;
  std::array<double, kNumDofs> rl_p;
  std::array<double, kNumDofs> rl_d;
  std::string policy_path;  // absolute path to cpg_policy.pt (resolved via ResolveRoot)
};
// Values equal 官方 RL_deploy_cpg/config/rl_params.yaml, now loaded at runtime from this
// tree's own config/robot.yaml + tasks/walk/config/walk.yaml — see walk_config.cpp.
// yaml-cpp is intentionally kept OUT of this header (only walk_config.cpp includes it),
// so downstream headers (walk_obs.h, walk_action.h, walk_history.h, walk_task.h, ...) and
// everything that merely needs the WalkConfig struct stay free of the yaml-cpp dependency.
WalkConfig walk_config();

// Resolves a path from YAML against the source-tree root: absolute paths (leading '/') pass
// through unchanged; relative paths are prefixed with PHYBOT_ROOT (injected by CMake via
// target_compile_definitions on every target compiling walk_config.cpp -- see CMakeLists.txt),
// so config-driven paths (e.g. policy_path) resolve the same regardless of the build/run cwd.
std::string ResolveRoot(const std::string& path);
}  // namespace phybot
