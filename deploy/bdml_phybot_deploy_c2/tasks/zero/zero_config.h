#pragma once
#include <array>
#include "phybot/robot_consts.h"
namespace phybot {
// [SIM config] For the upcoming ZeroState task. Populated from clean YAML, NOT hardcoded here:
//   config/robot.yaml               -> zero_pose (default_pose; single source of truth, shared
//                                       with WalkConfig::default_dof_pos -- do not redeclare it)
//   tasks/zero/config/zero.yaml     -> zero_p (sim_kp), zero_d (sim_kd), zero_totalTime (ramp_total_time)
// zero_p/zero_d are the SIM stiff-hold PD gains fed to mujoco (DataPackage.sim_P/sim_D). The
// REAL robot uses the motor-driver PD (~100/5) and is NOT configured here.
struct ZeroConfig {
  std::array<double, kNumDofs> zero_pose, zero_p, zero_d;
  double zero_totalTime;
};
// Loaded at runtime from config/robot.yaml + tasks/zero/config/zero.yaml — see zero_config.cpp.
// yaml-cpp is intentionally kept OUT of this header (only zero_config.cpp includes it), so
// downstream headers stay free of the yaml-cpp dependency.
ZeroConfig zero_config();
}  // namespace phybot
