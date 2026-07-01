// Loads WalkConfig from config/robot.yaml + tasks/walk/config/walk.yaml at runtime.
// [SIM/policy config] yaml-cpp is used only here -- kept out of walk_config.h so downstream
// headers stay clean (mirrors how policy_runner.h keeps Torch headers out of the public
// header; see policy_runner.cpp for the same pattern).
//
// CONFIG_DIR / WALK_CONFIG_DIR are injected by CMake via target_compile_definitions, the
// same interim pattern as GOLDEN_DIR (see CMakeLists.txt). yaml-cpp itself is wired in an
// interim way too, pointing at the official tree's ThirdParty/yaml -- both need proper
// vendoring into this tree's own third_party later.
#include "walk/walk_config.h"

#include <yaml-cpp/yaml.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#ifndef CONFIG_DIR
#error "CONFIG_DIR must be defined via target_compile_definitions (see CMakeLists.txt)"
#endif
#ifndef WALK_CONFIG_DIR
#error "WALK_CONFIG_DIR must be defined via target_compile_definitions (see CMakeLists.txt)"
#endif

namespace phybot {
namespace {

template <std::size_t N>
std::array<double, N> ToArray(const YAML::Node& node) {
  if (!node || !node.IsSequence() || node.size() != N) {
    throw std::runtime_error(
        "walk_config: expected a " + std::to_string(N) + "-element sequence");
  }
  std::array<double, N> out{};
  for (std::size_t i = 0; i < N; ++i) out[i] = node[i].as<double>();
  return out;
}

}  // namespace

WalkConfig walk_config() {
  const std::string robot_path = std::string(CONFIG_DIR) + "/robot.yaml";
  const std::string walk_path  = std::string(WALK_CONFIG_DIR) + "/walk.yaml";
  const YAML::Node robot = YAML::LoadFile(robot_path);
  const YAML::Node walk  = YAML::LoadFile(walk_path);

  WalkConfig c;
  c.ang_vel_scale       = walk["ang_vel_scale"].as<double>();
  c.dof_pos_scale       = walk["dof_pos_scale"].as<double>();
  c.dof_vel_scale       = walk["dof_vel_scale"].as<double>();
  c.lin_vel_scale       = walk["lin_vel_scale"].as<double>();
  c.walk_flag_threshold = walk["walk_flag_threshold"].as<double>();
  c.action_scale        = ToArray<kNumDofs>(walk["action_scale"]);
  c.default_dof_pos     = ToArray<kNumDofs>(robot["default_pose"]);
  c.rl_p                = ToArray<kNumDofs>(walk["sim_kp"]);
  c.rl_d                = ToArray<kNumDofs>(walk["sim_kd"]);
  return c;
}

}  // namespace phybot
