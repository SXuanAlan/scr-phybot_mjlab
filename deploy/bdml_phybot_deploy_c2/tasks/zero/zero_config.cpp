// Loads ZeroConfig from config/robot.yaml + tasks/zero/config/zero.yaml at runtime.
// [SIM config] yaml-cpp is used only here -- kept out of zero_config.h so downstream headers
// stay clean (same pattern as walk_config.cpp / policy_runner.cpp).
//
// CONFIG_DIR / ZERO_CONFIG_DIR are injected by CMake via target_compile_definitions, the
// same interim pattern as GOLDEN_DIR (see CMakeLists.txt). yaml-cpp itself is wired in an
// interim way too, pointing at the official tree's ThirdParty/yaml -- both need proper
// vendoring into this tree's own third_party later.
#include "zero/zero_config.h"

#include <yaml-cpp/yaml.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#ifndef CONFIG_DIR
#error "CONFIG_DIR must be defined via target_compile_definitions (see CMakeLists.txt)"
#endif
#ifndef ZERO_CONFIG_DIR
#error "ZERO_CONFIG_DIR must be defined via target_compile_definitions (see CMakeLists.txt)"
#endif

namespace phybot {
namespace {

template <std::size_t N>
std::array<double, N> ToArray(const YAML::Node& node) {
  if (!node || !node.IsSequence() || node.size() != N) {
    throw std::runtime_error(
        "zero_config: expected a " + std::to_string(N) + "-element sequence");
  }
  std::array<double, N> out{};
  for (std::size_t i = 0; i < N; ++i) out[i] = node[i].as<double>();
  return out;
}

}  // namespace

ZeroConfig zero_config() {
  const std::string robot_path = std::string(CONFIG_DIR) + "/robot.yaml";
  const std::string zero_path  = std::string(ZERO_CONFIG_DIR) + "/zero.yaml";
  const YAML::Node robot = YAML::LoadFile(robot_path);
  const YAML::Node zero  = YAML::LoadFile(zero_path);

  ZeroConfig c;
  c.zero_pose      = ToArray<kNumDofs>(robot["default_pose"]);
  c.zero_p         = ToArray<kNumDofs>(zero["sim_kp"]);
  c.zero_d         = ToArray<kNumDofs>(zero["sim_kd"]);
  c.zero_totalTime = zero["ramp_total_time"].as<double>();
  return c;
}

}  // namespace phybot
