#include "phybot/buttons_config.h"

#include <yaml-cpp/yaml.h>

#include <string>

// PHYBOT_ROOT is injected by CMake via target_compile_definitions on main_publisher
// (same interim pattern as walk_config.cpp / controller_config.cpp -- see CMakeLists.txt).
#ifndef PHYBOT_ROOT
#error "PHYBOT_ROOT must be defined"
#endif

namespace phybot {

ButtonsConfig load_buttons_config() {
  ButtonsConfig c;
  const YAML::Node y = YAML::LoadFile(std::string(PHYBOT_ROOT) + "/publisher/config/buttons.yaml");
  for (auto it = y.begin(); it != y.end(); ++it) {
    c.code[it->first.as<std::string>()] = it->second.as<int>();
  }
  return c;
}

}  // namespace phybot
