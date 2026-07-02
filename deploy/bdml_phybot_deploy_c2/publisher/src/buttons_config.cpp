#include "phybot/buttons_config.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
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

// Button-name -> ControllerState member-pointer table. Lives here (not the
// yaml-free input_mapping.h/buttons_config.h) since it's only needed to
// translate buttons.yaml's string keys into ButtonBinds.
bool ControllerState::* ButtonField(const std::string& name) {
  std::string u = name;
  std::transform(u.begin(), u.end(), u.begin(),
                 [](unsigned char ch) { return std::toupper(ch); });
  if (u == "A")     return &ControllerState::a;
  if (u == "B")     return &ControllerState::b;
  if (u == "X")     return &ControllerState::x;
  if (u == "Y")     return &ControllerState::y;
  if (u == "LB")    return &ControllerState::lb;
  if (u == "RB")    return &ControllerState::rb;
  if (u == "BACK")  return &ControllerState::back;
  if (u == "START") return &ControllerState::start;
  return nullptr;
}

std::vector<ButtonBind> BuildButtonBinds(const ButtonsConfig& bc) {
  std::vector<ButtonBind> binds;
  for (const auto& kv : bc.code) {
    if (bool ControllerState::* field = ButtonField(kv.first))
      binds.push_back({field, kv.second, false});
  }
  return binds;
}

}  // namespace phybot
