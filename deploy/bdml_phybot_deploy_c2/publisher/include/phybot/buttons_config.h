#pragma once
// Loads publisher/config/buttons.yaml (button -> DriveCmd.model select-code) and
// turns it into the ButtonBind list MappingState::binds expects.
// Kept OUT of input_mapping.h: that header is inline/yaml-cpp-free (see its own
// comment) since it's included by every consumer of MapControllerToDrive,
// including the yaml-free test_input_mapping. Only main_publisher.cpp (the app)
// links this translation unit and calls load_buttons_config() once at startup.
#include <map>
#include <string>
#include <vector>
#include "phybot/controller_state.h"
#include "phybot/input_mapping.h"
namespace phybot {
struct ButtonsConfig {
  std::map<std::string, int> code;
  int code_for(const std::string& b, int fb) const {
    auto it = code.find(b);
    return it == code.end() ? fb : it->second;
  }
};
ButtonsConfig load_buttons_config();

// Button-name (A/B/X/Y/LB/RB/BACK/START, case-insensitive) -> ControllerState
// member pointer, e.g. ButtonField("y") == &ControllerState::y. Unknown name ->
// nullptr. The name table lives in buttons_config.cpp (not here, and not in
// input_mapping.h) since it exists purely to translate buttons.yaml's string
// keys into ButtonBinds.
bool ControllerState::* ButtonField(const std::string& name);

// Turns a loaded ButtonsConfig into the ButtonBind list MappingState::binds
// expects, one bind per recognized button (unknown names skipped via
// ButtonField). This is what makes buttons.yaml fully drive the mapping: any
// button listed there binds to its code with no changes to input_mapping.h or
// main_publisher.cpp.
std::vector<ButtonBind> BuildButtonBinds(const ButtonsConfig& bc);
}  // namespace phybot
