#pragma once
// Loads publisher/config/buttons.yaml (button -> DriveCmd.model select-code).
// Kept OUT of input_mapping.h: that header is inline/yaml-cpp-free (see its own
// comment) since it's included by every consumer of MapControllerToDrive,
// including the yaml-free test_input_mapping. Only main_publisher.cpp (the app)
// links this translation unit and calls load_buttons_config() once at startup.
#include <map>
#include <string>
namespace phybot {
struct ButtonsConfig {
  std::map<std::string, int> code;
  int code_for(const std::string& b, int fb) const {
    auto it = code.find(b);
    return it == code.end() ? fb : it->second;
  }
};
ButtonsConfig load_buttons_config();
}  // namespace phybot
