#pragma once
#include <cstdint>
#include <string>
#include <algorithm>
#include <cctype>
#include "phybot/controller_state.h"
namespace phybot {

inline bool NameMatchesController(const std::string& name) {
  std::string n = name;
  std::transform(n.begin(), n.end(), n.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return n.find("xbox") != std::string::npos ||
         n.find("x-box") != std::string::npos ||
         n.find("microsoft") != std::string::npos;
}

// type: JS_EVENT_BUTTON(0x01) / JS_EVENT_AXIS(0x02), possibly OR'd with JS_EVENT_INIT(0x80).
inline void ApplyJsEvent(uint8_t type, uint8_t number, int16_t value, ControllerState& st) {
  type &= ~0x80;  // strip JS_EVENT_INIT
  if (type == 0x01) {                       // BUTTON
    const bool on = value != 0;
    switch (number) {
      case 0: st.a = on; break;     case 1: st.b = on; break;
      case 2: st.x = on; break;     case 3: st.y = on; break;
      case 4: st.lb = on; break;    case 5: st.rb = on; break;
      case 6: st.back = on; break;  case 7: st.start = on; break;
      case 8: st.guide = on; break; case 9: st.ls = on; break;
      case 10: st.rs = on; break;   default: break;  // unknown -> ignore
    }
  } else if (type == 0x02) {                // AXIS
    float v = (float)value / 32767.0f;
    if (v > 1.0f) v = 1.0f; else if (v < -1.0f) v = -1.0f;
    switch (number) {
      case 0: st.left_x = v; break;   case 1: st.left_y = v; break;
      case 2: st.lt = v; break;       case 3: st.right_x = v; break;
      case 4: st.right_y = v; break;  case 5: st.rt = v; break;
      case 6: st.dpad_x = v; break;   case 7: st.dpad_y = v; break;
      default: break;
    }
  }
}
}  // namespace phybot
