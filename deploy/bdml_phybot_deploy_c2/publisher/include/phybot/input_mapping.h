#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include "phybot/controller_state.h"
#include "phybot/input_protocol.h"
namespace phybot {
// One button -> DriveCmd.model binding: `field` selects which ControllerState
// button drives it (a pointer-to-member, e.g. &ControllerState::b), `code` is
// the select_code to latch on that button's rising edge, `prev` is this
// bind's own edge-detection state (one per bind -- replaces the old hardcoded
// prev_b/prev_y pair so ANY button can be bound, not just B/Y).
struct ButtonBind { bool ControllerState::* field; int code; bool prev; };

// binds is intentionally empty by default: this header stays yaml-cpp-free (see
// below), so it has no opinion on which buttons matter. The publisher app
// populates it from publisher/config/buttons.yaml at startup (via
// BuildButtonBinds() -- see buttons_config.h, kept OUT of this header so it
// stays yaml-free). Callers that don't touch yaml (e.g. tests) set binds
// directly, e.g. {{&ControllerState::b, 2, false}, {&ControllerState::y, 1, false}}.
struct MappingState {
  int8_t model = 1;
  std::vector<ButtonBind> binds;
};
inline float ip_dz(float v) { return (std::fabs(v) < 0.08f) ? 0.0f : v; }
inline DriveCmd MapControllerToDrive(const ControllerState& s, MappingState& m) {
  DriveCmd c;
  c.model = m.model;  // initialize from the retained state, as the current code does
  for (auto& b : m.binds) {
    bool now = s.*(b.field);
    if (now && !b.prev) c.model = (int8_t)b.code;  // rising edge -> latch this bind's code
    b.prev = now;
  }
  m.model = c.model;  // persist the latch across calls (mirrors the old m.model write)
  // NORMALIZED to [-1,1] (deadzone only); physical m/s scaling now applied task-side
  // by the harness (per-task controller.yaml AxisRange -- see controller_config.h).
  c.vx = -ip_dz(s.left_y);   // stick up (-y) -> forward
  c.vy =  ip_dz(s.left_x);
  c.wz = -ip_dz(s.right_x);
  return c;
}
}  // namespace phybot
