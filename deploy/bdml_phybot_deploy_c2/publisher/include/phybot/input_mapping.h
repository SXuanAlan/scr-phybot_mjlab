#pragma once
#include <cstdint>
#include <cmath>
#include "phybot/controller_state.h"
#include "phybot/input_protocol.h"
namespace phybot {
// zero_code/walk_code default to the zero/walk tasks' own select_code (see
// tasks/zero/config/controller.yaml, tasks/walk/config/controller.yaml) so a default-
// constructed MappingState (e.g. in tests) behaves correctly with no yaml involved.
// The publisher app overrides them from publisher/config/buttons.yaml at startup
// (see buttons_config.h) -- kept OUT of this header so it stays yaml-cpp-free.
struct MappingState { bool prev_b=false, prev_y=false; int8_t model=1;
  int zero_code=1, walk_code=2; };
inline float ip_dz(float v) { return (std::fabs(v) < 0.08f) ? 0.0f : v; }
inline DriveCmd MapControllerToDrive(const ControllerState& s, MappingState& m) {
  if (s.b && !m.prev_b) m.model = (int8_t)m.walk_code;   // B rising edge -> walk
  if (s.y && !m.prev_y) m.model = (int8_t)m.zero_code;   // Y rising edge -> zero
  m.prev_b = s.b; m.prev_y = s.y;
  DriveCmd c;
  c.model = m.model;
  // NORMALIZED to [-1,1] (deadzone only); physical m/s scaling now applied task-side
  // by the harness (per-task controller.yaml AxisRange -- see controller_config.h).
  c.vx = -ip_dz(s.left_y);   // stick up (-y) -> forward
  c.vy =  ip_dz(s.left_x);
  c.wz = -ip_dz(s.right_x);
  return c;
}
}  // namespace phybot
