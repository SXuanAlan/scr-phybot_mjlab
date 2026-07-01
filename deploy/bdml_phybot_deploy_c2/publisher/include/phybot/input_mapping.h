#pragma once
#include <cstdint>
#include <cmath>
#include "phybot/controller_state.h"
#include "phybot/input_protocol.h"
namespace phybot {
struct MappingState { bool prev_b=false, prev_y=false; int8_t model=1; };
inline float ip_dz(float v) { return (std::fabs(v) < 0.08f) ? 0.0f : v; }
inline DriveCmd MapControllerToDrive(const ControllerState& s, MappingState& m) {
  if (s.b && !m.prev_b) m.model = 2;   // B rising edge -> walk
  if (s.y && !m.prev_y) m.model = 1;   // Y rising edge -> zero
  m.prev_b = s.b; m.prev_y = s.y;
  DriveCmd c;
  c.model = m.model;
  c.vx = -ip_dz(s.left_y) * 0.6f;   // stick up (-y) -> forward
  c.vy =  ip_dz(s.left_x) * 0.1f;
  c.wz = -ip_dz(s.right_x) * 1.0f;
  return c;
}
}  // namespace phybot
