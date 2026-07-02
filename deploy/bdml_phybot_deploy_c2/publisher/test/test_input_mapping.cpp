#include "phybot/controller_state.h"
#include "phybot/input_mapping.h"
#include "check.h"
#include <cmath>
#include <cstdio>
static bool fa(float a, float b){ return std::fabs(a-b) < 1e-5f; }

// binds set up explicitly, yaml-free (mirrors what BuildButtonBinds(buttons.yaml)
// would produce for Y:1/B:2 -- see publisher/config/buttons.yaml -- without
// pulling yaml-cpp into this test).
static void test_button_edge_switch() {
  using namespace phybot;
  MappingState m;  ControllerState s{};
  m.binds = {{&ControllerState::b, 2, false}, {&ControllerState::y, 1, false}};
  CHECK(MapControllerToDrive(s, m).model == 1);   // default zero
  s.b = true;  CHECK(MapControllerToDrive(s, m).model == 2);  // B rising -> walk
  s.b = true;  CHECK(MapControllerToDrive(s, m).model == 2);  // held -> stays
  s.b = false; CHECK(MapControllerToDrive(s, m).model == 2);  // release -> stays walk
  s.y = true;  CHECK(MapControllerToDrive(s, m).model == 1);  // Y rising -> zero
}
static void test_axes_deadzone_scale() {
  using namespace phybot;
  MappingState m;  ControllerState s{};
  s.left_y = 0.05f;  CHECK(fa(MapControllerToDrive(s,m).vx, 0.0f));   // inside deadzone
  s.left_y = -1.0f;  CHECK(fa(MapControllerToDrive(s,m).vx, 1.0f));   // up -> +vx, NORMALIZED [-1,1]
  s.left_x = 1.0f;   CHECK(fa(MapControllerToDrive(s,m).vy, 1.0f));   // NORMALIZED [-1,1]
  s.right_x = 1.0f;  CHECK(fa(MapControllerToDrive(s,m).wz, -1.0f));  // NORMALIZED [-1,1]
}
// Proves the mapping is genuinely generic: a THIRD bind (X -> 3), which the old
// hardcoded B/Y-only MapControllerToDrive could never express, works with no
// code changes -- just another entry in binds (== what uncommenting `X: 3` in
// buttons.yaml now does via BuildButtonBinds).
static void test_generic_third_bind() {
  using namespace phybot;
  MappingState m;  ControllerState s{};
  m.binds = {{&ControllerState::b, 2, false}, {&ControllerState::y, 1, false},
             {&ControllerState::x, 3, false}};
  s.b = true;  CHECK(MapControllerToDrive(s, m).model == 2);  // B -> walk
  s.b = false; s.y = true;  CHECK(MapControllerToDrive(s, m).model == 1);  // Y -> zero
  s.y = false; s.x = true;  CHECK(MapControllerToDrive(s, m).model == 3);  // X -> 3, generic
}
int main(){ test_button_edge_switch(); test_axes_deadzone_scale(); test_generic_third_bind();
  if(g_check_failures==0) std::printf("ALL INPUT MAPPING TESTS PASSED\n");
  return g_check_failures==0?0:1; }
