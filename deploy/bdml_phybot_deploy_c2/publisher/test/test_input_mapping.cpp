#include "phybot/controller_state.h"
#include "phybot/input_mapping.h"
#include "check.h"
#include <cmath>
#include <cstdio>
static bool fa(float a, float b){ return std::fabs(a-b) < 1e-5f; }

static void test_button_edge_switch() {
  using namespace phybot;
  MappingState m;  ControllerState s{};
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
static void test_default_codes() {
  using namespace phybot;
  MappingState m;  ControllerState s{};
  CHECK(m.zero_code == 1);   // safe default, matches zero task's select_code
  CHECK(m.walk_code == 2);   // safe default, matches walk task's select_code
  s.b = true;  CHECK(MapControllerToDrive(s, m).model == m.walk_code);
  s.b = false; s.y = true; CHECK(MapControllerToDrive(s, m).model == m.zero_code);
}
int main(){ test_button_edge_switch(); test_axes_deadzone_scale(); test_default_codes();
  if(g_check_failures==0) std::printf("ALL INPUT MAPPING TESTS PASSED\n");
  return g_check_failures==0?0:1; }
