#include "phybot/controller_state.h"
#include "phybot/js_mapping.h"
#include "check.h"
#include <linux/joystick.h>   // JS_EVENT_BUTTON / JS_EVENT_AXIS / JS_EVENT_INIT
#include <cmath>
#include <cstdio>

static bool fapprox(float a, float b) { return std::fabs(a - b) < 1e-4f; }

static void test_name_match() {
  using phybot::NameMatchesController;
  CHECK(NameMatchesController("Microsoft Xbox One X pad"));
  CHECK(NameMatchesController("Xbox Wireless Controller"));
  CHECK(NameMatchesController("X-Box 360 pad"));
  CHECK(NameMatchesController("microsoft x-box one"));     // case-insensitive
  CHECK(!NameMatchesController("Logitech M500s Mouse"));
  CHECK(!NameMatchesController("3Dconnexion SpaceMouse"));
}

static void test_buttons() {
  using namespace phybot;
  ControllerState st{};
  ApplyJsEvent(JS_EVENT_BUTTON, 0, 1, st); CHECK(st.a);
  ApplyJsEvent(JS_EVENT_BUTTON, 1, 1, st); CHECK(st.b);
  ApplyJsEvent(JS_EVENT_BUTTON, 3, 1, st); CHECK(st.y);
  ApplyJsEvent(JS_EVENT_BUTTON, 0, 0, st); CHECK(!st.a);              // release
  ApplyJsEvent(JS_EVENT_BUTTON | JS_EVENT_INIT, 7, 1, st); CHECK(st.start);  // INIT flag stripped
  ApplyJsEvent(JS_EVENT_BUTTON, 99, 1, st);                          // unknown -> ignored, no crash
}

static void test_axes() {
  using namespace phybot;
  ControllerState st{};
  ApplyJsEvent(JS_EVENT_AXIS, 0,  32767, st); CHECK(fapprox(st.left_x, 1.0f));
  ApplyJsEvent(JS_EVENT_AXIS, 1, -32767, st); CHECK(fapprox(st.left_y, -1.0f));
  ApplyJsEvent(JS_EVENT_AXIS, 3,  16384, st); CHECK(fapprox(st.right_x, 0.5001f) || st.right_x > 0.49f);
  ApplyJsEvent(JS_EVENT_AXIS, 2,  32767, st); CHECK(fapprox(st.lt, 1.0f));   // trigger raw
  ApplyJsEvent(JS_EVENT_AXIS, 6, -32767, st); CHECK(fapprox(st.dpad_x, -1.0f));
}

int main() {
  test_name_match();
  test_buttons();
  test_axes();
  if (g_check_failures == 0) std::printf("ALL JS MAPPING TESTS PASSED\n");
  return g_check_failures == 0 ? 0 : 1;
}
