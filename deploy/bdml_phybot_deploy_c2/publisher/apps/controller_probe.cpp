#include "phybot/js_controller.h"
#include <cstdio>
#include <ctime>
int main() {
  phybot::JsController jc;
  std::printf("Discovering Xbox controller (move sticks / press buttons; Ctrl-C to quit)...\n");
  timespec ts{0, 20 * 1000 * 1000};  // 20 ms ~ 50 Hz
  int since_scan = 0;
  while (true) {
    if (!jc.connected()) {
      if (jc.discover()) std::printf("CONNECTED: \"%s\"\n", jc.name().c_str());
      else { if (since_scan++ % 50 == 0) std::printf("...no controller; retrying\n"); nanosleep(&ts, nullptr); continue; }
    }
    phybot::ControllerState s;
    if (!jc.poll(s)) { std::printf("DISCONNECTED; rediscovering...\n"); nanosleep(&ts, nullptr); continue; }
    std::printf("\rA%d B%d X%d Y%d LB%d RB%d | LX%+.2f LY%+.2f RX%+.2f RY%+.2f LT%+.2f RT%+.2f DX%+.0f DY%+.0f   ",
                s.a,s.b,s.x,s.y,s.lb,s.rb, s.left_x,s.left_y,s.right_x,s.right_y,s.lt,s.rt,s.dpad_x,s.dpad_y);
    std::fflush(stdout);
    nanosleep(&ts, nullptr);
  }
  return 0;
}
