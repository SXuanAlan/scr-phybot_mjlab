#include "phybot/js_controller.h"
#include "phybot/input_mapping.h"
#include "phybot/input_protocol.h"
#include "phybot/lcm_env.h"
#include "phybot/buttons_config.h"
#include <lcm/lcm-cpp.hpp>
#include <cstdio>
#include <ctime>
int main() {
  phybot::AnnounceLcmInit("publisher");
  lcm::LCM lcm;
  if (!lcm.good()) { std::printf("LCM init failed\n"); return 1; }
  phybot::JsController jc; phybot::MappingState ms;
  // buttons.yaml (button -> select_code) resolves the model codes once at startup;
  // fall back to the zero/walk tasks' own literal select_codes (1/2) if a key is
  // missing (see MappingState's defaults in input_mapping.h).
  const phybot::ButtonsConfig bc = phybot::load_buttons_config();
  ms.zero_code = bc.code_for("Y", 1);
  ms.walk_code = bc.code_for("B", 2);
  timespec ts{0, 10*1000*1000};  // ~100 Hz
  std::printf("[publisher] running; discovering controller...\n");
  while (true) {
    if (!jc.connected()) { if (!jc.discover()) { nanosleep(&ts,nullptr); continue; }
                           std::printf("[publisher] connected: %s\n", jc.name().c_str()); }
    phybot::ControllerState s;
    if (!jc.poll(s)) { nanosleep(&ts,nullptr); continue; }
    phybot::DriveCmd c = phybot::MapControllerToDrive(s, ms);
    custom_lcm::system2Algorithm_packet pkt; phybot::ip_encode(c, pkt);
    lcm.publish("Robot_ControlData", &pkt);
    nanosleep(&ts, nullptr);
  }
}
