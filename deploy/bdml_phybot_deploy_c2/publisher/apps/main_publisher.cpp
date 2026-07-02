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
  // buttons.yaml (button -> select_code) fully drives ms.binds: any button listed
  // there (see ButtonField in buttons_config.cpp for the recognized
  // A/B/X/Y/LB/RB/BACK/START names) binds to its code with no changes here --
  // e.g. uncommenting `X: 3` in buttons.yaml binds X with zero code edits.
  const phybot::ButtonsConfig bc = phybot::load_buttons_config();
  ms.binds = phybot::BuildButtonBinds(bc);
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
