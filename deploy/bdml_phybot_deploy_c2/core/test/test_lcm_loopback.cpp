#include "phybot/input_protocol.h"
#include "check.h"
#include <lcm/lcm-cpp.hpp>
#include <cstdio>
struct Handler {
  bool got=false; phybot::DriveCmd cmd;
  void on(const lcm::ReceiveBuffer*, const std::string&,
          const custom_lcm::system2Algorithm_packet* p) {
    got = phybot::ip_decode(*p, cmd);
  }
};
int main() {
  lcm::LCM lcm;
  if (!lcm.good()) { std::printf("LCM init failed (no multicast?)\n"); return 1; }
  Handler h;
  lcm.subscribe("Robot_ControlData", &Handler::on, &h);
  phybot::DriveCmd in; in.model=2; in.vx=0.42f; in.vy=-0.05f; in.wz=0.3f;
  custom_lcm::system2Algorithm_packet pkt; phybot::ip_encode(in, pkt);
  lcm.publish("Robot_ControlData", &pkt);
  // pump for up to ~500ms
  for (int i=0;i<50 && !h.got;++i) lcm.handleTimeout(10);
  CHECK(h.got);
  CHECK(h.cmd.model==2);
  CHECK(std::abs(h.cmd.vx-0.42f)<1e-6 && std::abs(h.cmd.wz-0.3f)<1e-6);
  if (g_check_failures==0) std::printf("LCM LOOPBACK PASSED\n");
  return g_check_failures==0?0:1;
}
