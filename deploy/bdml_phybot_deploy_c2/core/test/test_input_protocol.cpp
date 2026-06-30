#include "phybot/input_protocol.h"
#include "check.h"
#include <cmath>
#include <cstdio>
static bool fa(float a, float b){ return std::fabs(a-b) < 1e-6f; }

static void test_roundtrip() {
  using namespace phybot;
  DriveCmd in; in.model = 2; in.vx = 0.37f; in.vy = -0.08f; in.wz = 0.5f;
  custom_lcm::system2Algorithm_packet pkt;
  ip_encode(in, pkt);
  CHECK((uint16_t)pkt.header == 0xeb90);
  CHECK((uint16_t)pkt.frame_type == 0x0001);
  DriveCmd out;
  CHECK(ip_decode(pkt, out));
  CHECK(out.model == 2);
  CHECK(fa(out.vx, 0.37f) && fa(out.vy, -0.08f) && fa(out.wz, 0.5f));
}
static void test_decode_rejects_bad() {
  using namespace phybot;
  DriveCmd in; in.model = 1;
  custom_lcm::system2Algorithm_packet pkt; ip_encode(in, pkt);
  DriveCmd out;
  pkt.checksum = pkt.checksum ^ 0x7F;      // corrupt checksum
  CHECK(!ip_decode(pkt, out));
  ip_encode(in, pkt); pkt.header = 0x1234;  // bad header
  CHECK(!ip_decode(pkt, out));
}
int main(){ test_roundtrip(); test_decode_rejects_bad();
  if(g_check_failures==0) std::printf("ALL INPUT PROTOCOL TESTS PASSED\n");
  return g_check_failures==0?0:1; }
