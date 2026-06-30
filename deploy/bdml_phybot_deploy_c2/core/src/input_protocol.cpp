#include "phybot/input_protocol.h"
#include <cstring>
namespace phybot {
static constexpr uint16_t kHeader = 0xeb90, kFrame = 0x0001;
int8_t ip_checksum(const custom_lcm::system2Algorithm_packet& p) {
  const uint8_t* b = reinterpret_cast<const uint8_t*>(&p);
  int32_t s = 0; for (int i = 2; i < 63; ++i) s += b[i];
  return (int8_t)(s % 255);
}
void ip_encode(const DriveCmd& c, custom_lcm::system2Algorithm_packet& p) {
  p.header = (int16_t)kHeader; p.frame_type = (int16_t)kFrame;
  std::memset(p.data, 0, sizeof(p.data));
  p.data[0] = c.model;
  std::memcpy(p.data + 1, &c.vx, 4);
  std::memcpy(p.data + 5, &c.vy, 4);
  std::memcpy(p.data + 9, &c.wz, 4);
  p.checksum = ip_checksum(p);
}
bool ip_decode(const custom_lcm::system2Algorithm_packet& p, DriveCmd& out) {
  if ((uint16_t)p.header != kHeader) return false;
  if ((uint16_t)p.frame_type != kFrame) return false;
  if (ip_checksum(p) != p.checksum) return false;
  out.model = p.data[0];
  std::memcpy(&out.vx, p.data + 1, 4);
  std::memcpy(&out.vy, p.data + 5, 4);
  std::memcpy(&out.wz, p.data + 9, 4);
  return true;
}
}  // namespace phybot
