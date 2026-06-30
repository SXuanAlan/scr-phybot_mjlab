#pragma once
#include <cstdint>
#include "custom_lcm/system2Algorithm_packet.hpp"
namespace phybot {
struct DriveCmd { int8_t model{0}; float vx{0}, vy{0}, wz{0}; };
int8_t ip_checksum(const custom_lcm::system2Algorithm_packet& p);
void   ip_encode(const DriveCmd& c, custom_lcm::system2Algorithm_packet& p);
bool   ip_decode(const custom_lcm::system2Algorithm_packet& p, DriveCmd& out);
}  // namespace phybot
