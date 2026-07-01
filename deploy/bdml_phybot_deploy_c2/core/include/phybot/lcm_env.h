#pragma once
#include <cstdio>

namespace phybot {

// The vendored liblcm's default provider is udpm://239.255.76.67:7667. On a host
// with no usable multicast loopback route it aborts with a glibc "buffer overflow"
// (or hangs) somewhere inside its udpm init/handle path -- an uncatchable SIGABRT
// that fires during construction/use, so lcm::LCM::good() cannot guard it.
//
// We cannot catch that abort, but we can print the remedy BEFORE touching LCM so
// the user never sees a bare "buffer overflow" with no context. Deliberately
// non-fatal and non-probing: it always prints and never blocks, so it stays
// correct on the real robot (where multicast works over the NIC and LCM is fine).
inline void AnnounceLcmInit(const char* app) {
  std::fprintf(stderr,
    "[%s] opening LCM udpm://239.255.76.67:7667 ...\n"
    "      if this aborts (\"buffer overflow\") or hangs, this host has no usable\n"
    "      multicast route -- run (dev only):  sudo ./setup_lcm_local.sh on\n",
    app);
  std::fflush(stderr);
}

}  // namespace phybot
