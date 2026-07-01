#!/usr/bin/env bash
# LCM (udpm://239.255.76.67:7667) needs a multicast route. On a single dev
# machine there usually isn't a usable one, so the vendored liblcm aborts with a
# glibc "buffer overflow" (or hangs) inside its udpm init/handle. Forcing all
# multicast onto the loopback interface fixes it: the self-test loops locally.
#
# The REAL ROBOT does not need this — its NIC has working multicast. Dev-only.
#
#   sudo ./setup_lcm_local.sh on       # add loopback multicast route
#        ./setup_lcm_local.sh status   # show current state (no sudo)
#   sudo ./setup_lcm_local.sh off      # remove it
set -euo pipefail
GROUP="224.0.0.0/4"
cmd="${1:-status}"
case "$cmd" in
  on)
    sudo ip link set lo multicast on
    sudo ip route replace "$GROUP" dev lo
    echo "LCM loopback multicast: ON"
    ip route show | grep -E '224\.0\.0\.0/4' | sed 's/^/  /'
    echo "Now run (two terminals):  ./build/main_publisher   and   ./build/sim_teleop"
    ;;
  off)
    sudo ip route del "$GROUP" dev lo 2>/dev/null || true
    echo "LCM loopback multicast: OFF"
    ;;
  status)
    if ip route show | grep -qE '224\.0\.0\.0/4[[:space:]]+dev[[:space:]]+lo'; then
      echo "ON"; ip route show | grep -E '224\.0\.0\.0/4' | sed 's/^/  /'
    else
      echo "OFF  (enable with:  sudo ./setup_lcm_local.sh on )"
    fi
    ;;
  *)
    echo "usage: $0 {on|off|status}" >&2; exit 1 ;;
esac
