#!/usr/bin/env bash
# Populate third_party/ (a full mirror of the official ThirdParty) and assets/
# with the large binary dependencies. These are NOT committed to git (see
# .gitignore); run this once after cloning, before building. Idempotent.
# Deploy to the robot by rsyncing this whole directory (deps included) — it has
# no dependency on the official tree once populated.
#
# Usage:
#   ./vendor.sh                      # copy from ../phybot_deploy_c2 (default)
#   ./vendor.sh /path/to/official    # copy from a custom official-tree root
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:-$HERE/../phybot_deploy_c2}"

if [[ ! -d "$SRC/ThirdParty" ]]; then
  echo "ERROR: official deps not found under: $SRC" >&2
  echo "Pass the official deploy tree root as \$1 (must contain ThirdParty/, lcm_server/, RobotModel/)." >&2
  exit 1
fi

echo "Vendoring from: $SRC"

# third_party/ = full mirror of the official ThirdParty/ (libtorch, mujoco, yaml,
# eigen3, boost, pinocchio, glfw, gl, spdlog, nlohmann, urdfdom, qpOASES, LCM, ...)
rm -rf "$HERE/third_party"
mkdir -p "$HERE/third_party"
cp -a "$SRC/ThirdParty/." "$HERE/third_party/"

# working LCM (release build with liblcm + glib + headers — this is what CMake links)
cp -a "$SRC/lcm_server/lcm/release" "$HERE/third_party/lcm"

# robot model (sim only): XML + meshes
mkdir -p "$HERE/assets/robot"
rm -rf "$HERE/assets/robot/phybot_c2"
cp -a "$SRC/RobotModel/phybot_c2" "$HERE/assets/robot/phybot_c2"

echo "Done. third_party/ ($(du -sh "$HERE/third_party" | cut -f1)) + assets/ populated."
echo "Build:  cmake -S . -B build && cmake --build build -j2 && (cd build && ctest)"
