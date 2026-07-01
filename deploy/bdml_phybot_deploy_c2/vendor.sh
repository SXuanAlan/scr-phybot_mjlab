#!/usr/bin/env bash
# Populate third_party/ and assets/ with the large binary dependencies.
# These are NOT committed to git (see .gitignore); run this once after cloning,
# before building. Idempotent (re-run to refresh). Deploy to the robot by
# rsyncing this whole directory (deps included) — no dependency on the official tree.
#
# Usage:
#   ./vendor.sh                      # copy from ../phybot_deploy_c2 (default)
#   ./vendor.sh /path/to/deps_root   # copy from a custom official-tree root
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:-$HERE/../phybot_deploy_c2}"

if [[ ! -d "$SRC/ThirdParty/libtorch" ]]; then
  echo "ERROR: source deps not found under: $SRC" >&2
  echo "Pass the official deploy tree root as \$1 (must contain ThirdParty/, lcm_server/, RobotModel/)." >&2
  exit 1
fi

echo "Vendoring dependencies from: $SRC"
mkdir -p "$HERE/third_party" "$HERE/assets/robot"

copy() {  # copy() SRC_PATH DEST_PATH  — remove-then-copy so re-runs don't nest
  local s="$1" d="$2"
  echo "  $s  ->  ${d#$HERE/}"
  rm -rf "$d"
  cp -a "$s" "$d"
}

copy "$SRC/ThirdParty/libtorch"    "$HERE/third_party/libtorch"    # 760M (Torch)
copy "$SRC/ThirdParty/mujoco"      "$HERE/third_party/mujoco"      # 8M  (sim only)
copy "$SRC/ThirdParty/yaml"        "$HERE/third_party/yaml"        # yaml-cpp
copy "$SRC/lcm_server/lcm/release" "$HERE/third_party/lcm"         # liblcm + glib + lcm headers
copy "$SRC/RobotModel/phybot_c2"   "$HERE/assets/robot/phybot_c2"  # 144M (sim mesh model)

echo "Done. third_party/ + assets/ populated. Build with:  cmake -S . -B build && cmake --build build -j2"
