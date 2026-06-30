#pragma once
#include <array>
#include <cmath>
namespace phybot {
// 复现官方 Euler_ZYXToGravityVec: R = Rz(a0)Ry(a1)Rx(a2), 返回 -R^T.col(2)。
// 闭式: g = [sin(a1), -cos(a1)sin(a2), -cos(a1)cos(a2)] (与 yaw a0 无关)。
inline std::array<double, 3> GravityFromEulerZYX(const std::array<double, 3>& zyx) {
  const double a1 = zyx[1], a2 = zyx[2];
  return { std::sin(a1), -std::cos(a1) * std::sin(a2), -std::cos(a1) * std::cos(a2) };
}
}  // namespace phybot
