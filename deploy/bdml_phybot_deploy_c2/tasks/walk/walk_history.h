#pragma once
#include <array>
#include <algorithm>
#include <vector>
#include "phybot/robot_consts.h"
#include "walk/walk_config.h"
namespace phybot {
class WalkHistory {
 public:
  WalkHistory() : pos_(kHistLen * kNumDofs, 0.0), vel_(kHistLen * kNumDofs, 0.0) {}

  void Reset() {
    std::fill(pos_.begin(), pos_.end(), 0.0);
    std::fill(vel_.begin(), vel_.end(), 0.0);
  }

  void Push(const std::array<double, kNumDofs>& q,
            const std::array<double, kNumDofs>& dq, const WalkConfig& c) {
    const int n = kNumDofs, total = kHistLen * kNumDofs;
    for (int i = 0; i < total - n; ++i) { pos_[i] = pos_[i + n]; vel_[i] = vel_[i + n]; }
    for (int i = 0; i < n; ++i) {
      pos_[total - n + i] = q[i] * c.dof_pos_scale;   // quirk #1: NO default subtraction
      vel_[total - n + i] = dq[i] * c.dof_vel_scale;
    }
  }

  void NewestPos(float* out105) const { CopyTail(pos_, out105); }
  void NewestVel(float* out105) const { CopyTail(vel_, out105); }

 private:
  static void CopyTail(const std::vector<double>& buf, float* out) {
    const int k = kHistFramesInObs * kNumDofs;        // 105
    const int start = (int)buf.size() - k;
    for (int i = 0; i < k; ++i) out[i] = (float)buf[start + i];
  }
  std::vector<double> pos_, vel_;
};
}  // namespace phybot
