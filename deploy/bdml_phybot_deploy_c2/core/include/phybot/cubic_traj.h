#pragma once
namespace phybot {
// 官方 ZeroState::ThirdpolyVector 逐行移植(三次多项式插值, 边界速度可非零)。
inline void CubicTraj(const double* p0, const double* p0d,
                      const double* p1, const double* p1d,
                      double T, double t, double* pos, double* vel, int n) {
  if (t < 0.0) { for (int i=0;i<n;++i){ pos[i]=p0[i]; vel[i]=0.0; } return; }
  if (t <= T) {
    for (int i=0;i<n;++i) {
      double a0 = p0[i], a1 = p0d[i];
      double m = p1[i] - p0[i] - p0d[i]*T;
      double nn = p1d[i] - p0d[i];
      double a2 = 3.0*m/(T*T) - nn/T;
      double a3 = -2.0*m/(T*T*T) + nn/(T*T);
      pos[i] = a3*t*t*t + a2*t*t + a1*t + a0;
      vel[i] = 3.0*a3*t*t + 2.0*a2*t + a1;
    }
  } else { for (int i=0;i<n;++i){ pos[i]=p1[i]; vel[i]=0.0; } }
}
}  // namespace phybot
