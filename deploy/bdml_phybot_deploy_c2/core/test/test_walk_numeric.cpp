#include "phybot/robot_consts.h"
#include "phybot/data_package.h"
#include "check.h"
#include <cstdio>
#include "phybot/walk_config.h"
#include "phybot/gravity.h"
#include <cmath>
#include "phybot/walk_history.h"

static void test_datapackage_cpg_fields_zeroed() {
  phybot::DataPackage io;
  CHECK(io.motor_pos.size() == (size_t)phybot::kNumDofs);
  CHECK(io.motor_Pos_desire.size() == (size_t)phybot::kNumDofs);
  CHECK(io.imu_zyx.size() == 3);
  CHECK(io.js_vx_desire == 0.0);
  CHECK(io.motor_pos[0] == 0.0 && io.motor_pos[20] == 0.0);
  CHECK(io.sim_P[10] == 0.0);
}

static bool approx(double a, double b) { return std::fabs(a - b) < 1e-12; }
static bool fapprox(float a, float b) { return std::fabs(a - b) < 1e-6f; }  // 缩放浮点比较(避免脆性 ==)

static void test_walk_config_values() {
  auto c = phybot::walk_config();
  CHECK(approx(c.ang_vel_scale, 0.25));
  CHECK(approx(c.dof_vel_scale, 0.05));
  CHECK(approx(c.dof_pos_scale, 1.0));
  CHECK(approx(c.lin_vel_scale, 1.0));
  CHECK(approx(c.walk_flag_threshold, 0.3));
  CHECK(approx(c.action_scale[0], 0.2) && approx(c.action_scale[20], 0.2));
  CHECK(approx(c.default_dof_pos[4], -0.295) && approx(c.default_dof_pos[10], -0.295));
  CHECK(approx(c.default_dof_pos[0], -0.242));
  CHECK(approx(c.rl_p[12], 20.0) && approx(c.rl_d[12], 10.0));
}

static void test_gravity() {
  using phybot::GravityFromEulerZYX;
  auto g0 = GravityFromEulerZYX({0.0, 0.0, 0.0});
  CHECK(approx(g0[0], 0.0) && approx(g0[1], 0.0) && approx(g0[2], -1.0));
  auto gy = GravityFromEulerZYX({0.7, 0.0, 0.0});   // yaw-only → gravity unchanged
  CHECK(approx(gy[0], 0.0) && approx(gy[1], 0.0) && approx(gy[2], -1.0));
  auto gp = GravityFromEulerZYX({0.0, M_PI / 2, 0.0});  // pitch 90° → [1,0,0]
  CHECK(approx(gp[0], 1.0) && approx(gp[1], 0.0) && approx(gp[2], 0.0));
}

static void test_history_zero_init_and_push() {
  using namespace phybot;
  WalkConfig c = walk_config();
  WalkHistory h;
  float pos105[105], vel105[105];
  h.NewestPos(pos105); h.NewestVel(vel105);
  for (int i = 0; i < 105; ++i) { CHECK(pos105[i] == 0.0f); CHECK(vel105[i] == 0.0f); }

  std::array<double,kNumDofs> q{}, dq{};
  for (int i = 0; i < kNumDofs; ++i) { q[i] = 1.0; dq[i] = 2.0; }  // pos→1.0*1.0, vel→2.0*0.05=0.1
  h.Push(q, dq, c);
  h.NewestPos(pos105); h.NewestVel(vel105);
  // newest frame is the tail block [84..104]; quirk: pos stores q*scale, NO default sub
  CHECK(pos105[104] == 1.0f);        // q=1.0 * dof_pos_scale 1.0 (exact)
  CHECK(fapprox(vel105[104], 0.1f)); // dq=2.0 * dof_vel_scale 0.05
  CHECK(pos105[83] == 0.0f);         // still-zero older frame just before newest
}

static void test_history_ordering() {
  using namespace phybot;
  WalkConfig c = walk_config();
  WalkHistory h;
  std::array<double,kNumDofs> dq{};
  for (int f = 1; f <= 6; ++f) {           // push 6 frames with q[0]=f
    std::array<double,kNumDofs> q{}; q[0] = (double)f;
    h.Push(q, dq, c);
  }
  float pos105[105]; h.NewestPos(pos105);
  // newest 5 frames are f=2,3,4,5,6 (oldest f=1 shifted out of the 5-frame window)
  CHECK(pos105[0]  == 2.0f);    // frame f=2, dof 0
  CHECK(pos105[84] == 6.0f);    // newest frame f=6, dof 0 (start of tail block)
}

int main() {
  test_datapackage_cpg_fields_zeroed();
  test_walk_config_values();
  test_gravity();
  test_history_zero_init_and_push();
  test_history_ordering();
  if (g_check_failures == 0) std::printf("ALL WALK NUMERIC TESTS PASSED\n");
  return g_check_failures == 0 ? 0 : 1;
}
