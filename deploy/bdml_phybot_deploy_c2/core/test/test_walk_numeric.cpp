#include "phybot/robot_consts.h"
#include "phybot/data_package.h"
#include "check.h"
#include <cstdio>
#include "phybot/walk_config.h"
#include "phybot/gravity.h"
#include <cmath>

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

int main() {
  test_datapackage_cpg_fields_zeroed();
  test_walk_config_values();
  test_gravity();
  if (g_check_failures == 0) std::printf("ALL WALK NUMERIC TESTS PASSED\n");
  return g_check_failures == 0 ? 0 : 1;
}
