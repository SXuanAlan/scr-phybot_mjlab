#include "phybot/robot_consts.h"
#include "phybot/data_package.h"
#include "check.h"
#include <cstdio>

static void test_datapackage_cpg_fields_zeroed() {
  phybot::DataPackage io;
  CHECK(io.motor_pos.size() == (size_t)phybot::kNumDofs);
  CHECK(io.motor_Pos_desire.size() == (size_t)phybot::kNumDofs);
  CHECK(io.imu_zyx.size() == 3);
  CHECK(io.js_vx_desire == 0.0);
  CHECK(io.motor_pos[0] == 0.0 && io.motor_pos[20] == 0.0);
  CHECK(io.sim_P[10] == 0.0);
}

int main() {
  test_datapackage_cpg_fields_zeroed();
  if (g_check_failures == 0) std::printf("ALL WALK NUMERIC TESTS PASSED\n");
  return g_check_failures == 0 ? 0 : 1;
}
