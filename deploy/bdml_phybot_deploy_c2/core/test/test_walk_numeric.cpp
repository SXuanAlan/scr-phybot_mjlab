#include "phybot/robot_consts.h"
#include "phybot/data_package.h"
#include "check.h"
#include <cstdio>
#include "phybot/walk_config.h"
#include "phybot/gravity.h"
#include <cmath>
#include "phybot/walk_history.h"
#include "phybot/walk_obs.h"
#include "phybot/walk_action.h"

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

static void test_build_walk_obs() {
  using namespace phybot;
  WalkConfig c = walk_config();
  WalkObsInputs in{};
  in.gravity = {0.0, 0.0, -1.0};
  in.ang_vel = {0.4, 0.8, -0.2};               // *0.25 → 0.1,0.2,-0.05
  in.q = c.default_dof_pos;                     // q-default = 0
  for (int i = 0; i < kNumDofs; ++i) in.dq[i] = 0.2;   // *0.05 → 0.01
  for (int i = 0; i < kNumDofs; ++i) in.last_action[i] = 0.3;
  in.js_vx = 0.5; in.js_vy = -0.1; in.js_wz = 0.4;     // cmd=0.5,-0.1,0.1
  float zeros105[105] = {0};
  float obs[kObsDim];
  BuildWalkObs(in, c, zeros105, zeros105, obs);

  CHECK(fapprox(obs[0], 0.1f) && fapprox(obs[1], 0.2f) && fapprox(obs[2], -0.05f));  // ang_vel*0.25
  CHECK(obs[3] == 0.0f && obs[4] == 0.0f && obs[5] == -1.0f);    // gravity (exact)
  CHECK(obs[6] == 0.5f && obs[7] == -0.1f && fapprox(obs[8], 0.1f));  // commands (cmd2=wz*0.25)
  CHECK(obs[9] == 1.0f);                                         // |0.5|>=0.3 → walk
  CHECK(obs[10] == 0.0f && obs[30] == 0.0f);                     // (q-default)=0
  CHECK(fapprox(obs[31], 0.01f) && fapprox(obs[51], 0.01f));     // dq*0.05
  CHECK(obs[52] == 0.3f && obs[72] == 0.3f);                     // last_action raw (direct copy)
  CHECK(obs[73] == 0.0f && obs[177] == 0.0f);                    // zero pos history
  CHECK(obs[178] == 0.0f && obs[282] == 0.0f);                   // zero vel history
}

static void test_walk_flag_threshold() {
  using namespace phybot;
  WalkConfig c = walk_config();
  WalkObsInputs in{}; in.q = c.default_dof_pos;
  float z[105] = {0}; float obs[kObsDim];
  in.js_vx = 0.2; BuildWalkObs(in, c, z, z, obs); CHECK(obs[9] == 0.0f);  // |0.2|<0.3 → stand
  in.js_vx = 0.3; BuildWalkObs(in, c, z, z, obs); CHECK(obs[9] == 1.0f);  // 0.3 not < 0.3 → walk
  in.js_vx = -0.5; BuildWalkObs(in, c, z, z, obs); CHECK(obs[9] == 1.0f); // abs
}

static void test_pos_minus_default() {
  using namespace phybot;
  WalkConfig c = walk_config();
  WalkObsInputs in{}; in.q = c.default_dof_pos; in.q[0] += 0.1; in.q[20] -= 0.05;
  float z[105] = {0}; float obs[kObsDim];
  BuildWalkObs(in, c, z, z, obs);
  CHECK(std::fabs(obs[10] - 0.1f) < 1e-6f);     // (q0-default0)*1.0
  CHECK(std::fabs(obs[30] - (-0.05f)) < 1e-6f); // (q20-default20)*1.0
}

static void test_apply_walk_action() {
  using namespace phybot;
  WalkConfig c = walk_config();
  DataPackage io;
  std::array<double,kNumDofs> act{};
  act[0] = 1.0;    // → 1.0*0.2 + default[0]
  act[4] = 2.0;    // → 2.0*0.2 + (-0.295)
  ApplyWalkAction(act, c, io);
  CHECK(std::fabs(io.motor_Pos_desire[0] - (0.2 + c.default_dof_pos[0])) < 1e-12);
  CHECK(std::fabs(io.motor_Pos_desire[4] - (0.4 + (-0.295))) < 1e-12);
  CHECK(io.motor_Vel_desire[0] == 0.0 && io.motor_Torque_desire[0] == 0.0 && io.torq_desire[0] == 0.0);
  CHECK(io.sim_P[12] == 20.0 && io.sim_D[12] == 10.0);   // rl_p/rl_d at waist joint
}

int main() {
  test_datapackage_cpg_fields_zeroed();
  test_walk_config_values();
  test_gravity();
  test_history_zero_init_and_push();
  test_history_ordering();
  test_build_walk_obs();
  test_walk_flag_threshold();
  test_pos_minus_default();
  test_apply_walk_action();
  if (g_check_failures == 0) std::printf("ALL WALK NUMERIC TESTS PASSED\n");
  return g_check_failures == 0 ? 0 : 1;
}
