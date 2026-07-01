#include "zero/zero_task.h"
#include "zero/zero_config.h"
#include "phybot/cubic_traj.h"
#include "phybot/data_package.h"
#include "check.h"
#include <cmath>
#include <cstdio>

static void test_cubic_boundaries() {
  using namespace phybot;
  double p0[2]={0.0,1.0}, z[2]={0,0}, p1[2]={1.0,-1.0}, pos[2], vel[2];
  CubicTraj(p0,z,p1,z, 0.5, 0.0, pos, vel, 2);
  CHECK(std::fabs(pos[0]-0.0)<1e-12 && std::fabs(pos[1]-1.0)<1e-12);   // t=0 -> p0
  CHECK(std::fabs(vel[0])<1e-12 && std::fabs(vel[1])<1e-12);           // zero boundary vel
  CubicTraj(p0,z,p1,z, 0.5, 0.5, pos, vel, 2);
  CHECK(std::fabs(pos[0]-1.0)<1e-9 && std::fabs(pos[1]-(-1.0))<1e-9);  // t=T -> p1
  CubicTraj(p0,z,p1,z, 0.5, 0.25, pos, vel, 2);
  CHECK(std::fabs(pos[0]-0.5)<1e-9);                                   // symmetric midpoint
  CubicTraj(p0,z,p1,z, 0.5, 1.0, pos, vel, 2);
  CHECK(std::fabs(pos[0]-1.0)<1e-12);                                  // t>T -> p1 hold
}

static void test_zero_ramps_from_captured_pose() {
  using namespace phybot;
  ZeroState zs; DataPackage io;
  io.control_period = 0.002;
  for (int i=0;i<kNumDofs;++i) io.motor_pos[i] = 0.7;   // arbitrary power-on pose
  zs.enter(io);                                          // capture init_pose
  zs.run(io);                                            // first tick: t=0.002, near captured pose
  auto zc = zero_config();
  CHECK(std::fabs(io.motor_Pos_desire[0] - 0.7) < 0.05);          // NOT snapped to zero_pose
  CHECK(io.sim_P[0] == zc.zero_p[0] && io.sim_D[0] == zc.zero_d[0]);  // stiff gains 1000/100
  for (int k=0;k<300;++k) zs.run(io);                   // 300*0.002=0.6s > totalTime
  for (int i=0;i<kNumDofs;++i)
    CHECK(std::fabs(io.motor_Pos_desire[i] - zc.zero_pose[i]) < 1e-9);  // settled at zero_pose
  CHECK(std::fabs(zc.zero_pose[4] - (-0.295)) < 1e-12);            // zero_pose == default_dof_pos
}

int main(){ test_cubic_boundaries(); test_zero_ramps_from_captured_pose();
  if(g_check_failures==0) std::printf("ALL ZERO TASK TESTS PASSED\n");
  return g_check_failures==0?0:1; }
