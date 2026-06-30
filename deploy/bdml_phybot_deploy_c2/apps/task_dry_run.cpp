// Minimal dry-run: construct WalkTask, feed one synthetic frame, verify finite outputs.
// Usage: ./task_dry_run <path_to_cpg_policy.pt>
#include "phybot/data_package.h"
#include "phybot/robot_consts.h"
#include "walk/walk_task.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <cpg_policy.pt>\n", argv[0]);
        return 1;
    }

    phybot::WalkTask task(argv[1]);
    phybot::DataPackage io;

    // Synthetic non-zero input
    io.imu_zyx          = {0.01, 0.02, 0.0};
    io.imu_angular_vel  = {0.1, 0.0, 0.0};
    for (int i=0; i<phybot::kNumDofs; i++) {
        io.motor_pos[i] = phybot::walk_config().default_dof_pos[i];
        io.motor_vel[i] = 0.0;
    }
    io.js_vx_desire = 0.5;

    task.enter(io);
    // drive 10 frames to reach first inference
    for (int i=0;i<10;i++) task.run(io);

    std::printf("obs_dim=%d  act_dim=%d\n", phybot::kObsDim, phybot::kNumDofs);
    bool finite = true;
    for (int i=0;i<phybot::kNumDofs;i++) {
        if (!std::isfinite(io.motor_Pos_desire[i])) { finite=false; break; }
    }
    std::printf("motor_Pos_desire[0]=%.6f  all_finite=%s\n",
                io.motor_Pos_desire[0], finite?"yes":"no");
    std::printf("sim_P[0]=%.1f  sim_D[0]=%.1f\n", io.sim_P[0], io.sim_D[0]);
    return finite ? 0 : 1;
}
