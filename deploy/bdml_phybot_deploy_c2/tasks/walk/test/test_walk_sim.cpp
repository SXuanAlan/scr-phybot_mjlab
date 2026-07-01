// test_walk_sim: headless MuJoCo integration test — assert robot stays upright while walking.
// CTest: PASS if base height > 0.4 m throughout; FAIL otherwise.
#include "phybot/data_package.h"
#include "phybot/robot_consts.h"
#include "walk/walk_task.h"
#include "mujoco_backend.h"
#include <cmath>
#include <cstdio>
#include <cstring>

// Paths injected by CMake
#ifndef POLICY_PATH
#define POLICY_PATH "tasks/walk/test/golden/cpg_policy.pt"
#endif
#ifndef XML_PATH
#define XML_PATH "assets/robot/phybot_c2/xml/phybot_c2.xml"
#endif

int main() {
    constexpr double SIM_SECONDS  = 3.0;
    constexpr double HEIGHT_FLOOR = 0.4;
    constexpr double VX_CMD       = 0.4;
    constexpr int    TICKS        = static_cast<int>(SIM_SECONDS * 1000); // 3000

    std::printf("[walk_sim] policy=%s\n", POLICY_PATH);
    std::printf("[walk_sim] xml=%s\n",    XML_PATH);

    phybot::MujocoBackend backend(XML_PATH);
    phybot::WalkTask      task(POLICY_PATH);
    phybot::DataPackage   io;

    io.js_vx_desire     = VX_CMD;
    io.js_vy_desire     = 0.0;
    io.js_OmegaZ_desire = 0.0;

    backend.readToPackage(io);
    task.enter(io);

    double min_height = 1e9;
    double fall_time  = -1.0;

    for (int tick = 0; tick < TICKS; ++tick) {
        backend.readToPackage(io);
        task.run(io);
        backend.writeFromPackage(io);
        backend.step();

        double h = backend.baseHeight();
        if (h < min_height) min_height = h;
        if (h < HEIGHT_FLOOR && fall_time < 0.0) {
            fall_time = backend.simTime();
        }
    }

    double final_height = backend.baseHeight();
    double final_x      = backend.baseX();

    std::printf("[walk_sim] min_height=%.4f  final_height=%.4f  x_disp=%.4f\n",
                min_height, final_height, final_x);

    if (fall_time >= 0.0) {
        std::printf("[walk_sim] FAIL: robot fell at t=%.3fs (height<%.2f)\n",
                    fall_time, HEIGHT_FLOOR);
        return 1;
    }

    std::printf("[walk_sim] PASS: robot walked %.2f m in %.1f s, min height %.4f\n",
                final_x, SIM_SECONDS, min_height);
    return 0;
}
