// sim_walk: headless MuJoCo walk simulation using the CPG WalkTask.
// Usage: ./sim_walk <cpg_policy.pt> <phybot_c2.xml> [vx=0.4] [seconds=5.0]
#include "phybot/data_package.h"
#include "phybot/robot_consts.h"
#include "walk/walk_task.h"
#include "mujoco_backend.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Asset paths are baked in by CMake (absolute, vendored — CWD-independent).
// These #ifndef fallbacks are relative to the repo root, for a manual build
// without CMake; run such a build from deploy/bdml_phybot_deploy_c2/.
#ifndef POLICY_PATH
#define POLICY_PATH "core/test/golden/cpg_policy.pt"
#endif
#ifndef XML_PATH
#define XML_PATH "assets/robot/phybot_c2/xml/phybot_c2.xml"
#endif

int main(int argc, char** argv) {
    const char* policy_path = (argc > 1) ? argv[1] : POLICY_PATH;
    const char* xml_path    = (argc > 2) ? argv[2] : XML_PATH;
    double vx_cmd  = (argc > 3) ? std::atof(argv[3]) : 0.4;
    double sim_sec = (argc > 4) ? std::atof(argv[4]) : 5.0;

    std::printf("[sim_walk] policy=%s\n", policy_path);
    std::printf("[sim_walk] xml=%s\n",    xml_path);
    std::printf("[sim_walk] vx=%.2f  sim_sec=%.1f\n", vx_cmd, sim_sec);

    // Construct backend and task
    phybot::MujocoBackend backend(xml_path);
    phybot::WalkTask      task(policy_path);
    phybot::DataPackage   io;

    // Set joystick commands
    io.js_vx_desire    = vx_cmd;
    io.js_vy_desire    = 0.0;
    io.js_OmegaZ_desire = 0.0;

    // Read initial state so task has valid sensor data before enter()
    backend.readToPackage(io);
    task.enter(io);

    // Determine number of control ticks from sim time and XML timestep
    // (1 control tick = 1 physics step; policy fires every kDecimation ticks)
    double timestep = 0.001;  // default; backend prints actual value
    // We'll derive it from the sim time after a few steps later.
    // For now use a generous tick count based on sim_sec at 1 kHz.
    int total_ticks = static_cast<int>(sim_sec * 1000.0);  // 1000 Hz assumption

    std::vector<double> heights;
    heights.reserve(total_ticks);

    double min_height   = 1e9;
    double final_height = 0.0;
    double final_x      = 0.0;

    std::printf("[sim_walk] running %d ticks...\n", total_ticks);

    for (int tick = 0; tick < total_ticks; ++tick) {
        backend.readToPackage(io);
        task.run(io);
        backend.writeFromPackage(io);
        backend.step();

        double h = backend.baseHeight();
        heights.push_back(h);
        if (h < min_height) min_height = h;

        if (tick % 500 == 0) {
            std::printf("  tick=%5d  t=%.3fs  h=%.4f  x=%.4f\n",
                        tick, backend.simTime(), h, backend.baseX());
        }
    }

    final_height = backend.baseHeight();
    final_x      = backend.baseX();

    std::printf("\n[sim_walk] DONE\n");
    std::printf("  min_height=%.4f  final_height=%.4f  x_displacement=%.4f\n",
                min_height, final_height, final_x);

    bool pass = (min_height > 0.4);
    std::printf("  result: %s\n", pass ? "WALK_OK" : "FALL");
    return pass ? 0 : 1;
}
