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

#ifdef PHYBOT_ENABLE_GUI
#include "mujoco_viewer.h"
#include <thread>
#include <mutex>
#include <ctime>
#endif

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
    // --gui may appear anywhere; the rest are positional [policy xml vx seconds].
    bool gui = false;
    std::vector<const char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--gui") gui = true;
        else pos.push_back(argv[i]);
    }
    // Empty positional ("") falls back to the default, so `--gui`, `"" "" 0.8 10`
    // etc. all work without having to spell out the policy/xml paths.
    const char* policy_path = (pos.size() > 0 && pos[0][0]) ? pos[0] : POLICY_PATH;
    const char* xml_path    = (pos.size() > 1 && pos[1][0]) ? pos[1] : XML_PATH;
    double vx_cmd  = (pos.size() > 2 && pos[2][0]) ? std::atof(pos[2]) : 0.4;
    double sim_sec = (pos.size() > 3 && pos[3][0]) ? std::atof(pos[3]) : 5.0;
#ifndef PHYBOT_ENABLE_GUI
    if (gui) std::printf("[sim_walk] (built without GUI; running headless)\n");
    gui = false;
#endif

    std::printf("[sim_walk] policy=%s\n", policy_path);
    std::printf("[sim_walk] xml=%s\n",    xml_path);
    std::printf("[sim_walk] vx=%.2f  sim_sec=%.1f  gui=%d\n", vx_cmd, sim_sec, (int)gui);

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

    // One control tick == one physics step (WalkTask handles policy decimation).
    auto control_tick = [&]{
        backend.readToPackage(io);
        task.run(io);
        backend.writeFromPackage(io);
        backend.step();
    };

#ifdef PHYBOT_ENABLE_GUI
    if (gui) {
        phybot::MujocoViewer viewer("sim_walk");
        std::thread physics([&]{
            viewer.loadModel(backend.model(), backend.data());
            const timespec dt{0, 2 * 1000 * 1000};  // ~2 ms, loose real-time pacing
            while (!viewer.exitRequested()) {
                {
                    std::lock_guard<std::recursive_mutex> lk(viewer.mutex());
                    if (viewer.running()) { control_tick(); viewer.trackCamera(); }
                    else { mj_forward(backend.model(), backend.data()); }
                }
                nanosleep(&dt, nullptr);
            }
        });
        std::printf("[sim_walk] GUI: space=pause, drag=rotate, scroll=zoom, ESC/close=quit\n");
        viewer.renderLoop();   // blocks on the main thread until the window closes
        physics.join();
        return 0;
    }
#endif

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
        control_tick();

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
