// sim_walk: thin SimHarness shell -- headless (or --gui) MuJoCo run with the WALK
// task commanded directly (no LCM in the loop). SimHarness resolves the WalkTask's
// policy path itself, from tasks/walk/config/walk.yaml (via WalkConfig) -- this app
// never constructs a task or injects a policy path; see backends/mujoco/sim_harness.h.
//
// vx is NORMALIZED to [-1,1], the same convention the publisher sends over LCM
// (see publisher/include/phybot/input_mapping.h). WalkTask's own tasks/walk/config/
// controller.yaml maps it into m/s via TaskRouter::shape (vx range [-0.4, 1.0]).
//
// Usage: ./sim_walk [xml_path] [vx=0.4] [seconds=5.0] [--gui]
#include "sim_harness.h"
#include "sim_harness_gui.h"
#include "phybot/input_protocol.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Asset path is baked in by CMake (absolute, vendored -- CWD-independent). This
// #ifndef fallback is relative to the repo root, for a manual build without CMake;
// run such a build from deploy/bdml_phybot_deploy_c2/.
#ifndef XML_PATH
#define XML_PATH "assets/robot/phybot_c2/xml/phybot_c2.xml"
#endif

int main(int argc, char** argv) {
    using namespace phybot;

    // --gui may appear anywhere; the rest are positional [xml vx seconds].
    bool gui = false;
    std::vector<const char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--gui") gui = true;
        else pos.push_back(argv[i]);
    }
    // Empty positional ("") falls back to the default, so `--gui`, `"" 0.8 10`
    // etc. all work without having to spell out the xml path.
    const char* xml_path = (pos.size() > 0 && pos[0][0]) ? pos[0] : XML_PATH;
    double vx  = (pos.size() > 1 && pos[1][0]) ? std::atof(pos[1]) : 0.4;  // normalized [-1,1]
    double sec = (pos.size() > 2 && pos[2][0]) ? std::atof(pos[2]) : 5.0;

    std::printf("[sim_walk] xml=%s\n", xml_path);
    std::printf("[sim_walk] vx=%.2f (normalized)  sim_sec=%.1f  gui=%d\n", vx, sec, (int)gui);

    SimHarness h({xml_path});
    auto src = [&]{ DriveCmd c; c.model = 2; c.vx = (float)vx; return c; };  // WALK, normalized vx

    if (gui) {
        std::printf("[sim_walk] GUI: space=pause, drag=rotate, scroll=zoom, ESC/close=quit\n");
        RunGui(h, src, "sim_walk");
    } else {
        h.runHeadless(src, sec);
        std::printf("[sim_walk] height=%.4f x=%.4f\n", h.backend().baseHeight(), h.backend().baseX());
    }
    return 0;
}
