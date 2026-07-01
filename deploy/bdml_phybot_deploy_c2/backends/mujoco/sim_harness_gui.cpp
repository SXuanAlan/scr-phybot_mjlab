#include "sim_harness_gui.h"
#include "mujoco_viewer.h"
#include "phybot/input_protocol.h"
#include <thread>
#include <mutex>
#include <ctime>
namespace phybot {
void RunGui(SimHarness& h, SimHarness::CommandSource src, const char* title){
  MujocoViewer viewer(title);
  std::thread physics([&]{ viewer.loadModel(h.backend().model(), h.backend().data());
    const timespec dt{0,2*1000*1000};
    while(!viewer.exitRequested()){ { std::lock_guard<std::recursive_mutex> lk(viewer.mutex());
      if(viewer.running()){ h.tick(src()); viewer.trackCamera(); } else { src(); } }
      nanosleep(&dt,nullptr); } });
  viewer.renderLoop(); physics.join();
}
}
