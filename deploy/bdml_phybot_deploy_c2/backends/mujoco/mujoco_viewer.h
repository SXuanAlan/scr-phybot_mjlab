#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <mujoco/mujoco.h>

namespace mujoco { class Simulate; }  // ported GUI (backends/mujoco/gui)

namespace phybot {

// Thin wrapper over the vendored MuJoCo `simulate` UI. It renders an EXTERNALLY
// owned mjModel*/mjData* (e.g. MujocoBackend's) -- it never owns or steps them,
// so the control loop stays the single source of truth for physics.
//
// Threading mirrors the official main.cc: Simulate::Load() BLOCKS until the
// render thread is ready, so it must run on a different thread than RenderLoop.
// Therefore:
//   - construct on the main thread (creates the UI, no model yet);
//   - spawn a physics thread that calls loadModel() then runs the control loop,
//     locking mutex() around each read/tick/write/step;
//   - call renderLoop() on the main thread (blocks until the window closes).
//
//   MujocoViewer viewer("sim");
//   std::thread physics([&]{
//     viewer.loadModel(backend.model(), backend.data());   // blocks until ready
//     while (!viewer.exitRequested()) {
//       { std::lock_guard<std::recursive_mutex> lk(viewer.mutex());
//         if (viewer.running()) { /* read,tick,write,step */ viewer.trackCamera(); } }
//       nanosleep(...);
//     }
//   });
//   viewer.renderLoop();   // main thread
//   physics.join();
class MujocoViewer {
 public:
  explicit MujocoViewer(const char* title);
  ~MujocoViewer();

  MujocoViewer(const MujocoViewer&) = delete;
  MujocoViewer& operator=(const MujocoViewer&) = delete;

  // Physics thread: hand the live model/data to the UI. Blocks until the render
  // thread has finished the initial upload (see Simulate::Load).
  void loadModel(mjModel* m, mjData* d);

  // Physics thread locks this around each simulation step so the render thread
  // never reads mjData mid-step. Same recursive mutex the UI uses internally.
  std::recursive_mutex& mutex();

  bool exitRequested() const;  // window closed / ESC pressed
  bool running() const;        // false while the UI is paused (space bar)

  // Point the free camera at the floating base (call inside the locked step).
  void trackCamera();

  // Main thread: run the blocking UI loop; returns when the window is closed.
  void renderLoop();

 private:
  mjvCamera  cam_{};
  mjvOption  opt_{};
  mjvPerturb pert_{};
  mjData*    d_{nullptr};
  std::string title_;
  std::unique_ptr<mujoco::Simulate> sim_;
};

}  // namespace phybot
