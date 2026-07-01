#include "mujoco_viewer.h"

#include "glfw_adapter.h"
#include "simulate.h"

namespace phybot {

MujocoViewer::MujocoViewer(const char* title) : title_(title ? title : "sim") {
  mjv_defaultCamera(&cam_);
  mjv_defaultOption(&opt_);
  mjv_defaultPerturb(&pert_);
  // is_passive=false -> full interactive UI (menus, plots, mouse perturb),
  // matching the official mujoco_sim main.
  sim_ = std::make_unique<mujoco::Simulate>(
      std::make_unique<mujoco::GlfwAdapter>(), &cam_, &opt_, &pert_,
      /*is_passive=*/false);
}

// Out-of-line so unique_ptr<Simulate> sees the complete type here.
MujocoViewer::~MujocoViewer() = default;

void MujocoViewer::loadModel(mjModel* m, mjData* d) {
  d_ = d;
  sim_->LoadMessage(title_.c_str());
  sim_->Load(m, d, title_.c_str());  // blocks until the render thread is ready
  // Initial framing (same values as the official update loop's setup).
  cam_.type = mjCAMERA_FREE;
  cam_.distance = 3.0;
  cam_.elevation = -20.0;
  cam_.azimuth = 120.0;
}

std::recursive_mutex& MujocoViewer::mutex() { return sim_->mtx; }

bool MujocoViewer::exitRequested() const { return sim_->exitrequest.load() != 0; }

bool MujocoViewer::running() const { return sim_->run != 0; }

void MujocoViewer::trackCamera() {
  if (!d_) return;
  cam_.lookat[0] = d_->qpos[0];
  cam_.lookat[1] = d_->qpos[1];
  cam_.lookat[2] = d_->qpos[2];
}

void MujocoViewer::renderLoop() { sim_->RenderLoop(); }

}  // namespace phybot
