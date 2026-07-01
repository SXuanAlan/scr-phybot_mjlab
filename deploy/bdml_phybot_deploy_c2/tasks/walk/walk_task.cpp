#include "walk/walk_task.h"
#include "phybot/policy_runner.h"
#include "phybot/gravity.h"
#include "walk/walk_obs.h"
#include "walk/walk_action.h"
#include "phybot/data_package.h"
#include "phybot/task_registry.h"
#include <cstring>
#include <stdexcept>

namespace phybot {

WalkTask::WalkTask(const std::string& model_path)
    : Task("walk"),
      model_path_(model_path),
      cfg_(walk_config()) {}

WalkTask::~WalkTask() = default;

void WalkTask::enter(DataPackage& io) {
  step_num_ = 0;
  last_action_.fill(0.0);
  history_.Reset();
  if (!runner_) {
    // No explicit model_path (ctor default ""): fall back to policy_path resolved from this
    // task's own config (tasks/walk/config/walk.yaml -> policy_path), so a no-arg WalkTask()
    // -- e.g. one built by TaskRegistry's name-only factory -- self-registers a working
    // policy instead of requiring the caller to inject a path. cfg_ is already fully
    // constructed here (member-initialized before this ctor's body/enter() can ever run).
    std::string path = model_path_.empty() ? cfg_.policy_path : model_path_;
    if (path.empty())
      throw std::runtime_error("WalkTask: model_path is empty");
    runner_ = std::make_unique<PolicyRunner>(path);
  }
}

void WalkTask::run(DataPackage& io) {
  // 1. Advance step counter (matches rl_deploy_cpg::Step() exactly)
  if (step_num_ == 0) step_num_ = 1;
  else                ++step_num_;

  // 2. Inference on decimation ticks
  if (step_num_ % kDecimation == 0) {
    // Build obs from CURRENT sensors + PREVIOUS last_action + PREVIOUS history
    WalkObsInputs in{};
    in.gravity = GravityFromEulerZYX(io.imu_zyx);
    in.ang_vel = io.imu_angular_vel;
    for (int i = 0; i < kNumDofs; ++i) {
      in.q[i]           = io.motor_pos[i];
      in.dq[i]          = io.motor_vel[i];
      in.last_action[i] = last_action_[i];  // raw, unscaled
    }
    in.js_vx = io.js_vx_desire;
    in.js_vy = io.js_vy_desire;
    in.js_wz = io.js_OmegaZ_desire;

    float pos105[kHistFramesInObs * kNumDofs];
    float vel105[kHistFramesInObs * kNumDofs];
    history_.NewestPos(pos105);
    history_.NewestVel(vel105);

    float obs[kObsDim];
    BuildWalkObs(in, cfg_, pos105, vel105, obs);

    float raw[kNumDofs];
    runner_->forward(obs, kObsDim, raw, kNumDofs);

    for (int i = 0; i < kNumDofs; ++i) last_action_[i] = static_cast<double>(raw[i]);

    // Push history AFTER obs+forward (matches CPG cadence)
    history_.Push(io.motor_pos, io.motor_vel, cfg_);
  }

  // 3. Apply held action every tick
  ApplyWalkAction(last_action_, cfg_, io);
}

void WalkTask::exit(DataPackage& io) {
  step_num_ = 0;
  last_action_.fill(0.0);
  history_.Reset();
}

// NOTE: REGISTER_TASK(NAME, CLASS) token-pastes CLASS##_Reg; a namespace-qualified
// CLASS (e.g. phybot::WalkTask) would pave to the ill-formed `struct phybot::WalkTask_Reg`
// (verified: gcc rejects it -- qualified name does not name a class). So, exactly like
// zero_task.cpp's own REGISTER_TASK("zero", ZeroState), this uses the unqualified name
// from inside namespace phybot; the registered factory still constructs phybot::WalkTask.
REGISTER_TASK("walk", WalkTask)

}  // namespace phybot
