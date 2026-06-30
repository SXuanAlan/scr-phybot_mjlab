#include "walk/walk_task.h"
#include "phybot/policy_runner.h"
#include "phybot/gravity.h"
#include "phybot/walk_obs.h"
#include "phybot/walk_action.h"
#include "phybot/data_package.h"
#include <cstring>
#include <stdexcept>

namespace phybot {

WalkTask::WalkTask(const std::string& model_path)
    : Task(TaskId::WALK, "walk"),
      model_path_(model_path),
      cfg_(walk_config()) {}

WalkTask::~WalkTask() = default;

void WalkTask::enter(DataPackage& io) {
  step_num_ = 0;
  last_action_.fill(0.0);
  history_.Reset();
  if (!runner_) {
    if (model_path_.empty())
      throw std::runtime_error("WalkTask: model_path is empty");
    runner_ = std::make_unique<PolicyRunner>(model_path_);
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

}  // namespace phybot
