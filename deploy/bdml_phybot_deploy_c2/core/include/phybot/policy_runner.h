#pragma once
#include <memory>
#include <string>

namespace phybot {
// Thin JIT-script wrapper. Torch headers are NOT included here — only in policy_runner.cpp.
class PolicyRunner {
 public:
  explicit PolicyRunner(const std::string& model_path);
  ~PolicyRunner();
  // obs[obs_dim] float → action_out[act_dim] float (raw network output, no scaling)
  void forward(const float* obs, int obs_dim, float* action_out, int act_dim);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace phybot
