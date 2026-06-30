#include "phybot/policy_runner.h"
#include <torch/script.h>
#include <vector>

namespace phybot {

struct PolicyRunner::Impl {
  torch::jit::script::Module model;
};

PolicyRunner::PolicyRunner(const std::string& model_path)
    : impl_(new Impl) {
  torch::autograd::GradMode::set_enabled(false);
  impl_->model = torch::jit::load(model_path, torch::kCPU);
}

PolicyRunner::~PolicyRunner() = default;

void PolicyRunner::forward(const float* obs, int obs_dim,
                           float* action_out, int act_dim) {
  // Replicate official rl_deploy_cpg::Step() tensor construction exactly.
  auto t = torch::zeros({obs_dim}, torch::kFloat);
  auto acc = t.accessor<float, 1>();
  for (int i = 0; i < obs_dim; ++i) acc[i] = obs[i];
  t = t.to(torch::kCPU);
  t = t.unsqueeze(0);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(t);
  auto out = impl_->model.forward(inputs).toTensor();
  if (out.dim() > 1 && out.size(0) == 1) out = out.squeeze(0);
  auto out_acc = out.accessor<float, 1>();
  for (int i = 0; i < act_dim; ++i) action_out[i] = out_acc[i];
}

}  // namespace phybot
