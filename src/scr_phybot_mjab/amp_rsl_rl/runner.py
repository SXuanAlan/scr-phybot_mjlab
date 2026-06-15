"""mjlab adapter for amp-rsl-rl's AMP runner."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from amp_rsl_rl.runners import AMPOnPolicyRunner


class MjlabAmpRunner(AMPOnPolicyRunner):
  """Thin compatibility layer between mjlab configs and amp-rsl-rl."""

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    sim_cfg = getattr(env.cfg, "sim", None)
    mujoco_cfg = getattr(sim_cfg, "mujoco", None)
    if sim_cfg is not None and mujoco_cfg is not None and not hasattr(sim_cfg, "dt"):
      sim_cfg.dt = mujoco_cfg.timestep
    super().__init__(env, train_cfg, log_dir=log_dir, device=device)

  def _policy_module(self):
    return self.alg.actor if hasattr(self.alg, "actor") else self.alg.actor_critic

  def save(self, path, infos=None, save_onnx=False):
    env_state = {
      "common_step_counter": self.env.unwrapped.common_step_counter,
    }
    infos = {**(infos or {}), "env_state": env_state}
    return super().save(path, infos=infos, save_onnx=save_onnx)

  def load(
    self,
    path,
    load_optimizer=True,
    weights_only=False,
    load_cfg=None,
    strict=True,
    map_location=None,
    **kwargs,
  ):
    infos = super().load(
      path,
      load_optimizer=load_optimizer,
      weights_only=weights_only,
      load_cfg=load_cfg,
      strict=strict,
      map_location=map_location,
      **kwargs,
    )
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos

  def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    jit_model = self._policy_module().as_jit()
    jit_model.to("cpu")
    torch.jit.script(jit_model).save(os.path.join(path, filename))

  def export_policy_to_onnx(
    self,
    path: str,
    filename: str = "policy.onnx",
    verbose: bool = False,
  ) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    onnx_model = self._policy_module().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,
      output_names=onnx_model.output_names,
    )
