"""Microspine adhesion controller.

Implements Jiang et al. 2018 spine-asperity interaction model as a stateful
step-mode event. Also handles external force application and contact damping
in a single xfrc_applied write to avoid conflicts with other events.

Each env step, the controller:
  1. Detects wall contact via gap-based proximity (with hysteresis)
  2. Runs stochastic spine engagement/disengagement
  3. Computes shear force capacity from engaged spines
  4. Sets adhesion actuator ctrl proportional to normal holding capacity
  5. Writes combined external force + contact damping to xfrc_applied
  6. Caches privileged observations for RMA training
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from spider_rl_mjlab.assets.microspine.config import MicrospineCfg
from spider_rl_mjlab.assets.microspine.physics import (
    compute_total_shear_force,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

_OBS_PER_EE = 5


class MicrospineController:
    """Stateful step-mode event for microspine adhesion + force application.

    Merges external force and contact damping into a single xfrc_applied write
    so there is no conflict with other events.

    Use with ``mode="step"`` in the event manager.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        mcfg: MicrospineCfg = cfg.params["microspine_cfg"]
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        wall_scene_cfg: SceneEntityCfg = cfg.params["wall_cfg"]

        self._cfg = mcfg
        self._device = env.device
        self._num_envs = env.num_envs
        self._n_ee = len(mcfg.ee_body_names)
        self._n_spines = mcfg.n_spines

        # External force params (from event config).
        self._ext_force_phase1 = torch.tensor(
            cfg.params.get("total_force", (0.0, 0.0, 0.0)),
            device=self._device, dtype=torch.float32,
        )
        self._ext_force_phase2 = torch.tensor(
            cfg.params.get("total_force_after", (0.0, 0.0, 0.0)),
            device=self._device, dtype=torch.float32,
        )
        self._ext_duration = cfg.params.get("duration", 1e9)
        self._k_damp = cfg.params.get("k_damp", 5000.0)

        # Resolve entity and body IDs.
        self._asset = env.scene[asset_cfg.name]
        body_ids, _ = self._asset.find_bodies(list(mcfg.ee_body_names))
        self._ee_body_ids = body_ids

        # Adhesion actuator ctrl indices.
        mj_model = env.sim.mj_model
        entity_name = asset_cfg.name
        self._adhesion_ctrl_ids = torch.tensor(
            [mj_model.actuator(f"{entity_name}/adhesion_{b}").id
             for b in mcfg.ee_body_names],
            dtype=torch.long,
        )

        self._sim_data = env.sim.data

        # Wall geometry: inner face x = env_origin_x + wall_x_offset.
        wall_geom_name = f"{wall_scene_cfg.name}/wall_geom"
        wall_half_x = float(mj_model.geom(wall_geom_name).size[0])
        wall_entity = env.scene[wall_scene_cfg.name]
        self._wall_x_offset = float(wall_entity.cfg.init_state.pos[0]) - wall_half_x

        # Physics constants.
        self._tan_theta_th = math.tan(mcfg.theta_threshold)

        # EE surface offset for gap calculation (from collision geom).
        self._ee_surface_offsets = torch.zeros(self._n_ee, dtype=torch.float32, device=self._device)
        for ee_idx, body_name in enumerate(mcfg.ee_body_names):
            geom = mj_model.geom(f"{entity_name}/{body_name}_collision")
            self._ee_surface_offsets[ee_idx] = abs(float(geom.pos[0])) + float(geom.size[0])

        # Cache global body IDs for direct xfrc_applied access (used for += damping).
        self._global_body_ids = self._asset.data.indexing.body_ids[self._ee_body_ids]

        # Persistent state.
        N, E, S = self._num_envs, self._n_ee, self._n_spines
        self._spine_engaged = torch.zeros(N, E, S, dtype=torch.bool, device=self._device)
        self._spine_travel = torch.zeros(N, E, S, dtype=torch.float32, device=self._device)
        self._ee_was_in_contact = torch.zeros(N, E, dtype=torch.bool, device=self._device)

        # Surface parameters (domain-randomizable).
        self._surface_alpha = torch.full((N,), mcfg.alpha, device=self._device)
        self._surface_lambda = torch.full((N,), mcfg.lambda_, device=self._device)
        self._surface_eta = torch.full((N,), mcfg.eta, device=self._device)

        # Privileged observation cache.
        self._privileged_obs = torch.zeros(N, E * _OBS_PER_EE, device=self._device)

        if mcfg.randomize_surface:
            self._randomize_surface(torch.arange(N, device=self._device))

    # -- Public API -----------------------------------------------------------

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        **kwargs,
    ) -> None:
        """Tick microspine state, apply forces, sync adhesion actuators."""
        cfg = self._cfg
        N, E, S = self._num_envs, self._n_ee, self._n_spines

        # 1. Read EE state (world frame).
        ee_pos = self._asset.data.body_link_pos_w[:, self._ee_body_ids, :]   # (N, E, 3)
        ee_lin_vel = self._asset.data.body_link_lin_vel_w[:, self._ee_body_ids, :]  # (N, E, 3)
        ee_ang_vel = self._asset.data.body_link_ang_vel_w[:, self._ee_body_ids, :]  # (N, E, 3)

        # 2. Gap-based contact detection with hysteresis.
        wall_face_x = env.scene.env_origins[:, 0] + self._wall_x_offset  # (N,)
        gap = (ee_pos[:, :, 0] - wall_face_x.unsqueeze(1)).abs() - self._ee_surface_offsets  # (N, E)

        in_contact = torch.where(
            self._ee_was_in_contact,
            (gap <= cfg.contact_threshold + cfg.contact_hysteresis) &
            (gap >= -(cfg.max_penetration + cfg.penetration_hysteresis)),
            (gap <= cfg.contact_threshold) & (gap >= -cfg.max_penetration),
        )
        newly_in_contact = in_contact & ~self._ee_was_in_contact
        lost_contact = ~in_contact & self._ee_was_in_contact

        # 3. Spine engagement.
        self._handle_new_contacts(newly_in_contact)
        self._handle_lost_contacts(lost_contact)

        # 4. Shear force capacity.
        shear_capacity = compute_total_shear_force(
            self._spine_travel, self._spine_engaged, cfg.K, cfg.d_stroke, cfg.F_max,
        )  # (N, E)

        # 5. Adhesion actuator.
        contact_mask = in_contact.float()  # (N, E)
        normal_limit = shear_capacity * self._tan_theta_th * contact_mask  # (N, E)
        adhesion_ctrl = (normal_limit / cfg.max_adhesion_gain).clamp(0.0, 1.0)
        self._sim_data.ctrl[:, self._adhesion_ctrl_ids] = adhesion_ctrl

        # 6. Freeze robot on contact: zero root + joint velocities.
        any_in_contact = in_contact.any(dim=-1)  # (N,)
        mask6 = any_in_contact.unsqueeze(-1)  # (N, 1)
        root_vel = self._asset.data.root_link_vel_w  # (N, 6)
        self._asset.write_root_link_velocity_to_sim(
            torch.where(mask6, torch.zeros_like(root_vel), root_vel)
        )
        joint_vel = self._asset.data.joint_vel  # (N, num_joints)
        self._asset.write_joint_velocity_to_sim(
            torch.where(mask6, torch.zeros_like(joint_vel), joint_vel)
        )

        # 7. Update state and privileged obs.
        n_engaged = self._spine_engaged.sum(dim=-1)
        self._ee_was_in_contact = in_contact
        self._update_privileged_obs(n_engaged, normal_limit, in_contact, shear_capacity)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset spine state for specified environments."""
        self._spine_engaged[env_ids] = False
        self._spine_travel[env_ids] = 0.0
        self._ee_was_in_contact[env_ids] = False
        self._privileged_obs[env_ids] = 0.0
        self._sim_data.ctrl[env_ids[:, None], self._adhesion_ctrl_ids[None, :]] = 0.0
        if self._cfg.randomize_surface:
            self._randomize_surface(env_ids)

    def get_privileged_obs(self) -> torch.Tensor:
        """Return cached privileged observations.

        Shape: (num_envs, n_ee * 5) with per-EE values:
          [engaged_ratio, shear_utilization, normalized_normal, is_in_contact, surface_quality]
        """
        return self._privileged_obs

    # -- Internal methods -----------------------------------------------------

    def _handle_new_contacts(self, newly_in_contact: torch.Tensor) -> None:
        """Sample spine engagement for newly contacting EEs (no CPU-GPU sync)."""
        cfg = self._cfg
        mask = newly_in_contact.unsqueeze(-1).expand(-1, -1, self._n_spines)

        alpha = self._surface_alpha.unsqueeze(-1).unsqueeze(-1)
        lam_m = self._surface_lambda.unsqueeze(-1).unsqueeze(-1) * 1000.0
        p_engage = (1.0 - alpha) + alpha * (1.0 - torch.exp(-lam_m * cfg.d_stroke))

        engaged = torch.bernoulli(p_engage.expand_as(mask)).bool() & mask
        self._spine_engaged = torch.where(mask, engaged, self._spine_engaged)

        u = torch.rand_like(self._spine_travel).clamp(min=1e-7)
        travel = (-torch.log(u) / lam_m).clamp(0.0, cfg.d_stroke)
        is_immediate = torch.bernoulli((1.0 - alpha).expand_as(mask)).bool()
        travel = torch.where(is_immediate & mask, torch.zeros_like(travel), travel)
        self._spine_travel = torch.where(mask & engaged, travel, self._spine_travel)

    def _handle_lost_contacts(self, lost_contact: torch.Tensor) -> None:
        """Disengage all spines for EEs that lost wall contact (no CPU-GPU sync)."""
        mask = lost_contact.unsqueeze(-1).expand(-1, -1, self._n_spines)
        self._spine_engaged = self._spine_engaged & ~mask
        self._spine_travel = torch.where(mask, torch.zeros_like(self._spine_travel), self._spine_travel)

    def _randomize_surface(self, env_ids: torch.Tensor) -> None:
        """Re-sample surface asperity parameters for given environments."""
        cfg = self._cfg
        n = len(env_ids)
        self._surface_alpha[env_ids] = torch.rand(n, device=self._device) * (cfg.alpha_range[1] - cfg.alpha_range[0]) + cfg.alpha_range[0]
        self._surface_lambda[env_ids] = torch.rand(n, device=self._device) * (cfg.lambda_range[1] - cfg.lambda_range[0]) + cfg.lambda_range[0]
        self._surface_eta[env_ids] = torch.rand(n, device=self._device) * (cfg.eta_range[1] - cfg.eta_range[0]) + cfg.eta_range[0]

    def _update_privileged_obs(
        self,
        n_engaged: torch.Tensor,
        normal_limit: torch.Tensor,
        in_contact: torch.Tensor,
        shear_capacity: torch.Tensor,
    ) -> None:
        """Update the cached privileged observation tensor."""
        cfg = self._cfg
        E = self._n_ee

        engaged_ratio = n_engaged.float() / cfg.n_spines
        shear_util = (shear_capacity / (cfg.F_max * cfg.n_spines + 1e-6)).clamp(0.0, 1.0)
        normalized_normal = (normal_limit / cfg.max_adhesion_gain).clamp(0.0, 1.0)
        surface_quality = (
            self._surface_eta.unsqueeze(-1) * self._surface_lambda.unsqueeze(-1) / 0.49
        ).clamp(0.0, 1.0)

        obs = torch.stack([
            engaged_ratio,
            shear_util,
            normalized_normal,
            in_contact.float(),
            surface_quality.expand(-1, E),
        ], dim=-1)
        self._privileged_obs = obs.reshape(self._num_envs, -1)
