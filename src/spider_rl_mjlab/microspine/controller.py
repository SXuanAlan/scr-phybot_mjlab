"""Stochastic microspine controller that modulates MuJoCo adhesion actuators.

Implements Jiang et al. 2018 spine-asperity interaction model as a stateful
step-mode event. Each env step, the controller:

1. Determines which end-effectors are in contact with the wall
2. Runs stochastic engagement/disengagement per spine
3. Computes adhesion activation from engagement state
4. Writes activation to data.ctrl for the adhesion actuators
5. Caches privileged observations for future RMA training

Pattern follows mjlab's ``apply_body_impulse`` (events.py:359).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from spider_rl_mjlab.microspine.config import MicrospineCfg
from spider_rl_mjlab.microspine.physics import (
    compute_adhesion_activation,
    compute_engagement_probability,
    compute_spine_force,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg


# Number of privileged observation values per end-effector.
_OBS_PER_EE = 5


class MicrospineController:
    """Stateful step-mode event for microspine adhesion.

    Uses MuJoCo native adhesion actuators (constraint-based, zero delay)
    with stochastic engagement modeled after Jiang et al. 2018.

    Use with ``mode="step"`` in the event manager.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        mcfg: MicrospineCfg = cfg.params["microspine_cfg"]
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]

        self._cfg = mcfg
        self._device = env.device
        self._num_envs = env.num_envs
        self._n_ee = len(mcfg.ee_body_names)
        self._n_spines = mcfg.n_spines

        # Resolve entity and EE body IDs.
        self._asset = env.scene[asset_cfg.name]
        body_ids, _ = self._asset.find_bodies(list(mcfg.ee_body_names))
        self._ee_body_ids = body_ids  # local body indices within the entity

        # Find adhesion actuator ctrl indices in the global ctrl array.
        # In compiled scenes, actuator names are prefixed with the entity name.
        mj_model = env.sim.mj_model
        entity_name = asset_cfg.name  # e.g. "robot"
        self._adhesion_ctrl_ids: list[int] = []
        for body_name in mcfg.ee_body_names:
            act_name = f"{entity_name}/adhesion_{body_name}"
            act_id = mj_model.actuator(act_name).id
            self._adhesion_ctrl_ids.append(act_id)

        # Store reference to sim data for direct ctrl writes.
        self._sim_data = env.sim.data

        # -- Persistent state tensors --
        N, E, S = self._num_envs, self._n_ee, self._n_spines

        self._spine_engaged = torch.zeros(N, E, S, dtype=torch.bool, device=self._device)
        self._spine_travel = torch.zeros(N, E, S, dtype=torch.float32, device=self._device)
        self._ee_was_in_contact = torch.zeros(N, E, dtype=torch.bool, device=self._device)

        # Per-env surface parameters.
        self._surface_alpha = torch.full((N,), mcfg.alpha, device=self._device)
        self._surface_lambda = torch.full((N,), mcfg.lambda_, device=self._device)
        self._surface_eta = torch.full((N,), mcfg.eta, device=self._device)

        # Privileged observation cache: (num_envs, n_ee * _OBS_PER_EE).
        self._privileged_obs = torch.zeros(N, E * _OBS_PER_EE, device=self._device)

        # Randomize surface params for all envs at init.
        if mcfg.randomize_surface:
            self._randomize_surface(torch.arange(N, device=self._device))

    # -- Public API -----------------------------------------------------------

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        microspine_cfg: MicrospineCfg,
    ) -> None:
        """Tick microspine state and update adhesion ctrl.

        Called every env step by the event manager (mode="step").
        ``env_ids`` is None for step events (operates on all envs).
        """
        del env_ids, asset_cfg, microspine_cfg  # Unused at runtime.
        cfg = self._cfg
        N, E, S = self._num_envs, self._n_ee, self._n_spines

        # 1. Read EE positions (world frame).
        ee_pos = self._asset.data.body_link_pos_w[:, self._ee_body_ids, :]  # (N, E, 3)

        # 2. Compute distance to wall plane per EE.
        # Wall inner face is at x = env_origin_x + wall_x_offset.
        env_origin_x = env.scene.env_origins[:, 0]  # (N,)
        wall_face_x = env_origin_x + cfg.wall_x_offset  # (N,)
        ee_x = ee_pos[:, :, 0]  # (N, E)
        dist_to_wall = (wall_face_x.unsqueeze(1) - ee_x).abs()  # (N, E)

        # 3. Contact determination.
        in_contact = dist_to_wall < cfg.contact_threshold  # (N, E)
        newly_in_contact = in_contact & ~self._ee_was_in_contact  # (N, E)
        lost_contact = ~in_contact & self._ee_was_in_contact  # (N, E)

        # 4. Engagement logic.
        self._handle_new_contacts(newly_in_contact)
        self._handle_lost_contacts(lost_contact)
        self._check_asperity_hold()

        # 5. Compute adhesion activation per EE.
        n_engaged = self._spine_engaged.sum(dim=-1)  # (N, E) int
        activation = compute_adhesion_activation(
            n_engaged=n_engaged,
            n_spines=S,
            alpha=self._surface_alpha,
            lambda_=self._surface_lambda,
            eta=self._surface_eta,
        )  # (N, E), values in [0, 1]

        # Zero activation for EEs not in contact.
        activation = activation * in_contact.float()

        # 6. Write adhesion ctrl to sim data.
        for i, ctrl_id in enumerate(self._adhesion_ctrl_ids):
            self._sim_data.ctrl[:, ctrl_id] = activation[:, i]

        # 7. Update state and privileged obs.
        self._ee_was_in_contact = in_contact
        self._update_privileged_obs(n_engaged, activation, in_contact)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset spine state for specified environments."""
        self._spine_engaged[env_ids] = False
        self._spine_travel[env_ids] = 0.0
        self._ee_was_in_contact[env_ids] = False
        self._privileged_obs[env_ids] = 0.0

        # Zero adhesion ctrl for reset envs.
        for ctrl_id in self._adhesion_ctrl_ids:
            self._sim_data.ctrl[env_ids, ctrl_id] = 0.0

        if self._cfg.randomize_surface:
            self._randomize_surface(env_ids)

    def get_privileged_obs(self) -> torch.Tensor:
        """Return cached privileged observations.

        Shape: (num_envs, n_ee * 5) where per-EE values are:
          [engaged_ratio, shear_capacity, normal_capacity, is_in_contact, surface_quality]
        """
        return self._privileged_obs

    # -- Internal methods -----------------------------------------------------

    def _handle_new_contacts(self, newly_in_contact: torch.Tensor) -> None:
        """Sample spine engagement for newly contacting EEs."""
        if not newly_in_contact.any():
            return

        cfg = self._cfg
        # Expand mask to per-spine: (N, E, S)
        mask = newly_in_contact.unsqueeze(-1).expand(-1, -1, self._n_spines)

        # Each spine has probability (1-alpha) of immediate engagement
        # plus alpha probability of engaging after some travel.
        # For simplicity in the simplified model: each spine engages with
        # probability derived from the two-phase distribution.
        # P(engage) = (1 - alpha) + alpha * (1 - exp(-lambda * d_stroke))
        alpha = self._surface_alpha.unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
        lam_m = self._surface_lambda.unsqueeze(-1).unsqueeze(-1) * 1000.0  # per m
        p_engage = (1.0 - alpha) + alpha * (1.0 - torch.exp(-lam_m * cfg.d_stroke))
        p_engage = p_engage.expand_as(mask)

        # Sample engagement.
        engaged = torch.bernoulli(p_engage).bool() & mask
        self._spine_engaged = torch.where(mask, engaged, self._spine_engaged)

        # Sample travel distance from Exponential(lambda) for engaged spines,
        # clamped to [0, d_stroke].
        lam_sample = self._surface_lambda.unsqueeze(-1).unsqueeze(-1) * 1000.0  # per m
        # Exponential sample: -ln(U) / lambda
        u = torch.rand_like(self._spine_travel).clamp(min=1e-7)
        travel = -torch.log(u) / lam_sample
        travel = travel.clamp(0.0, cfg.d_stroke)
        # Immediate engagement (delta term): travel = 0
        is_immediate = torch.bernoulli((1.0 - alpha).expand_as(mask)).bool()
        travel = torch.where(is_immediate & mask, torch.zeros_like(travel), travel)

        self._spine_travel = torch.where(mask & engaged, travel, self._spine_travel)

    def _handle_lost_contacts(self, lost_contact: torch.Tensor) -> None:
        """Disengage all spines for EEs that lost wall contact."""
        if not lost_contact.any():
            return
        mask = lost_contact.unsqueeze(-1).expand(-1, -1, self._n_spines)
        self._spine_engaged = torch.where(mask, torch.zeros_like(self._spine_engaged), self._spine_engaged)
        self._spine_travel = torch.where(mask, torch.zeros_like(self._spine_travel), self._spine_travel)

    def _check_asperity_hold(self) -> None:
        """Check if engaged spines can still hold their asperities.

        Spines that exceed the asperity strength limit disengage.
        """
        cfg = self._cfg
        # Compute force per engaged spine.
        force = compute_spine_force(
            self._spine_travel, cfg.K, cfg.d_stroke, cfg.F_max
        )  # (N, E, S)

        # Probability of holding.
        eta = self._surface_eta.unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
        hold_prob = compute_engagement_probability(force, eta)  # (N, E, S)

        # Only check engaged spines — unengaged spines stay unengaged.
        check_mask = self._spine_engaged
        holds = torch.bernoulli(hold_prob).bool()
        self._spine_engaged = self._spine_engaged & (~check_mask | holds)

    def _randomize_surface(self, env_ids: torch.Tensor) -> None:
        """Re-sample surface asperity parameters for given environments."""
        cfg = self._cfg
        n = len(env_ids)
        self._surface_alpha[env_ids] = (
            torch.rand(n, device=self._device) * (cfg.alpha_range[1] - cfg.alpha_range[0])
            + cfg.alpha_range[0]
        )
        self._surface_lambda[env_ids] = (
            torch.rand(n, device=self._device) * (cfg.lambda_range[1] - cfg.lambda_range[0])
            + cfg.lambda_range[0]
        )
        self._surface_eta[env_ids] = (
            torch.rand(n, device=self._device) * (cfg.eta_range[1] - cfg.eta_range[0])
            + cfg.eta_range[0]
        )

    def _update_privileged_obs(
        self,
        n_engaged: torch.Tensor,
        activation: torch.Tensor,
        in_contact: torch.Tensor,
    ) -> None:
        """Update the cached privileged observation tensor.

        Per-EE layout (5 values):
          0: engaged_ratio    — n_engaged / n_spines
          1: shear_capacity   — activation (normalized adhesion strength)
          2: normal_capacity  — activation * preload_ratio
          3: is_in_contact    — 1.0 if in contact, 0.0 otherwise
          4: surface_quality  — normalized f(eta, lambda)
        """
        cfg = self._cfg
        E = self._n_ee

        engaged_ratio = n_engaged.float() / cfg.n_spines  # (N, E)
        preload_ratio = cfg.preload_force / cfg.F_max
        normal_cap = activation * preload_ratio  # (N, E)
        surface_quality = (
            self._surface_eta.unsqueeze(-1) * self._surface_lambda.unsqueeze(-1) / 0.49
        ).clamp(0.0, 1.0)  # (N, E)

        # Pack into flat tensor: (N, E*5)
        obs = torch.stack([
            engaged_ratio,
            activation,
            normal_cap,
            in_contact.float(),
            surface_quality.expand(-1, E),
        ], dim=-1)  # (N, E, 5)

        self._privileged_obs = obs.reshape(self._num_envs, -1)
