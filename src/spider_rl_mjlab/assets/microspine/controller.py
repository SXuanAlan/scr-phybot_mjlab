"""Stochastic microspine controller — anchor-based shear constraint + adhesion.

Implements Jiang et al. 2018 spine-asperity interaction model as a stateful
step-mode event. Each env step, the controller:

1. Determines which end-effectors are in contact with the wall (with hysteresis)
2. Runs stochastic engagement/disengagement per spine
3. Computes shear force capacity from engaged spine states
4. Applies shear via anchor constraint (spring-damper clamped to capacity)
5. Applies normal via adhesion actuator ctrl
6. Caches privileged observations for future RMA training

Shear force model: spines lock into asperities, acting as rigid anchors.
A high-stiffness spring-damper pulls the EE back to its anchor position.
When the load exceeds the spine capacity, the anchor slides (spines slip
and re-engage), producing a force equal to the capacity limit.

Pattern follows mjlab's ``apply_body_impulse`` (events.py:359).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from spider_rl_mjlab.assets.microspine.config import MicrospineCfg
from spider_rl_mjlab.assets.microspine.physics import (
    compute_loading_angle_failure,
    compute_spine_force,
    compute_total_shear_force,
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

    Shear force uses an anchor-based constraint: when an EE contacts the wall,
    its wall-tangent position is recorded as the anchor. A stiff spring-damper
    resists displacement from the anchor, clamped to the engaged spine capacity.
    When load exceeds capacity, the anchor slides (slip with constant friction).

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
        self._debug = True
        self._debug_step_count = 0
        self._debug_interval_steps = 0  # computed on first call

        # Resolve entity and EE body IDs.
        self._asset = env.scene[asset_cfg.name]
        body_ids, _ = self._asset.find_bodies(list(mcfg.ee_body_names))
        self._ee_body_ids = body_ids  # local body indices within the entity

        # Find adhesion actuator ctrl indices in the global ctrl array.
        mj_model = env.sim.mj_model
        entity_name = asset_cfg.name
        self._adhesion_ctrl_ids = torch.tensor(
            [mj_model.actuator(f"{entity_name}/adhesion_{b}").id for b in mcfg.ee_body_names],
            dtype=torch.long,
        )

        # Store reference to sim data for direct ctrl writes.
        self._sim_data = env.sim.data

        # Compute wall inner face x-offset from env origin.
        wall_body_name = f"{wall_scene_cfg.name}/wall_body"
        wall_geom_name = f"{wall_scene_cfg.name}/wall_geom"
        wall_entity = env.scene[wall_scene_cfg.name]
        wall_center_x = float(mj_model.body(wall_body_name).pos[0]) + wall_entity.cfg.init_state.pos[0]
        wall_half_x = float(mj_model.geom(wall_geom_name).size[0])
        self._wall_x_offset = wall_center_x - wall_half_x

        # -- Force physics constants --
        self._tan_theta_th = math.tan(mcfg.theta_threshold)

        # Wall outward normal (unit, world frame). Shape (1, 1, 3) for broadcasting.
        self._wall_normal = torch.tensor([[[-1.0, 0.0, 0.0]]], device=self._device)

        # Gravity direction (unit vector, world frame). Shape (1, 1, 3).
        self._gravity_dir = torch.tensor([[[0.0, 0.0, -1.0]]], device=self._device)

        # Wall-up axis: project world-up onto wall tangent plane. Shape (1, 1, 3).
        world_up = -self._gravity_dir  # (1, 1, 3)
        up_dot_n = (world_up * self._wall_normal).sum(dim=-1, keepdim=True)
        wall_up = world_up - up_dot_n * self._wall_normal
        self._wall_up = wall_up / (wall_up.norm(dim=-1, keepdim=True) + 1e-8)  # unit (1, 1, 3)

        # -- Persistent state tensors --
        N, E, S = self._num_envs, self._n_ee, self._n_spines

        self._spine_engaged = torch.zeros(N, E, S, dtype=torch.bool, device=self._device)
        self._spine_travel = torch.zeros(N, E, S, dtype=torch.float32, device=self._device)
        self._ee_was_in_contact = torch.zeros(N, E, dtype=torch.bool, device=self._device)

        # Anchor position along wall-up per EE (scalar projection). (N, E)
        self._ee_anchor = torch.zeros(N, E, dtype=torch.float32, device=self._device)

        # Contact age counter per EE for D-term ramp-up. (N, E)
        self._contact_age = torch.zeros(N, E, dtype=torch.float32, device=self._device)

        # Integral accumulator for PID controller. (N, E)
        self._displacement_integral = torch.zeros(N, E, dtype=torch.float32, device=self._device)

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

    def set_debug(self, enabled: bool = True) -> None:
        """Enable or disable debug printing for env 0 contact state changes."""
        self._debug = enabled

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        microspine_cfg: MicrospineCfg,
        wall_cfg: SceneEntityCfg,
    ) -> None:
        """Tick microspine state and apply forces.

        Called every env step by the event manager (mode="step").
        ``env_ids`` is None for step events (operates on all envs).
        """
        del env_ids, asset_cfg, microspine_cfg, wall_cfg  # Unused at runtime.
        cfg = self._cfg
        N, E, S = self._num_envs, self._n_ee, self._n_spines

        # 1. Read EE positions and velocities (world frame).
        ee_pos = self._asset.data.body_link_pos_w[:, self._ee_body_ids, :]  # (N, E, 3)
        ee_vel = self._asset.data.body_link_lin_vel_w[:, self._ee_body_ids, :]  # (N, E, 3)

        # 2. Compute distance to wall plane per EE.
        env_origin_x = env.scene.env_origins[:, 0]  # (N,)
        wall_face_x = env_origin_x + self._wall_x_offset  # (N,)
        ee_x = ee_pos[:, :, 0]  # (N, E)
        dist_to_wall = (wall_face_x.unsqueeze(1) - ee_x).abs()  # (N, E)

        # 3. Contact determination with hysteresis.
        enter_threshold = cfg.contact_threshold
        exit_threshold = cfg.contact_threshold + cfg.contact_hysteresis
        newly_in_contact = (dist_to_wall < enter_threshold) & ~self._ee_was_in_contact
        lost_contact = (dist_to_wall > exit_threshold) & self._ee_was_in_contact
        in_contact = (self._ee_was_in_contact | newly_in_contact) & ~lost_contact

        # 4. Engagement logic.
        self._handle_new_contacts(newly_in_contact, ee_pos)
        self._handle_lost_contacts(lost_contact)

        # 4b. Loading angle check (Eq. 8): disengage spines being pulled off.
        # Skip for EEs that just engaged this step, or when speed is too low
        # (angle is undefined at near-zero velocities — noise dominates).
        away_sign = torch.sign(ee_pos[:, :, 0] - wall_face_x.unsqueeze(1))
        away_sign = torch.where(away_sign == 0, -torch.ones_like(away_sign), away_sign)
        v_pullout = (ee_vel[:, :, 0] * away_sign).clamp(min=0.0)
        v_shear = torch.norm(ee_vel[:, :, 1:], dim=-1)
        speed = torch.sqrt(v_pullout**2 + v_shear**2)
        angle_fail = compute_loading_angle_failure(v_pullout, v_shear, cfg.theta_threshold)
        moving_fast = speed > 0.5  # only check when EE is clearly being pulled off [m/s]
        engaged_ratio = self._spine_engaged.sum(dim=-1).float() / self._n_spines  # (N, E)
        highly_engaged = engaged_ratio > 0.9  # only check when >90% spines engaged
        angle_eligible = in_contact & ~newly_in_contact & moving_fast & highly_engaged
        angle_fail_spines = (angle_fail & angle_eligible).unsqueeze(-1).expand(-1, -1, S)
        n_before = self._spine_engaged.sum(dim=-1)  # for debug
        self._spine_engaged = self._spine_engaged & ~angle_fail_spines

        # DEBUG: log loading angle disengagement for env 0
        if self._debug:
            n_after = self._spine_engaged.sum(dim=-1)
            disengaged = n_before[0] - n_after[0]
            ee_names = self._cfg.ee_body_names
            for i in range(self._n_ee):
                if "hand" in ee_names[i] and disengaged[i] > 0:
                    print(f"[MS ANGLE] {ee_names[i]}: lost {int(disengaged[i])} spines, "
                          f"v_pull={v_pullout[0,i]:.3f} v_shear={v_shear[0,i]:.3f} speed={speed[0,i]:.3f}")

        # 5. Compute shear force capacity from engaged spines.
        shear_capacity = compute_total_shear_force(
            self._spine_travel, self._spine_engaged, cfg.K, cfg.d_stroke, cfg.F_max
        )  # (N, E) in Newtons — this is the MAX force the spines can provide

        # 6. Anchor-based shear constraint.
        # Project EE position and velocity onto wall-up axis (scalar).
        ee_along_up = (ee_pos * self._wall_up).sum(dim=-1)  # (N, E)
        vel_along_up = (ee_vel * self._wall_up).sum(dim=-1)  # (N, E)

        # PID force controller: smooth anchor constraint.
        # P: proportional to displacement (spring)
        # I: accumulated displacement (builds up steady-state holding force)
        # D: proportional to velocity (damping, ramped up over 0.5s)
        displacement = ee_along_up - self._ee_anchor  # (N, E)

        # I term: accumulate displacement, reset on lost contact.
        dt = env.step_dt
        self._displacement_integral = torch.where(
            in_contact,
            (self._displacement_integral + displacement * dt).clamp(-cfg.anchor_integral_limit, cfg.anchor_integral_limit),
            torch.zeros_like(self._displacement_integral),
        )

        # D term: ramp up over 0.5s to avoid contact impulse.
        self._contact_age = torch.where(in_contact, self._contact_age + 1.0, torch.zeros_like(self._contact_age))
        ramp_steps = 0.5 / dt
        d_scale = (self._contact_age / ramp_steps).clamp(0.0, 1.0)

        raw_force = (
            -cfg.anchor_kp * displacement
            - cfg.anchor_ki * self._displacement_integral
            - cfg.anchor_kd * vel_along_up * d_scale
        )  # (N, E)

        # Clamp to spine capacity (acts like max static/kinetic friction).
        applied_shear = raw_force.clamp(-shear_capacity, shear_capacity)  # (N, E)

        # Anchor sliding: when load exceeds capacity, update anchor so that
        # the P-term force equals the clamped value (spines slip and re-engage).
        # Also reset integral to prevent windup.
        overloaded = raw_force.abs() > shear_capacity
        new_anchor = ee_along_up + applied_shear / cfg.anchor_kp
        self._displacement_integral = torch.where(overloaded, torch.zeros_like(self._displacement_integral), self._displacement_integral)
        self._ee_anchor = torch.where(overloaded, new_anchor, self._ee_anchor)

        # Zero forces for EEs not in contact.
        contact_mask = in_contact.float()
        applied_shear = applied_shear * contact_mask

        # Normal force limit from loading angle.
        normal_limit = applied_shear.abs() * self._tan_theta_th * contact_mask

        # 7. Apply forces.
        # 7a. Shear via xfrc_applied along wall-up direction.
        shear_force_3d = self._wall_up * applied_shear.unsqueeze(-1)  # (N, E, 3)
        zero_torque = torch.zeros_like(shear_force_3d)
        self._asset.write_external_wrench_to_sim(
            forces=shear_force_3d,
            torques=zero_torque,
            body_ids=self._ee_body_ids,
        )

        # 7b. Normal via adhesion actuator (constraint-based pull-off resistance).
        adhesion_ctrl = (normal_limit / cfg.max_adhesion_gain).clamp(0.0, 1.0)
        self._sim_data.ctrl[:, self._adhesion_ctrl_ids] = adhesion_ctrl

        # 8. Update state and privileged obs.
        n_engaged = self._spine_engaged.sum(dim=-1)  # (N, E) int
        self._ee_was_in_contact = in_contact
        self._update_privileged_obs(n_engaged, applied_shear.abs(), normal_limit, in_contact, shear_capacity)

        # DEBUG: print env 0 hand states every ~1 second
        if self._debug:
            if self._debug_interval_steps == 0:
                self._debug_interval_steps = max(1, round(1.0 / env.step_dt))
            self._debug_step_count += 1
            if self._debug_step_count >= self._debug_interval_steps:
                self._debug_step_count = 0
                ee_names = self._cfg.ee_body_names
                for i in range(self._n_ee):
                    if "hand" not in ee_names[i]:
                        continue
                    state = "ON " if in_contact[0, i] else "OFF"
                    eng = int(n_engaged[0, i])
                    cap = shear_capacity[0, i].item()
                    f = applied_shear[0, i].abs().item()
                    disp = displacement[0, i].item() * 1000  # mm
                    print(f"[MS env0] {ee_names[i]}: {state} eng={eng}/{self._n_spines} cap={cap:.1f}N F={f:.2f}N disp={disp:.1f}mm")

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset spine state for specified environments."""
        self._spine_engaged[env_ids] = False
        self._spine_travel[env_ids] = 0.0
        self._ee_was_in_contact[env_ids] = False
        self._ee_anchor[env_ids] = 0.0
        self._contact_age[env_ids] = 0.0
        self._displacement_integral[env_ids] = 0.0
        self._privileged_obs[env_ids] = 0.0

        # Zero adhesion ctrl for reset envs.
        self._sim_data.ctrl[env_ids[:, None], self._adhesion_ctrl_ids[None, :]] = 0.0

        # Zero shear forces (xfrc_applied) for reset envs.
        zero_forces = torch.zeros(len(env_ids), self._n_ee, 3, device=self._device)
        self._asset.write_external_wrench_to_sim(
            forces=zero_forces,
            torques=zero_forces,
            env_ids=env_ids,
            body_ids=self._ee_body_ids,
        )

        if self._cfg.randomize_surface:
            self._randomize_surface(env_ids)

    def get_privileged_obs(self) -> torch.Tensor:
        """Return cached privileged observations.

        Shape: (num_envs, n_ee * 5) where per-EE values are:
          [engaged_ratio, shear_utilization, normal_utilization, is_in_contact, surface_quality]
        """
        return self._privileged_obs

    # -- Internal methods -----------------------------------------------------

    def _handle_new_contacts(self, newly_in_contact: torch.Tensor, ee_pos: torch.Tensor) -> None:
        """Sample spine engagement and set anchor for newly contacting EEs."""
        if not newly_in_contact.any():
            return

        cfg = self._cfg

        # Set anchor to current EE position along wall-up.
        ee_along_up = (ee_pos * self._wall_up).sum(dim=-1)  # (N, E)
        self._ee_anchor = torch.where(newly_in_contact, ee_along_up, self._ee_anchor)

        # Expand mask to per-spine: (N, E, S)
        mask = newly_in_contact.unsqueeze(-1).expand(-1, -1, self._n_spines)

        # P(engage) = (1 - alpha) + alpha * (1 - exp(-lambda * d_stroke))
        alpha = self._surface_alpha.unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
        lam_m = self._surface_lambda.unsqueeze(-1).unsqueeze(-1) * 1000.0  # per m
        p_engage = (1.0 - alpha) + alpha * (1.0 - torch.exp(-lam_m * cfg.d_stroke))
        p_engage = p_engage.expand_as(mask)

        # Sample engagement.
        engaged = torch.bernoulli(p_engage).bool() & mask
        self._spine_engaged = torch.where(mask, engaged, self._spine_engaged)

        # Sample travel distance from Exponential(lambda) for engaged spines.
        lam_sample = self._surface_lambda.unsqueeze(-1).unsqueeze(-1) * 1000.0
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
        applied_shear: torch.Tensor,
        normal_limit: torch.Tensor,
        in_contact: torch.Tensor,
        shear_capacity: torch.Tensor,
    ) -> None:
        """Update the cached privileged observation tensor.

        Per-EE layout (5 values):
          0: engaged_ratio        — n_engaged / n_spines
          1: shear_utilization    — applied shear / capacity (0 if no capacity)
          2: normalized_normal    — normal force limit / max_adhesion_gain
          3: is_in_contact        — 1.0 if in contact, 0.0 otherwise
          4: surface_quality      — η·λ / 0.49
        """
        cfg = self._cfg
        E = self._n_ee

        engaged_ratio = n_engaged.float() / cfg.n_spines
        shear_util = (applied_shear / (shear_capacity + 1e-6)).clamp(0.0, 1.0)
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
        ], dim=-1)  # (N, E, 5)

        self._privileged_obs = obs.reshape(self._num_envs, -1)
