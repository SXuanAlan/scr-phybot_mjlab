"""Microspine adhesion controller — contactfilter + adhesion actuator.

Implements Jiang et al. 2018 spine-asperity interaction model as a stateful
step-mode event. Each env step, the controller:

1. Determines which end-effectors are in contact with the wall (with hysteresis)
2. Runs stochastic engagement/disengagement per spine
3. Checks loading angle for spine pull-out (Eq. 8)
4. Computes shear force capacity from engaged spine states
5. Syncs engagement state to the contactfilter (Warp kernel, runs every substep)
6. Sets adhesion actuator ctrl proportional to normal holding capacity
7. Caches privileged observations for future RMA training

Shear friction is handled by the contactfilter callback at every physics substep
(0.002s), which rotates the contact frame for anisotropic friction: high along
wall-up (resists downward sliding), low laterally. This is constraint-based with
zero delay — the friction force is recomputed by the solver every substep.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from spider_rl_mjlab.assets.microspine.config import MicrospineCfg
from spider_rl_mjlab.assets.microspine.contact_filter import MicrospineContactFilter
from spider_rl_mjlab.assets.microspine.physics import (
    compute_loading_angle_failure,
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

    Uses MuJoCo Warp's contactfilter callback for anisotropic shear friction
    (recomputed every physics substep) and native adhesion actuators for
    normal adhesion (constraint-based, zero delay).

    Stochastic spine engagement follows Jiang et al. 2018.

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
        self._debug_interval_steps = 0

        # Resolve entity and EE body IDs.
        self._asset = env.scene[asset_cfg.name]
        body_ids, _ = self._asset.find_bodies(list(mcfg.ee_body_names))
        self._ee_body_ids = body_ids

        # Find adhesion actuator ctrl indices in the global ctrl array.
        mj_model = env.sim.mj_model
        entity_name = asset_cfg.name
        self._adhesion_ctrl_ids = torch.tensor(
            [mj_model.actuator(f"{entity_name}/adhesion_{b}").id for b in mcfg.ee_body_names],
            dtype=torch.long,
        )

        self._sim_data = env.sim.data

        # Wall geometry: half-size from model (baked into XML), center read at runtime.
        # mj_model.body(...).pos[0] returns 0 for runtime-positioned entities (position
        # is applied via free joint qpos, not MJCF body pos), so we read the wall
        # entity's world-frame body position at each step instead.
        wall_geom_name = f"{wall_scene_cfg.name}/wall_geom"
        self._wall_half_x = float(mj_model.geom(wall_geom_name).size[0])
        self._wall_entity = env.scene[wall_scene_cfg.name]

        # Force physics constants.
        self._tan_theta_th = math.tan(mcfg.theta_threshold)

        # Wall outward normal (unit, world frame). Shape (1, 1, 3) for broadcasting.
        self._wall_normal = torch.tensor([[[-1.0, 0.0, 0.0]]], device=self._device)

        # Wall-up axis: project world-up onto wall tangent plane. Shape (1, 1, 3).
        world_up = torch.tensor([[[0.0, 0.0, 1.0]]], device=self._device)
        up_dot_n = (world_up * self._wall_normal).sum(dim=-1, keepdim=True)
        wall_up = world_up - up_dot_n * self._wall_normal
        self._wall_up = wall_up / (wall_up.norm(dim=-1, keepdim=True) + 1e-8)

        # -- Build contactfilter --
        # Geom-to-EE lookup table: geom_to_ee[geom_id] = ee_index, or -1.
        ngeom = mj_model.ngeom
        geom_to_ee = [-1] * ngeom
        self._hand_geom_ids_by_ee = torch.full((self._n_ee,), -1, dtype=torch.long, device=self._device)
        self._ee_surface_offsets = torch.zeros((self._n_ee,), dtype=torch.float32, device=self._device)
        for ee_idx, body_name in enumerate(mcfg.ee_body_names):
            geom = mj_model.geom(f"{entity_name}/{body_name}_collision")
            geom_id = geom.id
            geom_to_ee[geom_id] = ee_idx
            geom_radius = float(geom.size[0])
            # Distance from body origin to the front-most point of the hand sphere
            # along the wall-normal axis. This corrects the body-origin gap estimate.
            self._ee_surface_offsets[ee_idx] = abs(float(geom.pos[0])) + geom_radius
            if "hand" in body_name:
                self._hand_geom_ids_by_ee[ee_idx] = geom_id

        wall_geom_id = mj_model.geom(wall_geom_name).id
        self._wall_geom_id = wall_geom_id
        self._geom_to_ee_torch = torch.tensor(geom_to_ee, dtype=torch.long, device=self._device)

        self._contact_filter = MicrospineContactFilter(
            geom_to_ee=geom_to_ee,
            wall_geom_id=wall_geom_id,
            num_envs=env.num_envs,
            n_ee=self._n_ee,
            engaged_friction=mcfg.engaged_friction,
            lateral_friction=mcfg.lateral_friction,
            device=self._device,
        )

        # Register callback and re-capture CUDA graph.
        env.sim.wp_model.callback.contactfilter = self._contact_filter
        env.sim.create_graph()

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
        """Tick microspine state and sync to contactfilter + adhesion actuators.

        Called every env step by the event manager (mode="step").
        ``env_ids`` is None for step events (operates on all envs).
        """
        del env_ids, asset_cfg, microspine_cfg, wall_cfg
        cfg = self._cfg
        N, E, S = self._num_envs, self._n_ee, self._n_spines

        # 1. Read EE positions and velocities (world frame).
        ee_pos = self._asset.data.body_link_pos_w[:, self._ee_body_ids, :]  # (N, E, 3)
        ee_vel = self._asset.data.body_link_lin_vel_w[:, self._ee_body_ids, :]  # (N, E, 3)

        # 2. Distance-based hand-wall proximity with hysteresis.
        # body_link_pos_w is absolute world frame, already includes env origins.
        wall_center_x = self._wall_entity.data.body_link_pos_w[:, 0, 0]  # (N,)
        wall_face_x = wall_center_x - self._wall_half_x  # (N,)
        hand_wall_gap = (ee_pos[:, :, 0] - wall_face_x.unsqueeze(1)).abs() - self._ee_surface_offsets.unsqueeze(0)
        contact_limit = torch.full_like(hand_wall_gap, cfg.contact_threshold)
        release_limit = torch.full_like(hand_wall_gap, cfg.contact_threshold + cfg.contact_hysteresis)
        min_gap = torch.full_like(hand_wall_gap, -cfg.max_penetration)
        release_min_gap = torch.full_like(hand_wall_gap, -(cfg.max_penetration + cfg.penetration_hysteresis))
        in_contact = torch.where(
            self._ee_was_in_contact,
            (hand_wall_gap <= release_limit) & (hand_wall_gap >= release_min_gap),
            (hand_wall_gap <= contact_limit) & (hand_wall_gap >= min_gap),
        )
        # Note: real_hand_wall_contact gate removed — gap-based check with hysteresis
        # is sufficient now that wall_face_x reads runtime body position correctly.
        # The contactfilter kernel only affects real MuJoCo contacts anyway.
        newly_in_contact = in_contact & ~self._ee_was_in_contact
        lost_contact = ~in_contact & self._ee_was_in_contact

        # 4. Engagement logic.
        self._handle_new_contacts(newly_in_contact)
        self._handle_lost_contacts(lost_contact)

        # 4b. Loading angle check (Eq. 8): disengage spines being pulled off.
        away_sign = torch.sign(ee_pos[:, :, 0] - wall_face_x.unsqueeze(1))
        away_sign = torch.where(away_sign == 0, -torch.ones_like(away_sign), away_sign)
        v_pullout = (ee_vel[:, :, 0] * away_sign).clamp(min=0.0)
        v_shear = torch.norm(ee_vel[:, :, 1:], dim=-1)
        speed = torch.sqrt(v_pullout**2 + v_shear**2)
        angle_fail = torch.zeros_like(in_contact, dtype=torch.bool)
        moving_fast = speed > 0.5
        engaged_ratio = self._spine_engaged.sum(dim=-1).float() / self._n_spines
        highly_engaged = engaged_ratio > 0.9
        angle_eligible = in_contact & ~newly_in_contact & moving_fast & highly_engaged
        angle_fail_spines = (angle_fail & angle_eligible).unsqueeze(-1).expand(-1, -1, S)
        n_before = self._spine_engaged.sum(dim=-1)
        self._spine_engaged = self._spine_engaged & ~angle_fail_spines

        if self._debug:
            n_after = self._spine_engaged.sum(dim=-1)
            disengaged = n_before[0] - n_after[0]
            ee_names = self._cfg.ee_body_names
            for i in range(self._n_ee):
                if "hand" in ee_names[i] and disengaged[i] > 0:
                    print(
                        f"[MS ANGLE] {ee_names[i]}: lost {int(disengaged[i])} spines, "
                        f"v_pull={v_pullout[0,i]:.3f} v_shear={v_shear[0,i]:.3f} speed={speed[0,i]:.3f}"
                    )

        # 5. Compute shear force capacity from engaged spines.
        shear_capacity = compute_total_shear_force(
            self._spine_travel, self._spine_engaged, cfg.K, cfg.d_stroke, cfg.F_max
        )  # (N, E) in Newtons

        # 6. Sync engagement state to contactfilter (Warp kernel).
        # Per-EE engagement: True if any spine is engaged for that EE.
        ee_engaged = self._spine_engaged.any(dim=-1) & in_contact  # (N, E) bool
        self._contact_filter.update_engagement(ee_engaged)

        # 7. Set adhesion actuator ctrl.
        # Normal holding capacity = shear_capacity * tan(θ_th).
        contact_mask = in_contact.float()
        normal_limit = shear_capacity * self._tan_theta_th * contact_mask
        adhesion_ctrl = (normal_limit / cfg.max_adhesion_gain).clamp(0.0, 1.0)
        self._sim_data.ctrl[:, self._adhesion_ctrl_ids] = adhesion_ctrl

        # 8. Update state and privileged obs.
        n_engaged = self._spine_engaged.sum(dim=-1)
        self._ee_was_in_contact = in_contact
        self._update_privileged_obs(n_engaged, normal_limit, in_contact, shear_capacity)

        if self._debug:
            if self._debug_interval_steps == 0:
                self._debug_interval_steps = max(1, round(1.0 / env.step_dt))
            self._debug_step_count += 1
            if self._debug_step_count >= self._debug_interval_steps:
                self._debug_step_count = 0
                ee_names = self._cfg.ee_body_names
                # Use efc.force + contact.efc_address to inspect solver-side
                # contact loads, and print any additional runtime force arrays
                # exposed by the simulator wrapper.
                if not hasattr(self, '_efc_force_torch'):
                    import warp as wp

                    def _maybe_torch_view(obj):
                        if obj is None:
                            return None
                        if isinstance(obj, torch.Tensor):
                            return obj
                        try:
                            return wp.to_torch(obj)
                        except Exception:
                            return None

                    raw = self._sim_data.struct
                    self._efc_force_torch = wp.to_torch(raw.efc.force)              # (nworld, max_efc)
                    self._contact_efc_addr = wp.to_torch(raw.contact.efc_address)   # (naconmax, 6)
                    self._contact_geom_torch = wp.to_torch(raw.contact.geom)        # (naconmax, 2)
                    self._contact_worldid_torch = wp.to_torch(raw.contact.worldid)  # (naconmax,)
                    self._contact_frame_torch = wp.to_torch(raw.contact.frame)      # (naconmax, 3, 3)
                    self._contact_dim_torch = wp.to_torch(raw.contact.dim)          # (naconmax,)
                    self._contact_friction_torch = wp.to_torch(raw.contact.friction) # (naconmax, 5)
                    self._contact_dist_torch = _maybe_torch_view(getattr(raw.contact, "dist", None))
                    self._contact_pos_torch = _maybe_torch_view(getattr(raw.contact, "pos", None))
                    self._nacon_torch = wp.to_torch(self._sim_data.struct.nacon)    # (1,) active contacts
                    self._qfrc_constraint_torch = _maybe_torch_view(getattr(raw, "qfrc_constraint", None))
                    self._qfrc_actuator_torch = _maybe_torch_view(getattr(raw, "qfrc_actuator", None))
                    self._actuator_force_torch = _maybe_torch_view(getattr(raw, "actuator_force", None))
                    if self._qfrc_constraint_torch is None:
                        self._qfrc_constraint_torch = _maybe_torch_view(getattr(self._sim_data, "qfrc_constraint", None))
                    if self._qfrc_actuator_torch is None:
                        self._qfrc_actuator_torch = _maybe_torch_view(getattr(self._sim_data, "qfrc_actuator", None))
                    if self._actuator_force_torch is None:
                        self._actuator_force_torch = _maybe_torch_view(getattr(self._sim_data, "actuator_force", None))
                    print(f"[MS DEBUG] efc.force shape={self._efc_force_torch.shape}")
                    print(f"[MS DEBUG] contact.efc_address shape={self._contact_efc_addr.shape}")
                    print(f"[MS DEBUG] contact.geom shape={self._contact_geom_torch.shape}")
                    if self._contact_dist_torch is not None:
                        print(f"[MS DEBUG] contact.dist shape={tuple(self._contact_dist_torch.shape)}")
                    if self._contact_pos_torch is not None:
                        print(f"[MS DEBUG] contact.pos shape={tuple(self._contact_pos_torch.shape)}")
                    if self._qfrc_constraint_torch is not None:
                        print(f"[MS DEBUG] qfrc_constraint shape={tuple(self._qfrc_constraint_torch.shape)}")
                    if self._qfrc_actuator_torch is not None:
                        print(f"[MS DEBUG] qfrc_actuator shape={tuple(self._qfrc_actuator_torch.shape)}")
                    if self._actuator_force_torch is not None:
                        print(f"[MS DEBUG] actuator_force shape={tuple(self._actuator_force_torch.shape)}")

                # Build hand geom lookup once.
                if not hasattr(self, '_hand_geom_ids_list'):
                    mj_model = env.sim.mj_model
                    self._hand_geom_ids_list = {
                        name: mj_model.geom(f"robot/{name}_collision").id
                        for name in self._cfg.ee_body_names
                        if "hand" in name
                    }
                    self._wall_geom_id_cached = mj_model.geom("wall/wall_geom").id
                    print(f"[MS DEBUG] hand geom IDs: {self._hand_geom_ids_list}")
                    print(f"[MS DEBUG] wall geom ID: {self._wall_geom_id_cached}")
                    # Print names of geoms contacting the wall in world 0.
                    raw = self._sim_data.struct
                    import warp as wp
                    mj_model_dbg = env.sim.mj_model
                    cg = wp.to_torch(raw.contact.geom)
                    cw = wp.to_torch(raw.contact.worldid)
                    nc = int(wp.to_torch(raw.nacon)[0].item())
                    wall_gid_dbg = self._wall_geom_id_cached
                    wall_contact_names = []
                    for ci in range(min(nc, 400)):
                        if int(cw[ci].item()) != 0:
                            continue
                        g0, g1 = int(cg[ci, 0].item()), int(cg[ci, 1].item())
                        if g0 == wall_gid_dbg or g1 == wall_gid_dbg:
                            other = g1 if g0 == wall_gid_dbg else g0
                            wall_contact_names.append(mj_model_dbg.geom(other).name)
                    print(f"[MS DEBUG] wall contacts(world0)={wall_contact_names}")

                naconmax = self._contact_geom_torch.shape[0]
                efc_force = self._efc_force_torch   # (nworld, max_efc)
                c_geom = self._contact_geom_torch   # (naconmax, 2)
                c_wid = self._contact_worldid_torch # (naconmax,)
                c_efc = self._contact_efc_addr      # (naconmax, 6)
                c_dim = self._contact_dim_torch     # (naconmax,)
                c_friction = self._contact_friction_torch  # (naconmax, 5)
                c_dist = getattr(self, "_contact_dist_torch", None)
                c_pos = getattr(self, "_contact_pos_torch", None)
                wall_gid = self._wall_geom_id_cached
                nacon = int(self._nacon_torch[0].item())

                hand_lines = []
                hotspot_lines = []

                for ee_idx, ee_name in enumerate(ee_names):
                    if "hand" not in ee_name:
                        continue
                    state = "ON " if in_contact[0, ee_idx] else "OFF"
                    eng = int(n_engaged[0, ee_idx].item())
                    cap = shear_capacity[0, ee_idx].item()
                    adh = adhesion_ctrl[0, ee_idx].item()
                    gap = hand_wall_gap[0, ee_idx].item()

                    hand_gid = self._hand_geom_ids_list.get(ee_name, -1)
                    f_normal_total = 0.0
                    f_tangent1_total = 0.0
                    f_tangent2_total = 0.0
                    n_hand_contacts = 0
                    n_loaded_contacts = 0
                    max_contact_force = 0.0
                    min_contact_dist = float("inf")
                    max_penetration = 0.0
                    best_contact_summary = None
                    for ci in range(naconmax):  # scan all slots, not just first nacon
                        if int(c_wid[ci].item()) != 0:
                            continue
                        g0, g1 = int(c_geom[ci, 0].item()), int(c_geom[ci, 1].item())
                        if not ((g0 == hand_gid and g1 == wall_gid) or (g1 == hand_gid and g0 == wall_gid)):
                            continue
                        n_hand_contacts += 1
                        condim = int(c_dim[ci].item())
                        addrs = [int(c_efc[ci, k].item()) for k in range(condim)]
                        forces = [
                            efc_force[0, a].item() if 0 <= a < efc_force.shape[1] else float("nan")
                            for a in addrs
                        ]
                        fric = [float(c_friction[ci, k].item()) for k in range(5)]
                        if condim >= 1:
                            f_normal_total += forces[0]
                        if condim >= 2:
                            f_tangent1_total += forces[1]
                        if condim >= 3:
                            f_tangent2_total += forces[2]

                        dist = float("nan")
                        pos_summary = ""
                        if c_dist is not None:
                            dist = float(c_dist[ci].item())
                            min_contact_dist = min(min_contact_dist, dist)
                            max_penetration = max(max_penetration, max(0.0, -dist))
                        if c_pos is not None:
                            pos = c_pos[ci]
                            pos_summary = f" pos=({float(pos[0].item()):.3f},{float(pos[1].item()):.3f},{float(pos[2].item()):.3f})"

                        contact_force_mag = sum(abs(f) for f in forces if not math.isnan(f))
                        max_contact_force = max(max_contact_force, contact_force_mag)
                        if contact_force_mag > 1.0e-3:
                            n_loaded_contacts += 1
                            summary = (
                                f"ci={ci} load={contact_force_mag:.1f}N "
                                + (f"dist={dist:.4f}" if not math.isnan(dist) else "dist=?")
                                + pos_summary + " "
                                f"fn={forces[0]:.1f}" +
                                (f" ft1={forces[1]:.1f}" if condim >= 2 else "") +
                                (f" ft2={forces[2]:.1f}" if condim >= 3 else "") +
                                f" mu=({fric[0]:.1f},{fric[1]:.1f})"
                            )
                            if best_contact_summary is None or contact_force_mag > best_contact_summary[0]:
                                best_contact_summary = (contact_force_mag, summary)

                    qfrc_constraint_norm = float("nan")
                    qfrc_actuator_norm = float("nan")
                    actuator_force = float("nan")
                    if self._qfrc_constraint_torch is not None:
                        q = self._qfrc_constraint_torch[0]
                        qfrc_constraint_norm = float(torch.linalg.vector_norm(q).item())
                    if self._qfrc_actuator_torch is not None:
                        q = self._qfrc_actuator_torch[0]
                        qfrc_actuator_norm = float(torch.linalg.vector_norm(q).item())
                    if self._actuator_force_torch is not None:
                        actuator_force = float(self._actuator_force_torch[0, self._adhesion_ctrl_ids[ee_idx]].item())
                    dist_summary = ""
                    if min_contact_dist != float("inf"):
                        dist_summary = f" dmin={min_contact_dist:>7.4f}m pen={max_penetration:>7.4f}m"
                    hand_lines.append(
                        f"[MS] {ee_name:<10} {state} "
                        f"eng={eng:>3}/{self._n_spines} "
                        f"act={actuator_force:>4.0f}N gap={gap:>6.3f}m "
                        f"contacts={n_hand_contacts}/{n_loaded_contacts} "
                        f"F=({f_normal_total:>5.1f}, {f_tangent1_total:>5.1f}, {f_tangent2_total:>5.1f})N "
                        f"peak={max_contact_force:>5.1f}N cap={cap:>6.1f}N{dist_summary}"
                    )
                    if best_contact_summary is not None:
                        hotspot_lines.append(f"[MS HOT] {ee_name}: {best_contact_summary[1]}")

                print(f"[MS STEP] nacon={nacon} ||qfrc_c||={qfrc_constraint_norm:.1f} ||qfrc_a||={qfrc_actuator_norm:.1f}")
                for line in hand_lines:
                    print(line)
                for line in hotspot_lines:
                    print(line)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset spine state for specified environments."""
        self._spine_engaged[env_ids] = False
        self._spine_travel[env_ids] = 0.0
        self._ee_was_in_contact[env_ids] = False
        self._privileged_obs[env_ids] = 0.0

        # Zero adhesion ctrl for reset envs.
        self._sim_data.ctrl[env_ids[:, None], self._adhesion_ctrl_ids[None, :]] = 0.0

        # Sync zeroed engagement to contactfilter.
        self._contact_filter.update_engagement(
            torch.zeros(self._num_envs, self._n_ee, dtype=torch.int32, device=self._device)
        )

        if self._cfg.randomize_surface:
            self._randomize_surface(env_ids)

    def get_privileged_obs(self) -> torch.Tensor:
        """Return cached privileged observations.

        Shape: (num_envs, n_ee * 5) where per-EE values are:
          [engaged_ratio, shear_utilization, normalized_normal, is_in_contact, surface_quality]
        """
        return self._privileged_obs

    # -- Internal methods -----------------------------------------------------

    def _get_real_hand_wall_contacts(self) -> torch.Tensor:
        """Return a per-env, per-EE mask of active hand-wall contacts."""
        import warp as wp

        if not hasattr(self, "_contact_geom_runtime_torch"):
            raw = self._sim_data.struct
            self._contact_geom_runtime_torch = wp.to_torch(raw.contact.geom)        # (naconmax, 2)
            self._contact_worldid_runtime_torch = wp.to_torch(raw.contact.worldid)  # (naconmax,)
            self._contact_efc_addr_runtime_torch = wp.to_torch(raw.contact.efc_address)  # (naconmax, 6)
            self._efc_force_runtime_torch = wp.to_torch(raw.efc.force)  # (nworld, max_efc)
            self._nacon_runtime_torch = wp.to_torch(raw.nacon)                      # (1,)

        contact_mask = torch.zeros((self._num_envs, self._n_ee), dtype=torch.bool, device=self._device)

        nacon = int(self._nacon_runtime_torch[0].item())
        if nacon <= 0:
            return contact_mask

        c_geom = self._contact_geom_runtime_torch[:nacon]
        c_wid = self._contact_worldid_runtime_torch[:nacon].long()
        g0 = c_geom[:, 0].long()
        g1 = c_geom[:, 1].long()
        wall_gid = self._wall_geom_id
        geom_to_ee = self._geom_to_ee_torch

        ee_from_g0 = geom_to_ee[g0]
        ee_from_g1 = geom_to_ee[g1]
        match0 = (ee_from_g0 >= 0) & (g1 == wall_gid)
        match1 = (ee_from_g1 >= 0) & (g0 == wall_gid)

        if match0.any():
            contact_mask[c_wid[match0], ee_from_g0[match0]] = True
        if match1.any():
            contact_mask[c_wid[match1], ee_from_g1[match1]] = True

        return contact_mask

    def _get_hand_wall_normal_forces(self) -> torch.Tensor:
        """Return summed normal contact force per env and end-effector."""
        if not hasattr(self, "_contact_geom_runtime_torch"):
            _ = self._get_real_hand_wall_contacts()

        normal_force = torch.zeros((self._num_envs, self._n_ee), dtype=torch.float32, device=self._device)

        nacon = int(self._nacon_runtime_torch[0].item())
        if nacon <= 0:
            return normal_force

        c_geom = self._contact_geom_runtime_torch[:nacon]
        c_wid = self._contact_worldid_runtime_torch[:nacon].long()
        c_efc = self._contact_efc_addr_runtime_torch[:nacon]
        g0 = c_geom[:, 0].long()
        g1 = c_geom[:, 1].long()
        wall_gid = self._wall_geom_id
        geom_to_ee = self._geom_to_ee_torch
        efc_force = self._efc_force_runtime_torch

        ee_from_g0 = geom_to_ee[g0]
        ee_from_g1 = geom_to_ee[g1]
        match0 = (ee_from_g0 >= 0) & (g1 == wall_gid)
        match1 = (ee_from_g1 >= 0) & (g0 == wall_gid)

        for match_mask, ee_ids in ((match0, ee_from_g0), (match1, ee_from_g1)):
            match_idx = torch.nonzero(match_mask, as_tuple=False).squeeze(-1)
            if match_idx.numel() == 0:
                continue
            world_ids = c_wid[match_idx]
            ee_sel = ee_ids[match_idx]
            efc_addr = c_efc[match_idx, 0].long()
            valid = (efc_addr >= 0) & (efc_addr < efc_force.shape[1])
            if not valid.any():
                continue
            world_ids = world_ids[valid]
            ee_sel = ee_sel[valid]
            efc_addr = efc_addr[valid]
            normal_vals = efc_force[world_ids, efc_addr].abs().to(torch.float32)
            normal_force.index_put_((world_ids, ee_sel), normal_vals, accumulate=True)

        return normal_force

    def _handle_new_contacts(self, newly_in_contact: torch.Tensor) -> None:
        """Sample spine engagement for newly contacting EEs."""
        if not newly_in_contact.any():
            return

        cfg = self._cfg
        mask = newly_in_contact.unsqueeze(-1).expand(-1, -1, self._n_spines)

        # P(engage) = (1 - alpha) + alpha * (1 - exp(-lambda * d_stroke))
        alpha = self._surface_alpha.unsqueeze(-1).unsqueeze(-1)
        lam_m = self._surface_lambda.unsqueeze(-1).unsqueeze(-1) * 1000.0
        p_engage = (1.0 - alpha) + alpha * (1.0 - torch.exp(-lam_m * cfg.d_stroke))
        p_engage = p_engage.expand_as(mask)

        engaged = torch.bernoulli(p_engage).bool() & mask
        self._spine_engaged = torch.where(mask, engaged, self._spine_engaged)

        # Sample travel distance from Exponential(lambda) for engaged spines.
        lam_sample = self._surface_lambda.unsqueeze(-1).unsqueeze(-1) * 1000.0
        u = torch.rand_like(self._spine_travel).clamp(min=1e-7)
        travel = -torch.log(u) / lam_sample
        travel = travel.clamp(0.0, cfg.d_stroke)
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
        ], dim=-1)  # (N, E, 5)

        self._privileged_obs = obs.reshape(self._num_envs, -1)
