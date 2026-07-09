"""Rizon4 microspine bench teleop: operational-space (task-space) force control.

The loading layer commands a Cartesian force at the microspine TCP and realizes
it as joint torques via ``tau = Jᵀ·F_cmd + Nᵀ·τ₀`` (OSC), applied through
``qfrc_applied`` on the CPU-mirror-computed Jacobian. The arm's position control
is disabled so it does not fight the torque; gravity is already compensated by
the model (``gravcomp=1.0``). This maps the *full* limit-surface envelope (all
directions reach failure), unlike the old position-drag + leash which saturated
on the arm's compliance.
"""
from __future__ import annotations
from dataclasses import dataclass
import mujoco
import numpy as np
import torch
import scr_phybot_mjab.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from scr_phybot_mjab.assets.rizon4.rizon4_constants import (
    RIZON4_HOME_JOINT_POS,
    RIZON4_JOINT_NAMES,
)
from scr_phybot_mjab.microspine.config import MicrospineLimitSurfaceCfg
from scr_phybot_mjab.tasks.microspine_rizon4.config.rizon4.env_cfgs import (
    rizon4_microspine_env_cfg,
)


@dataclass
class Reading:
    load_tln: np.ndarray
    margin: float
    released: bool
    engaged: bool


class PadTeleop:
    def __init__(
        self,
        device: str = "cuda:0",
        Kp_null: float = 5.0,
        Kd_null: float = 1.0,
        Dtask: float = 2.0,
    ):
        # OSC gains (tunable): Kp_null/Kd_null resolve the arm's redundancy in the
        # task null-space (posture toward home, damped) so the task force sees a
        # well-conditioned arm with ZERO task-space preload; Dtask is Cartesian
        # damping folded into the commanded force for stability.
        self.device = device
        self.Kp_null = Kp_null
        self.Kd_null = Kd_null
        self.Dtask = Dtask
        cfg = rizon4_microspine_env_cfg(play=False, attachment_mode="constraint")
        # Free the null-space (no IK posture pull) and disable the arm's position
        # servo so it does not fight the injected torque. The builtin position
        # actuator divides effort_limit/stiffness at build, so stiffness cannot be
        # exactly 0; a tiny value makes its restoring force negligible (~1e-6 N·m
        # over the whole workspace) — effectively a pure zero actuator. Damping 0.
        cfg.actions["cartesian_pose"].posture_weight = 0.0
        for a in cfg.scene.entities["robot"].articulation.actuators:
            a.stiffness = 1e-6; a.damping = 0.0
        # Restore the true envelope (f_n0/f_l0 back to the real ~15, not the
        # bench's de-fanged 200) AND collapse every limit-surface range to its
        # midpoint. Collapsing kills the per-env domain randomization so the
        # env's actual release envelope is exactly `self.limit_params` below —
        # the bench maps one deterministic true envelope, and every release
        # lands on it (not on a per-reset random draw).
        d = MicrospineLimitSurfaceCfg()
        m = cfg.commands["microspine_affordance"].microspine_cfg
        mid = lambda r: 0.5 * (r[0] + r[1])
        _range_attrs = (
            "f_t_max_range", "f_rev_max_range", "mu_adhesion_range",
            "f_n0_range", "mu_lateral_range", "f_l0_range",
        )
        for owner in (m, cfg.events["microspine_adhesion"].params["microspine_cfg"]):
            owner.limit_surface.f_n0_range = d.f_n0_range
            owner.limit_surface.f_l0_range = d.f_l0_range
            for attr in _range_attrs:
                v = mid(getattr(owner.limit_surface, attr))
                setattr(owner.limit_surface, attr, (v, v))
        cfg.scene.num_envs = 1
        cfg.episode_length_s = 1.0e9
        self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
        self.affordance = self.env.command_manager._terms["microspine_affordance"]
        self.executor = self.env.event_manager.get_term_cfg("microspine_adhesion").func
        self.robot = self.env.scene["robot"]
        self.wall = self.env.scene["wall"]
        self._pid = self.robot.find_bodies(["microspine_pad"])[0]
        self._half = float(self.env.sim.mj_model.geom("wall/wall_geom").size[0])
        self.mj_model = self.env.sim.mj_model
        self.mj_data = mujoco.MjData(self.mj_model)  # CPU mirror (viewer + OSC)

        # Task site + arm DOF/qpos addressing (names may or may not be prefixed).
        def _site_id():
            try:
                return self.mj_model.site("microspine_tcp").id
            except KeyError:
                return self.mj_model.site("robot/microspine_tcp").id
        self._site_id = _site_id()

        def _joint(name):
            try:
                return self.mj_model.joint(name)
            except KeyError:
                return self.mj_model.joint("robot/" + name)
        self._arm_dof = [int(_joint(n).dofadr[0]) for n in RIZON4_JOINT_NAMES]
        self._arm_qadr = [int(_joint(n).qposadr[0]) for n in RIZON4_JOINT_NAMES]
        self._q_rest = np.array(
            [RIZON4_HOME_JOINT_POS[n] for n in RIZON4_JOINT_NAMES], dtype=np.float64
        )
        self._zero_act = torch.zeros(1, 7, device=self.device)

        mid = lambda r: 0.5 * (r[0] + r[1])
        ls = m.limit_surface
        self.limit_params = {
            "f_t_max": mid(ls.f_t_max_range), "f_rev_max": mid(ls.f_rev_max_range),
            "mu_adhesion": mid(ls.mu_adhesion_range), "f_n0": mid(ls.f_n0_range),
            "mu_lateral": mid(ls.mu_lateral_range), "f_l0": mid(ls.f_l0_range),
        }
        self.home()

    # --- kinematics -------------------------------------------------------
    def _pad_pose(self):
        pid = self._pid[0]
        return (self.robot.data.body_link_pos_w[:, pid].clone(),
                self.robot.data.body_link_quat_w[:, pid].clone())

    def sync_display(self):
        self.mj_data.qpos[:] = self.env.sim.data.qpos[0].detach().cpu().numpy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    # --- OSC core ---------------------------------------------------------
    def _control_tick(self, F_cmd: np.ndarray) -> None:
        """One task-space force control step. F_cmd is a world-frame 3-vector."""
        # 1. Sync the CPU mirror from the GPU sim.
        self.mj_data.qpos[:] = self.env.sim.data.qpos[0].detach().cpu().numpy()
        self.mj_data.qvel[:] = self.env.sim.data.qvel[0].detach().cpu().numpy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # 2-3. Site Jacobian restricted to the 7 arm DOFs.
        nv = self.mj_model.nv
        jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, jacr, self._site_id)
        Ja = jacp[:, self._arm_dof]  # (3, 7), force only

        q_arm = self.mj_data.qpos[self._arm_qadr]
        qd_arm = self.mj_data.qvel[self._arm_dof]

        # 6. Task-space damping folded into the commanded force.
        F = np.asarray(F_cmd, dtype=np.float64) - self.Dtask * (Ja @ qd_arm)

        # 4. Task torque.
        tau_task = Ja.T @ F  # (7,)

        # 5. Kinematic null-space posture (zero task-space preload).
        Jpinv = np.linalg.pinv(Ja)
        N = np.eye(7) - Jpinv @ Ja
        tau0 = -self.Kp_null * (q_arm - self._q_rest) - self.Kd_null * qd_arm
        tau_null = N @ tau0

        # 7. Total torque (gravity already compensated by the model).
        tau = tau_task + tau_null

        # 8. Apply to the sim arm DOFs and step (zeros action; position control off).
        tau_t = torch.as_tensor(tau, dtype=torch.float32, device=self.device)
        self.env.sim.data.qfrc_applied[0, self._arm_dof] = tau_t
        self.env.step(self._zero_act)

    def _engage_frame_world(self):
        """(t̂, l̂, n̂_out) for env0/EE0 as numpy 3-vectors (world)."""
        t_hat, l_hat, n_hat = self.affordance._engage_frame()
        return (t_hat[:, 0][0].detach().cpu().numpy(),
                l_hat[:, 0][0].detach().cpu().numpy(),
                n_hat[:, 0][0].detach().cpu().numpy())

    # --- actions ----------------------------------------------------------
    def home(self):
        # Reset to the arm's home joint pose; the null-space posture holds it there
        # (position control is off, gravity compensated). press_engage() then drives
        # the pad into the wall with a Cartesian force.
        self.env.reset(); self.executor.reset()
        for _ in range(5):
            self._control_tick(np.zeros(3))

    def press_engage(self):
        # Steady INTO-wall force (-n̂) until the state machine latches, then relax
        # and re-seat the anchor at the settled pad to null the press preload.
        _, _, n_hat = self._engage_frame_world()
        F_in = 20.0 * (-n_hat)
        for _ in range(40):
            self._control_tick(F_in)
            if bool(self.affordance.engaged[0, 0].item()):
                break
        for _ in range(10):
            self._control_tick(np.zeros(3))
        self.executor.anchor_pos_w = self.executor.pad_center_world().clone()
        for _ in range(5):
            self._control_tick(np.zeros(3))

    def apply(self, force_tln) -> Reading:
        t_hat, l_hat, n_hat = self._engage_frame_world()
        ft, fl, fn = force_tln
        F_cmd = ft * t_hat + fl * l_hat + fn * n_hat  # world 3-vector
        self._control_tick(F_cmd)
        load = self.affordance.anchor_load_tnl[0, 0].detach().cpu().numpy()
        return Reading(
            load_tln=load,
            margin=float(self.affordance.hold_margin[0].item()),
            released=bool(self.affordance.forced_release_this_step[0, 0].item()),
            engaged=bool(self.affordance.engaged[0, 0].item()),
        )
