"""ToddlerBot reference IK and independent point-contact static feasibility.

The static solve balances every floating-base/passive DOF, including gear
reaction forces, and bounds each actual motor separately. It does not add
unmeasured contact couples. A successful solve is a quasi-static necessary
check, not a dynamic rollout or a stochastic reliability certificate.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares, linprog, minimize
from scipy.spatial.transform import Rotation

from scr_phybot_mjab.assets.toddlerbot.toddlerbot_constants import (
    MOTOR_NAMES,
    MOTORS,
    MOUNT_SIGNATURE,
    ROOT,
    TORQUE_LIMITS,
    UNIT_NAMES,
    UNIT_PRELOAD,
    get_spec,
)
from scr_phybot_mjab.microspine.empirical_limit_surface import EmpiricalLimitSurface

REFERENCE_PATH = ROOT.parent / "motions/toddlerbot_climb.npz"
WALL_X = 0.4
PIVOT = np.array([WALL_X, 0.0, 0.0])
UNIT_ROT = Rotation.from_euler("z", 90, degrees=True).as_matrix()


def wall_rotation(angle_deg):
    return Rotation.from_euler("y", 90 - angle_deg, degrees=True)


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


class ToddlerReferenceModel:
    def __init__(self):
        # In-plane rake angles can follow spread limbs while every face points
        # into the wall. Actual local rake frames are also used in force checks.
        self.unit_rotations = np.repeat(UNIT_ROT[None], 4, axis=0)
        self.model = get_spec().compile()
        self.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        self.data = mujoco.MjData(self.model)
        self.motor_jids = np.array([self.model.joint(n).id for n in MOTOR_NAMES])
        self.motor_qadr = self.model.jnt_qposadr[self.motor_jids]
        self.motor_dofs = self.model.jnt_dofadr[self.motor_jids]
        self.site_ids = np.array([self.model.site(n + "_contact_center").id for n in UNIT_NAMES])
        self.unit_bids = np.array([self.model.body(n).id for n in UNIT_NAMES])
        self.table = EmpiricalLimitSurface()
        # Neck stays at its neutral pose; all motor limits remain in the audit.
        self.variable_motors = [i for i, n in enumerate(MOTOR_NAMES) if not n.startswith("neck")]
        n = 3 + len(self.variable_motors)
        self.D = np.zeros((self.model.nv, n))
        self.D[:3, :3] = np.eye(3)
        self.Q = np.zeros((self.model.nq, n))
        self.Q[:3, :3] = np.eye(3)
        for k, i in enumerate(self.variable_motors, 3):
            self.D[self.motor_dofs[i], k] = 1
            self.Q[self.motor_qadr[i], k] = 1
        for eq in range(self.model.neq):
            if self.model.eq_type[eq] == mujoco.mjtEq.mjEQ_JOINT:
                j1, j2 = self.model.eq_obj1id[eq], self.model.eq_obj2id[eq]
                ratio = self.model.eq_data[eq, 1]
                self.D[self.model.jnt_dofadr[j1]] = ratio * self.D[self.model.jnt_dofadr[j2]]
                self.Q[self.model.jnt_qposadr[j1]] = ratio * self.Q[self.model.jnt_qposadr[j2]]
        w1, w2 = [MOTOR_NAMES.index(f"waist_act_{i}") for i in (1, 2)]
        for joint, a, b in (("waist_roll", -0.29166667, 0.29166667), ("waist_yaw", 0.20833333, 0.20833333)):
            j = self.model.joint(joint).id
            self.D[self.model.jnt_dofadr[j]] = a * self.D[self.motor_dofs[w1]] + b * self.D[self.motor_dofs[w2]]
            self.Q[self.model.jnt_qposadr[j]] = a * self.Q[self.motor_qadr[w1]] + b * self.Q[self.motor_qadr[w2]]
        self.lower = np.r_[[0.15, -0.10, 0.25], self.model.jnt_range[self.motor_jids[self.variable_motors], 0] + 0.01]
        self.upper = np.r_[[0.395, 0.10, 0.65], self.model.jnt_range[self.motor_jids[self.variable_motors], 1] - 0.01]
        self.home = np.r_[[0.31, 0.0, 0.42], [MOTORS[MOTOR_NAMES[i]]["home_pos"] for i in self.variable_motors]]
        self.home = np.clip(self.home, self.lower + 1e-6, self.upper - 1e-6)
        self.jp = np.zeros((3, self.model.nv))
        self.jr = self.jp.copy()
        m = self.model
        self.collision_geoms = [i for i in range(m.ngeom) if m.geom_contype[i]]
        self.collision_pairs = []
        for k, a in enumerate(self.collision_geoms):
            ba = m.body_weldid[m.geom_bodyid[a]]
            for b in self.collision_geoms[k + 1 :]:
                bb = m.body_weldid[m.geom_bodyid[b]]
                if ba == bb or m.body_weldid[m.body_parentid[ba]] == bb or m.body_weldid[m.body_parentid[bb]] == ba:
                    continue
                self.collision_pairs.append((a, b))

    def set_x(self, x):
        self.data.qpos[:] = self.Q @ x
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def solve_ik(self, targets, initial=None, posture=None, max_nfev=180, avoid_collisions=True, continuity_weight=0.0):
        initial = self.home.copy() if initial is None else initial.copy()
        posture = initial if posture is None else posture
        m, d = self.model, self.data

        def evaluate(x, jac=False):
            self.set_x(x)
            residual, derivative = [], []
            for sid, p, target_rot in zip(self.site_ids, targets, self.unit_rotations):
                rot = d.site_xmat[sid].reshape(3, 3)
                residual.extend(100 * (d.site_xpos[sid] - p))
                if jac:
                    mujoco.mj_jacSite(m, d, self.jp, self.jr, int(sid))
                    derivative.append(100 * self.jp @ self.D)
                # Align the contact face, but let rake rotate in the wall plane
                # as the rigid forearm/shin spreads. A weak rake preference
                # chooses a branch; it is not a kinematic orientation lock.
                for axis, axis_weight in ((1, 3.0), (2, 0.08)):
                    v = rot[:, axis]
                    residual.extend(axis_weight * (v - target_rot[:, axis]))
                    if jac:
                        derivative.append(-axis_weight * skew(v) @ self.jr @ self.D)
            if avoid_collisions:
                for gid in self.collision_geoms:
                    if m.geom(gid).name.endswith("_c_0_0"):
                        continue
                    R = d.geom_xmat[gid].reshape(3, 3)
                    p = d.geom_xpos[gid] + R @ (np.sign(R[0]) * m.geom_size[gid])
                    error = max(0.0, p[0] - WALL_X + 0.001)
                    residual.append(100 * error)
                    if jac:
                        mujoco.mj_jac(m, d, self.jp, self.jr, p, int(m.geom_bodyid[gid]))
                        derivative.append(100 * self.jp[0:1] @ self.D if error else np.zeros((1, len(x))))
                for g1, g2 in self.collision_pairs:
                    segment = np.zeros(6)
                    dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.01, segment)
                    error = max(0.0, 0.0005 - dist)
                    residual.append(100 * error)
                    if jac:
                        row = np.zeros((1, len(x)))
                        if error and abs(dist) > 1e-10:
                            normal = (segment[3:] - segment[:3]) / dist
                            mujoco.mj_jac(m, d, self.jp, self.jr, segment[:3], int(m.geom_bodyid[g1]))
                            row = normal @ self.jp @ self.D
                            mujoco.mj_jac(m, d, self.jp, self.jr, segment[3:], int(m.geom_bodyid[g2]))
                            row = (100 * (row - normal @ self.jp @ self.D))[None]
                        derivative.append(row)
            weight = np.r_[[0.15, 0.2, 0.1], np.full(len(x) - 3, 0.004)]
            residual.extend(weight * (x - posture))
            if jac:
                derivative.append(np.diag(weight))
            if continuity_weight:
                residual.extend(continuity_weight * (x - initial))
                if jac:
                    derivative.append(continuity_weight * np.eye(len(x)))
            return np.vstack(derivative) if jac else np.asarray(residual)

        result = least_squares(
            evaluate,
            np.clip(initial, self.lower + 1e-7, self.upper - 1e-7),
            jac=lambda x: evaluate(x, True),
            bounds=(self.lower, self.upper),
            max_nfev=max_nfev,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-9,
        )
        self.set_x(result.x)
        return result.x, self.ik_errors(targets, iterations=result.nfev)

    def ik_errors(self, targets, iterations=0):
        d = self.data
        pos = float(np.linalg.norm(d.site_xpos[self.site_ids] - targets, axis=1).max())
        preference = max(
            Rotation.from_matrix(target.T @ d.site_xmat[s].reshape(3, 3)).magnitude()
            for s, target in zip(self.site_ids, self.unit_rotations)
        )
        face = max(
            np.arccos(np.clip(target[:, 1] @ d.site_xmat[s].reshape(3, 3)[:, 1], -1.0, 1.0))
            for s, target in zip(self.site_ids, self.unit_rotations)
        )
        return {
            "position_error_m": pos,
            "orientation_error_rad": face,
            "orientation_preference_error_rad": preference,
            "iterations": iterations,
        }

    def state(self, x, angle_deg=90):
        self.set_x(x)
        q = self.data.qpos.copy()
        r = wall_rotation(angle_deg)
        q[:3] = r.apply(q[:3] - PIVOT) + PIVOT
        q[3:7] = r.as_quat(scalar_first=True)
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        return q

    def geometry_audit(self, angle_deg):
        m, d = self.model, self.data
        inward = wall_rotation(angle_deg).apply([1.0, 0.0, 0.0])
        penetration = 0.0
        offenders = []
        self_depth = 0.0
        for i in range(m.ngeom):
            if not m.geom_contype[i]:
                continue
            R = d.geom_xmat[i].reshape(3, 3)
            kind = m.geom_type[i]
            extent = (np.abs(R.T @ inward) @ m.geom_size[i]) if kind == mujoco.mjtGeom.mjGEOM_BOX else m.geom_size[i, 0]
            depth = float((d.geom_xpos[i] - PIVOT) @ inward + extent)
            if depth > penetration:
                penetration = depth
            if depth > 0.0005:
                offenders.append(m.geom(i).name)
        for c in d.contact:
            if c.dist < self_depth:
                self_depth = float(c.dist)
        limits = []
        for j in range(m.njnt):
            if m.jnt_limited[j]:
                v = d.qpos[m.jnt_qposadr[j]]
                if v < m.jnt_range[j, 0] - 1e-6 or v > m.jnt_range[j, 1] + 1e-6:
                    limits.append(m.joint(j).name)
        return {
            "wall_penetration_m": max(0.0, penetration),
            "wall_collision_geoms": offenders,
            "self_penetration_m": -self_depth,
            "joint_limit_violations": limits,
            "internal_equality_error": float(np.max(np.abs(d.efc_pos[: d.ne]), initial=0.0)),
        }

    def statics(
        self, engaged, angle_deg=90.0, press=-1, quantile=0.5, torque_fraction=1.0, initial=None, load_scale=None
    ):
        m, d = self.model, self.data
        eq = d.efc_J.reshape(-1, m.nv)[: d.ne]
        N = null_space(eq, rcond=1e-7)
        normal = wall_rotation(angle_deg).apply([-1.0, 0.0, 0.0])
        blocks = []
        for sid in self.site_ids:
            mujoco.mj_jacSite(m, d, self.jp, self.jr, int(sid))
            # Match the runtime unit-local rake frame, including small IK
            # orientation errors. Force variables are loads on the wall.
            rake = -d.site_xmat[sid].reshape(3, 3)[:, 2]
            rake -= np.dot(rake, normal) * normal
            rake /= np.linalg.norm(rake)
            lateral = np.cross(normal, rake)
            basis = np.column_stack((-rake, -lateral, -normal))
            blocks.append(N.T @ self.jp.T @ basis)
        B = np.zeros((m.nv, m.nu))
        B[self.motor_dofs, np.arange(m.nu)] = 1
        A = np.column_stack((*blocks, N.T @ B))
        b = N.T @ d.qfrc_bias
        limits = TORQUE_LIMITS * torque_fraction
        bounds = []
        load_scale = np.ones(4) if load_scale is None else np.asarray(load_scale)
        for i, on in enumerate(engaged):
            bounds.extend(
                [(a * load_scale[i], b * load_scale[i]) for a, b in ((-15, 30), (-30, 30), (-60, 10))]
                if on
                else [(0, 0), (0, 0), (-UNIT_PRELOAD, -UNIT_PRELOAD)]
                if i == press
                else [(0, 0)] * 3
            )
        bounds.extend([(-v, v) for v in limits])
        support = np.flatnonzero(engaged)

        def margins(z):
            f = z[:12].reshape(4, 3)
            cap, _, valid, _ = self.table.query_numpy(f[support, 0], f[support, 1], quantile)
            return np.where(valid, cap - f[support, 2], -100.0)

        def obj(z):
            return float(np.sum((z[12:] / limits) ** 2) + 0.001 * np.sum(z[:12] ** 2))

        lp = linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=b, bounds=bounds, method="highs")
        if not lp.success:
            return {"pass": False, "reason": "force_balance_or_motor_limits", "lp_status": lp.message}
        start = lp.x
        seed_source = "linear_program"
        projection_residual = None
        if initial is not None and np.shape(initial) == start.shape and np.isfinite(initial).all():
            # A neighboring frame's forces do not exactly balance this pose.
            # Project them onto this frame's linear equilibrium and bounds;
            # discarding them used to force every retry back to the same LP seed.
            prior = np.asarray(initial)
            lo, hi = np.asarray(bounds).T
            scale = np.r_[np.full(12, 10.0), limits]
            projection = minimize(
                lambda z: 0.5 * np.sum(((z - prior) / scale) ** 2),
                np.clip(prior, lo, hi),
                jac=lambda z: (z - prior) / scale**2,
                method="SLSQP",
                bounds=bounds,
                constraints=({"type": "eq", "fun": lambda z: A @ z - b, "jac": lambda z: A},),
                options={"maxiter": 200, "ftol": 1e-12},
            )
            projection_residual = float(np.max(np.abs(A @ projection.x - b)))
            if (
                projection_residual < 1e-7
                and np.isfinite(projection.x).all()
                and np.all(projection.x >= lo - 1e-8)
                and np.all(projection.x <= hi + 1e-8)
            ):
                start = projection.x
                seed_source = "projected_initial"
        result = minimize(
            obj,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=(
                {"type": "eq", "fun": lambda z: A @ z - b, "jac": lambda z: A},
                {"type": "ineq", "fun": margins},
            ),
            options={"maxiter": 160, "ftol": 1e-8},
        )
        z = result.x
        residual = float(np.max(np.abs(A @ z - b)))
        margin = float(np.min(margins(z), initial=100.0))
        ratios = np.abs(z[12:]) / TORQUE_LIMITS
        return {
            "pass": residual < 1e-4 and margin >= -1e-4 and float(ratios.max()) <= torque_fraction + 1e-5,
            "reason": str(result.message),
            "balance_residual": residual,
            "normal_margin_n": margin,
            "peak_motor_fraction": float(ratios.max()),
            "limiting_motor": MOTOR_NAMES[int(ratios.argmax())],
            "forces_tln": z[:12].reshape(4, 3).tolist(),
            "motor_torque": z[12:].tolist(),
            "quantile": quantile,
            "press_unit": int(press),
            "seed_source": seed_source,
            "seed_projection_residual": projection_residual,
        }


def load_reference(path=REFERENCE_PATH):
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k].copy() for k in data.files}
    if tuple(result["motor_names"]) != MOTOR_NAMES or tuple(result["unit_names"]) != UNIT_NAMES:
        raise ValueError("Reference motor/unit ordering does not match ToddlerBot asset")
    if float(result["preload_n"]) != UNIT_PRELOAD:
        raise ValueError("Reference preload differs from task")
    if str(result.get("mount_signature", "")) != MOUNT_SIGNATURE:
        raise ValueError("Reference unit mounts differ from the asset; regenerate the reference")
    return result
