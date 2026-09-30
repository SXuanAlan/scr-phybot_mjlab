"""Generate and audit a four-unit climbing cycle; failed audits stay visible."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scr_phybot_mjab.assets.toddlerbot.toddlerbot_constants import (
    MOTOR_NAMES,
    MOUNT_SIGNATURE,
    SPEED_LIMITS,
    UNIT_MOUNTS,
    UNIT_NAMES,
    UNIT_PRELOAD,
    speed_torque_limits,
)
from scr_phybot_mjab.tools.climbing_reference.toddlerbot import (
    REFERENCE_PATH,
    UNIT_ROT,
    WALL_X,
    ToddlerReferenceModel,
    load_reference,
    wall_rotation,
)


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * t * (10 + t * (-15 + 6 * t))


def generate(args):
    rig = ToddlerReferenceModel()
    targets = np.array([[WALL_X, 0.12, 0.23], [WALL_X, -0.12, 0.23], [WALL_X, 0.18, 0.61], [WALL_X, -0.18, 0.61]])
    rig.unit_rotations = (
        Rotation.from_euler("x", np.array([35, -35, -35, 35])[:, None], degrees=True).as_matrix() @ UNIT_ROT
    )
    posture = rig.home.copy()
    posture[:3] = [0.37, 0.0, 0.42]
    for k, i in enumerate(rig.variable_motors, 3):
        name = MOTOR_NAMES[i]
        if any(part in name for part in ("shoulder_yaw", "elbow", "wrist", "ankle")):
            posture[k] = 0.0
        if name.endswith("shoulder_pitch"):
            posture[k] = -np.pi if name.startswith("left") else np.pi
        if name.endswith("shoulder_roll"):
            posture[k] = np.deg2rad(30)
    reused = load_reference(args.kinematics) if args.kinematics else None
    if reused is not None:
        if (
            float(reused["fps"]) != args.fps
            or len(reused["x"]) != 14 * args.fps + 1
            or not np.isclose(np.linalg.norm(reused["cycle_delta"]), args.step)
            or not np.allclose(reused["contact_targets"], targets)
            or not np.allclose(reused["unit_orientation_preferences"], rig.unit_rotations)
        ):
            raise ValueError(
                "Reused kinematics must have the same FPS, duration, step, contact targets and orientation preferences"
            )
        initial = reused["x"][0].copy()
        rig.set_x(initial)
        err = rig.ik_errors(targets)
    else:
        initial, err = rig.solve_ik(targets, initial=posture, posture=posture, max_nfev=500)
        # Refining with the same targets removes dependence on the standing seed.
        initial, err = rig.solve_ik(targets, initial=initial, posture=posture, max_nfev=300)
    print("Initial IK", err, "root", initial[:3], flush=True)
    cycle_duration = 14.0
    fps = args.fps
    times = np.arange(round(cycle_duration * fps) + 1) / fps
    qpos = []
    motor = []
    poses = []
    engage = []
    presses = []
    audits = []
    xs = []
    torques = []
    forces = []
    x = initial.copy()
    # Hands then feet, with exactly one released contact. The final 2 s bring
    # the posture back to the translated start, closing the cycle.
    order = (2, 0, 3, 1)
    for frame, t in enumerate(times):
        k = min(int(t / 3), 4)
        p = targets.copy()
        on = np.ones(4, dtype=bool)
        load_scale = np.ones(4)
        press = -1
        for done in order[:k]:
            p[done, 2] += args.step
        preferred = initial.copy()
        preferred[2] += args.step * smooth(t / 12)
        if k < 4:
            limb = order[k]
            u = (t - k * 3) / 3
            # Transfer load before release and restore it after seating.
            # This also makes the feedforward continuous at a support switch.
            if u < 0.15:
                load_scale[limb] = 1 - smooth(u / 0.15)
            elif u >= 0.75:
                load_scale[limb] = smooth((u - 0.75) / 0.15)
            local = smooth((u - 0.15) / 0.5)
            p[limb, 2] += args.step * local
            if 0.15 <= u < 0.75:
                on[limb] = False
                if u < 0.65:
                    p[limb, 0] -= 0.006 * np.sin(np.pi * np.clip((u - 0.15) / 0.5, 0, 1)) ** 2
                else:
                    press = limb
                    p[limb, 0] += 0.0002 * np.sin(np.pi * (u - 0.65) / 0.1)
            preferred[1] += (0.025 if limb in (1, 3) else -0.025) * np.sin(np.pi * u) ** 2
        if k == 4:
            closed = initial.copy()
            closed[2] += args.step
            if t == 12.0:
                settle_start = x.copy()
            preferred = settle_start + (closed - settle_start) * smooth((t - 12) / 2)
            x = preferred.copy()
        if reused is not None:
            x = reused["x"][frame].copy()
            rig.set_x(x)
            ik = rig.ik_errors(p)
        elif frame == len(times) - 1:
            # The translated first pose already satisfies these contacts.
            # Set the periodic boundary exactly, including unconstrained wrists.
            x = closed.copy()
            rig.set_x(x)
            ik = rig.ik_errors(p)
        else:
            continuity = 0.06 + (smooth((t - 12) / 2) if k == 4 else 0.0)
            x, ik = rig.solve_ik(p, initial=x, posture=preferred, max_nfev=100, continuity_weight=continuity)
        if frame == 0:
            initial = x.copy()
        errors = np.linalg.norm(rig.data.site_xpos[rig.site_ids] - p, axis=1)
        ik["holding_position_error_m"] = float(errors[on].max(initial=0.0))
        state = rig.state(x, args.angle)
        geometry = rig.geometry_audit(args.angle)
        result = rig.statics(
            on,
            args.angle,
            press=press,
            quantile=args.quantile,
            torque_fraction=args.torque_fraction,
            load_scale=load_scale,
        )
        qpos.append(state)
        xs.append(x.copy())
        motor.append(state[rig.motor_qadr])
        poses.append(rig.data.site_xpos[rig.site_ids].copy())
        engage.append(on)
        presses.append(press)
        torques.append(result.get("motor_torque", np.zeros(len(MOTOR_NAMES))))
        forces.append(result.get("forces_tln", np.zeros((4, 3))))
        audits.append(
            {
                "frame": frame,
                "time_s": float(t),
                "ik": {k: float(v) for k, v in ik.items()},
                "geometry": geometry,
                "load_scale": load_scale.tolist(),
                "statics": result,
            }
        )
        if frame % 50 == 0:
            print(f"frame {frame}/{len(times)} static={result['pass']} root={x[:3].round(3)}", flush=True)
    initial_failed_frames = [i for i, audit in enumerate(audits) if not audit["statics"]["pass"]]
    # The empirical limit surface has multiple force-allocation branches.
    # A cold SLSQP failure is not an infeasibility certificate. Retry from nearby
    # feasible allocations with the same contact mask, preserving every bound.
    for frame in initial_failed_frames:
        candidates = [
            j
            for j, audit in enumerate(audits)
            if audit["statics"]["pass"] and np.array_equal(engage[j], engage[frame]) and presses[j] == presses[frame]
        ]
        candidates.sort(key=lambda j: (abs(j - frame), j))
        audit = audits[frame]
        audit["initial_statics"] = audit["statics"]
        audit["static_retries"] = []
        rig.state(xs[frame], args.angle)
        for seed_frame in candidates[:4]:
            retry = rig.statics(
                engage[frame],
                args.angle,
                press=presses[frame],
                quantile=args.quantile,
                torque_fraction=args.torque_fraction,
                load_scale=audit["load_scale"],
                initial=np.r_[np.asarray(forces[seed_frame]).ravel(), torques[seed_frame]],
            )
            audit["static_retries"].append({"seed_frame": seed_frame, "statics": retry})
            if retry["pass"]:
                audit["statics"] = retry
                torques[frame] = retry["motor_torque"]
                forces[frame] = retry["forces_tln"]
                break
        print(
            f"retry frame {frame}: static={audit['statics']['pass']} attempts={len(audit['static_retries'])}",
            flush=True,
        )
    qpos = np.asarray(qpos)
    velocity = np.zeros((len(times), rig.model.nv))
    for i in range(1, len(times) - 1):
        mujoco.mj_differentiatePos(rig.model, velocity[i], 2 / fps, qpos[i - 1], qpos[i + 1])
    cycle_delta = wall_rotation(args.angle).apply([0, 0, args.step])
    geometry_pass = all(
        a["geometry"]["wall_penetration_m"] < 0.0005
        and a["geometry"]["self_penetration_m"] < 0.0005
        and not a["geometry"]["joint_limit_violations"]
        and a["ik"]["holding_position_error_m"] < 0.0005
        and a["ik"]["position_error_m"] < 0.002
        and a["ik"]["orientation_error_rad"] < np.deg2rad(5)
        for a in audits
    )
    static_pass = all(a["statics"]["pass"] for a in audits)
    # Runtime linearly interpolates positions. Bound each keyframe using both
    # adjacent segment speeds; centered differences can hide fast reversals.
    segment_speed = np.abs(np.diff(np.asarray(motor), axis=0)) * fps
    speed_bound = np.maximum(
        np.vstack((segment_speed[0], segment_speed)), np.vstack((segment_speed, segment_speed[-1]))
    )
    speed_ratios = np.max(segment_speed, axis=0) / SPEED_LIMITS
    available = speed_torque_limits(speed_bound)
    torque_speed_fraction = float(np.max(np.abs(torques) / np.maximum(available, 1e-9)))
    closure = float(np.max(np.abs(qpos[-1, 7:] - qpos[0, 7:])))
    report = {
        "robot": "ToddlerBot 2XM",
        "mass_kg": float(rig.model.body_mass.sum()),
        "wall_angle_deg": args.angle,
        "preload_n": UNIT_PRELOAD,
        "capacity_assumption": "unchanged 10 N-seating empirical distribution; user hypothesis",
        "contact_model": "one force-only connect per limb, no local contact couples",
        "unit_mass_with_adapter_kg": 0.015,
        "mount_layout": "posterior forearms and anterior shins; one rigid unit per limb",
        "mount_signature": MOUNT_SIGNATURE,
        "mounts": UNIT_MOUNTS,
        "contact_span_m": {"arms": float(targets[2, 1] - targets[3, 1]), "legs": float(targets[0, 1] - targets[1, 1])},
        "orientation_constraint": "face aligned to wall; in-plane rake follows limb with a soft preference",
        "waist_model": "ideal zero-backlash coupled tendons; original motors and torque limits",
        "motor_limits": "upstream low-speed tau_max, not continuous thermal ratings",
        "duration_s": cycle_duration,
        "step_along_wall_m": args.step,
        "frames": len(times),
        "quantile": args.quantile,
        "geometry_pass": geometry_pass,
        "static_pass": static_pass,
        "static_pass_frames": sum(a["statics"]["pass"] for a in audits),
        "static_initial_failed_frames": initial_failed_frames,
        "static_recovered_frames": [i for i in initial_failed_frames if audits[i]["statics"]["pass"]],
        "static_retry_method": "up to four nearby feasible force seeds, projected onto current equilibrium and bounds",
        "preload_frames": sum(p >= 0 for p in presses),
        "preload_pass_frames": sum(p >= 0 and a["statics"]["pass"] for p, a in zip(presses, audits)),
        "max_position_error_m": max(a["ik"]["position_error_m"] for a in audits),
        "max_holding_position_error_m": max(a["ik"]["holding_position_error_m"] for a in audits),
        "max_self_penetration_m": max(a["geometry"]["self_penetration_m"] for a in audits),
        "max_wall_penetration_m": max(a["geometry"]["wall_penetration_m"] for a in audits),
        "max_motor_speed_fraction": float(speed_ratios.max()),
        "max_motor_torque_speed_fraction": torque_speed_fraction,
        "cycle_joint_closure_error_rad": closure,
        "passed": bool(
            geometry_pass
            and static_pass
            and speed_ratios.max() < 1
            and torque_speed_fraction <= 1.00001
            and closure < 0.001
        ),
        "dynamic_execution_validated": False,
        "stochastic_reliability_validated": False,
        "frames_audit": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        qpos=qpos,
        qvel=velocity,
        motor_pos=np.asarray(motor),
        motor_torque=np.asarray(torques),
        unit_pos=np.asarray(poses),
        engaged=np.asarray(engage),
        press_unit=np.asarray(presses),
        forces_tln=np.asarray(forces),
        motor_names=np.asarray(MOTOR_NAMES),
        joint_names=np.asarray(
            [
                rig.model.joint(j).name
                for j in range(rig.model.njnt)
                if rig.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
            ]
        ),
        unit_names=np.asarray(UNIT_NAMES),
        fps=fps,
        preload_n=UNIT_PRELOAD,
        mount_signature=MOUNT_SIGNATURE,
        contact_targets=targets,
        unit_orientation_preferences=rig.unit_rotations,
        wall_angle_deg=args.angle,
        cycle_delta=cycle_delta,
        static_pass=np.asarray([a["statics"]["pass"] for a in audits]),
        x=np.asarray(xs),
    )
    report["reference_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report["empirical_table_sha256"] = hashlib.sha256(rig.table.path.read_bytes()).hexdigest()
    args.output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "frames_audit"}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--angle", type=float, default=90.0)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--torque-fraction", type=float, default=1.0)
    parser.add_argument("--kinematics", type=Path, help="Reuse a same-size cycle's IK and recompute the physics audit")
    args = parser.parse_args()
    if not (
        0 < args.angle <= 90
        and args.step > 0
        and args.fps > 0
        and 0 < args.quantile < 1
        and 0 < args.torque_fraction <= 1
    ):
        parser.error("Invalid physical parameters")
    result = generate(args)
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
