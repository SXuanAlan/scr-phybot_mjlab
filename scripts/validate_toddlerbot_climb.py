"""Measure a bounded PD reference replay in real MJLab physics (no trained policy)."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv

from scr_phybot_mjab.assets.toddlerbot.toddlerbot_constants import MOTOR_NAMES, TORQUE_LIMITS
from scr_phybot_mjab.tasks.toddler_climb.env_cfg import toddler_climb_env_cfg
from scr_phybot_mjab.tools.climbing_reference.toddlerbot import REFERENCE_PATH, load_reference


@torch.inference_mode()
def validate(args):
    archive = load_reference(args.reference)
    cfg = toddler_climb_env_cfg(
        angle_deg=float(archive["wall_angle_deg"]), stochastic=args.stochastic, reference_file=str(args.reference)
    )
    cfg.scene.num_envs = args.num_envs
    cfg.auto_reset = False
    cfg.seed = 42
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        obs, _ = env.reset()
        command = env.command_manager.get_term("climb_reference")
        grip = env.event_manager.get_term_cfg("microspine_grip").func
        assert env.sim.data.eq_active[:, grip._eq_ids].all()
        m = env.sim.mj_model
        assert all(m.eq_type[i] == mujoco.mjtEq.mjEQ_CONNECT for i in grip._eq_ids.cpu().tolist())
        torch.testing.assert_close(command.robot.data.joint_pos[:, command.motor_ids], command.joint_pos_ref)
        action = env.action_manager.get_term("joint_pos")
        assert tuple(action.target_names) == MOTOR_NAMES
        frames = []
        finite = True
        terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        actual_ids = [m.actuator("robot/" + name).id for name in MOTOR_NAMES]
        limits = torch.tensor(TORQUE_LIMITS, device=env.device)
        for i in range(round(args.seconds / env.step_dt)):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros((env.num_envs, 30), device=env.device))
            finite = bool(torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values()))
            if not finite:
                break
            assert (env.sim.data.ne <= env.sim.data.njmax).all()
            error = torch.linalg.vector_norm(command.robot.data.root_link_pos_w - command.sample("qpos")[:, :3], dim=-1)
            fraction = (env.sim.data.actuator_force[:, actual_ids].abs() / limits).max()
            assert float(fraction) <= 1.00001
            frames.append(
                {
                    "time_s": (i + 1) * env.step_dt,
                    "grips": command.engaged.cpu().tolist(),
                    "preload_n": command.preload.cpu().tolist(),
                    "unit_load_tln": command.loads.cpu().tolist(),
                    "root_error_m": error.cpu().tolist(),
                    "peak_motor_fraction": float(fraction),
                    "failed_grips": command.failed_grips.cpu().tolist(),
                }
            )
            if bool((terminated | truncated).any()):
                break
        # Partial reset must leave other worlds' reference clocks, randomness
        # and anchor positions intact.
        fields = ("elapsed", "engaged", "strength", "loading")
        saved = {k: getattr(command, k)[1:].clone() for k in fields}
        anchors = env.sim.model.eq_data[1:, grip._eq_ids].clone()
        env.reset(env_ids=torch.tensor([0], device=env.device))
        for k, v in saved.items():
            torch.testing.assert_close(getattr(command, k)[1:], v)
        torch.testing.assert_close(env.sim.model.eq_data[1:, grip._eq_ids], anchors)
        completed = bool(finite and len(frames) == round(args.seconds / env.step_dt) and not terminated.any())
        report = {
            "reference": str(args.reference),
            "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "device": args.device,
            "num_envs": args.num_envs,
            "seed": 42,
            "stochastic": args.stochastic,
            "controller": "zero residual action; PD with audited static feedforward and servo speed limits",
            "trained_policy": False,
            "finite_physics": finite,
            "partial_reset_isolated": True,
            "requested_seconds": args.seconds,
            "simulated_seconds": len(frames) * env.step_dt,
            "completed_without_termination": completed,
            "frames": frames,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH.with_name("toddlerbot_climb_a60.npz"))
    parser.add_argument("--output", type=Path, default=Path("docs/toddlerbot_climb/replay_a60.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=14.0)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0 or args.num_envs < 2:
        parser.error("Use positive duration and at least two environments")
    report = validate(args)
    raise SystemExit(0 if report["completed_without_termination"] else 2)


if __name__ == "__main__":
    main()
