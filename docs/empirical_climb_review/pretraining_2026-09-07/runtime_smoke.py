"""Short runtime integration probe, not a trained-policy or climbing-success test."""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
import scr_phybot_mjab.tasks  # noqa: F401
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import phybot_c2_climb_wall_env_cfg


def all_finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(v) for v in value)
    return True


out = Path(__file__).with_suffix('.json')
report = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': '4 environments per angle, 100 policy steps, reference joint-position PD targets; no learned policy, no training, no success claim.',
    'angles': [],
}
for angle in (60, 80):
    torch.manual_seed(20260907)
    cfg = phybot_c2_climb_wall_env_cfg(
        domain_rand=False, unit_layout='corners', wall_angle_deg=angle)
    cfg.scene.num_envs = 4
    cfg.commands['climb_motion'].sampling_mode = 'start'
    cfg.commands['climb_motion'].height_offset_range = (0., 0.)
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode=None)
    try:
        env.reset()
        robot = env.scene['robot']
        motion = env.command_manager.get_term('climb_motion')
        aff = env.command_manager.get_term('microspine_affordance')
        act = env.action_manager.get_term('joint_pos')
        row = {
            'wall_angle_deg': angle, 'num_envs': 4, 'policy_steps': 100,
            'step_dt': env.step_dt, 'seeded_supports': aff.engaged.sum(-1).tolist(),
            'new_latches': 0, 'release_reason_counts': {},
            'minimum_preload_at_new_latch_n': None,
            'termination_events': 0, 'truncation_events': 0,
            'environments_with_termination': 0,
            'observations_rewards_states_finite': True,
            'env0_release_details': [],
        }
        original_update = aff._update_command

        def recorded_update():
            original_update()
            on = aff.engaged_this_step
            row['new_latches'] += int(on.sum())
            if bool(on.any()):
                preload = aff.unit_preload_n[on]
                assert bool(torch.isfinite(preload).all())
                assert bool((preload >= 10.).all()), preload
                value = float(preload.min())
                previous = row['minimum_preload_at_new_latch_n']
                row['minimum_preload_at_new_latch_n'] = value if previous is None else min(previous, value)
            reasons = aff.release_reason[aff.forced_release_this_step]
            for reason in reasons.tolist():
                key = str(reason)
                row['release_reason_counts'][key] = row['release_reason_counts'].get(key, 0) + 1
            for pad in aff.forced_release_this_step[0].nonzero().flatten().tolist():
                row['env0_release_details'].append({
                    'policy_step': int(env.episode_length_buf[0]),
                    'reference_frame': int(motion.time_steps[0]),
                    'pad': aff._mcfg.ee_body_names[pad],
                    'reason': int(aff.release_reason[0, pad]),
                    'unit_loads_shear_lateral_normal_n': aff.unit_load_tln[0, pad].tolist(),
                    'unit_normal_capacities_n': aff.unit_normal_capacity[0, pad].tolist(),
                    'unit_in_measured_domain': aff.unit_empirical_supported[0, pad].tolist(),
                    'unit_early_failure': aff.unit_empirical_early_failed[0, pad].tolist(),
                })

        aff._update_command = recorded_update
        ever_terminated = torch.zeros(4, dtype=torch.bool, device=env.device)
        with torch.inference_mode():
            for _ in range(100):
                target = robot.data.default_joint_pos.clone()
                target[:, motion._joint_ids] = motion.ref_joint_pos
                action = (target[:, act._target_ids] - act._offset) / act._scale
                obs, reward, terminated, truncated, _ = env.step(action)
                row['observations_rewards_states_finite'] &= all_finite((obs, reward, robot.data.joint_pos, robot.data.root_link_pos_w, aff.unit_load_tln))
                row['termination_events'] += int(terminated.sum())
                row['truncation_events'] += int(truncated.sum())
                ever_terminated |= terminated
        row['environments_with_termination'] = int(ever_terminated.sum())
        row['final_support_counts_after_possible_resets'] = aff.engaged.sum(-1).tolist()
        row['runtime_completed'] = True
        report['angles'].append(row)
        out.write_text(json.dumps(report, indent=2) + '\n')
        print('SMOKE_RESULT', json.dumps({k: v for k, v in row.items() if k != 'env0_release_details'}), flush=True)
    finally:
        env.close()

print('REPORT', out, flush=True)
