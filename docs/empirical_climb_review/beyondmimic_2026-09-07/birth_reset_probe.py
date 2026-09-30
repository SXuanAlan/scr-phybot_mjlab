"""Compare reference-PD births before/after immediate anchor initialization.

No learned policy, extra immunity, capacity change or training is used.
The mature variant separately tests whether an RSI latch should reseat.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
import scr_phybot_mjab.tasks  # noqa: F401
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import phybot_c2_climb_wall_env_cfg


report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'rows': []}
for angle in (45., 60., 80.):
    cfg = phybot_c2_climb_wall_env_cfg(
        domain_rand=False, unit_layout='corners', wall_angle_deg=angle)
    cfg.scene.num_envs = 4
    cfg.commands['climb_motion'].sampling_mode = 'start'
    cfg.commands['climb_motion'].height_offset_range = (0., 0.)
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    try:
        robot = env.scene['robot']
        motion = env.command_manager.get_term('climb_motion')
        aff = env.command_manager.get_term('microspine_affordance')
        grip = env.event_manager.get_term_cfg('microspine_grip').func
        act = env.action_manager.get_term('joint_pos')
        for variant in ('baseline', 'immediate', 'immediate_mature'):
            torch.manual_seed(20260907)
            env.reset()
            before = env.sim.data.eq_active[:, grip._eq_ids].sum(-1).tolist()
            if variant != 'baseline':
                grip.assert_now(env)
                if variant == 'immediate_mature':
                    grip._engage_age[:] = grip._reseat_max_age + 1
                env.sim.forward()
            row = {'wall_angle_deg': angle, 'variant': variant,
                   'physical_anchors_before': before,
                   'physical_anchors_at_first_step': env.sim.data.eq_active[:, grip._eq_ids].sum(-1).tolist(),
                   'releases': [], 'new_latches': 0, 'trace_env0': [], 'finite': True}
            with torch.no_grad():
                for step in range(100):
                    target = robot.data.default_joint_pos.clone()
                    target[:, motion._joint_ids] = motion.ref_joint_pos
                    action = (target[:, act._target_ids] - act._offset) / act._scale
                    obs, reward, terminated, truncated, _ = env.step(action)
                    row['finite'] &= bool(torch.isfinite(robot.data.joint_pos).all())
                    row['new_latches'] += int(aff.engaged_this_step.sum())
                    if step < 5:
                        row['trace_env0'].append({
                            'step': step+1, 'supports': aff.engaged[0].tolist(),
                            'max_shear_per_pad': aff.unit_load_tln[0, :, :, 0].max(-1).values.tolist(),
                            'anchor_load_w': grip.anchor_load_w[0].tolist(),
                            'root_velocity_w': robot.data.root_link_lin_vel_w[0].tolist(),
                            'clamped_targets': int(act.target_clamped[0].sum()),
                        })
                    for env_id, pad in aff.forced_release_this_step.nonzero().tolist():
                        row['releases'].append([step+1, env_id, pad, int(aff.release_reason[env_id, pad])])
            row['final_supports'] = aff.engaged.sum(-1).tolist()
            report['rows'].append(row)
            Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
            print('RESULT', json.dumps({k:v for k,v in row.items() if k != 'trace_env0'}), flush=True)
    finally:
        env.close()
