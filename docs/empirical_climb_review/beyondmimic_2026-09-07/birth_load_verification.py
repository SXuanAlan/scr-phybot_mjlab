"""Canonical births with current initialization, recording actual release loads.

No training, PD changes or capacity changes. Terminations are disabled only
in this diagnostic so autoresets cannot hide the fate of the original grips.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
import scr_phybot_mjab.tasks  # noqa: F401
from scr_phybot_mjab.tasks.climb_wall.config.phybot_c2.env_cfgs import phybot_c2_climb_wall_env_cfg
from scr_phybot_mjab.assets.motions import CAD_CLIMB_CORNERS60_MOTION


report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
          'no_training': True, 'terminations_disabled_for_diagnosis': True,
          'reference_pd_only': True, 'cases': []}
for layout, angle in (('all', 45.), ('all', 90.), ('corners', 45.), ('corners', 60.)):
    cfg = phybot_c2_climb_wall_env_cfg(domain_rand=False, unit_layout=layout, wall_angle_deg=angle)
    cfg.scene.num_envs = 4
    cfg.terminations = {}
    motion_cfg = cfg.commands['climb_motion']
    motion_cfg.sampling_mode = 'start'
    motion_cfg.height_offset_range = (0., 0.)
    if layout == 'corners':
        motion_cfg.motion_file = str(CAD_CLIMB_CORNERS60_MOTION)
    torch.manual_seed(20260907)
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    try:
        env.reset()
        robot = env.scene['robot']
        motion = env.command_manager.get_term('climb_motion')
        aff = env.command_manager.get_term('microspine_affordance')
        grip = env.event_manager.get_term_cfg('microspine_grip').func
        act = env.action_manager.get_term('joint_pos')
        row = {'unit_layout': layout, 'wall_angle_deg': angle,
               'frames_after_reset': motion.time_steps.tolist(),
               'physical_anchors_before_first_step': env.sim.data.eq_active[:, grip._eq_ids].sum(-1).tolist(),
               'initial_seed_releases': [], 'new_latches': 0, 'first_six_steps_env0': [],
               'release_loads_env0': [], 'finite': True}
        with torch.no_grad():
            for step in range(25):
                target = robot.data.default_joint_pos.clone()
                target[:, motion._joint_ids] = motion.ref_joint_pos
                action = (target[:, act._target_ids] - act._offset) / act._scale
                obs, reward, _, _, _ = env.step(action)
                row['finite'] &= bool(torch.isfinite(reward).all() & torch.isfinite(robot.data.joint_pos).all())
                row['new_latches'] += int(aff.engaged_this_step.sum())
                for env_id, pad in (aff.forced_release_this_step & aff.released_was_seeded).nonzero().tolist():
                    row['initial_seed_releases'].append([step+1, env_id, pad, int(aff.release_reason[env_id, pad])])
                for pad in aff.forced_release_this_step[0].nonzero().flatten().tolist():
                    row['release_loads_env0'].append({
                        'step': step+1, 'pad': aff._mcfg.ee_body_names[pad],
                        'reason': int(aff.release_reason[0, pad]),
                        'unit_shear_lateral_normal_n': aff.unit_load_tln[0, pad].tolist(),
                        'unit_normal_capacity_n': aff.unit_normal_capacity[0, pad].tolist(),
                        'unit_supported': aff.unit_empirical_supported[0, pad].tolist(),
                        'unit_early_failed': aff.unit_empirical_early_failed[0, pad].tolist(),
                        'pad_anchor_force_w_n': grip.anchor_load_w[0, pad].tolist(),
                        'pad_anchor_moment_w_nm': grip.anchor_moment_w[0, pad].tolist(),
                    })
                if step < 6:
                    row['first_six_steps_env0'].append({
                        'step': step+1, 'engaged': aff.engaged[0].tolist(),
                        'hold_age': aff._steps_since_engage[0].tolist(),
                        'max_unit_shear_per_pad_n': aff.unit_load_tln[0, :, :, 0].max(-1).values.tolist(),
                        'root_velocity_w': robot.data.root_link_lin_vel_w[0].tolist(),
                        'target_clamped_joints': int(act.target_clamped[0].sum()),
                    })
        row['final_supports_without_resets'] = aff.engaged.sum(-1).tolist()
        assert row['finite'] and row['frames_after_reset'] == [0]*4
        assert row['physical_anchors_before_first_step'] == [4]*4
        report['cases'].append(row)
        Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
        print('RESULT', json.dumps({k:v for k,v in row.items() if k not in ('first_six_steps_env0', 'release_loads_env0')}), flush=True)
    finally:
        env.close()
