"""Export reviewable task defaults; does not create an environment or runner."""
import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import scr_phybot_mjab.tasks  # noqa: F401
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.registry import load_env_cfg


def serial(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: serial(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, slice):
        return {'slice': [value.start, value.stop, value.step]}
    if callable(value):
        return value.__module__ + '.' + value.__qualname__
    if isinstance(value, Path):
        return str(value)
    return value


report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
          'training_started': False, 'tasks': {}}
for task in ('Mjlab-Climb-Wall-Phybot-C2', 'Mjlab-Climb-Wall-Phybot-C2-Corners-A60',
             'Mjlab-Climb-Wall-Phybot-C2-Corners-A80'):
    training = TrainConfig.from_task(task)
    cfg, agent = training.env, training.agent
    play = load_env_cfg(task, play=True)
    motion = cfg.commands['climb_motion']
    report['tasks'][task] = serial({
        'gpu_ids': training.gpu_ids, 'num_envs_per_worker': cfg.scene.num_envs,
        'episode_length_s': cfg.episode_length_s, 'physics_dt_s': cfg.sim.mujoco.timestep,
        'decimation': cfg.decimation, 'nconmax': cfg.sim.nconmax, 'njmax': cfg.sim.njmax,
        'actor_observation_dimension_runtime': 150, 'critic_observation_dimension_runtime': 170,
        'actions': cfg.actions, 'motion': motion,
        'microspine': cfg.commands['microspine_affordance'],
        'curriculum': cfg.curriculum, 'rewards': cfg.rewards, 'terminations': cfg.terminations,
        'domain_randomization_events': {k:v for k,v in cfg.events.items()
                                       if k != 'microspine_grip'},
        'runner': agent,
        'evaluation_motion': play.commands['climb_motion'], 'evaluation_curriculum': play.curriculum,
        'reference_sha256': hashlib.sha256(Path(motion.motion_file).read_bytes()).hexdigest(),
    })
Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
print('Configuration exported; no environment or training runner was started.')
