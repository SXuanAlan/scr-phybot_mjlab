# CLAUDE.md

## Project

MuJoCo humanoid robot simulation (mjlab framework). Implementing microspine adhesion contact model.

## Build & Run

```bash
pip install -e .
# Headless evaluation (outputs scores.json with verdict)
python scripts/evaluate_microspine.py --num-envs 16 --device cuda:0 --output .microspine-tuning/sim-results/iteration-NNN/scores.json
```

## Key Files

```
src/spider_rl_mjlab/
├── assets/microspine/
│   ├── config.py              # MicrospineCfg dataclass
│   └── controller.py          # MicrospineController (skeleton, implement here)
├── assets/forearm_hand/
│   └── forearm_hand_constants.py  # Robot spec, actuators, collision config
├── assets/wall/
│   └── wall_constants.py      # Wall entity
└── tasks/microspine_test/
    ├── microspine_test_env_cfg.py           # Base env config (sim params, rewards)
    ├── config/forearm_hand/env_cfgs.py      # Forearm-hand config (contact pairs, events)
    └── mdp/
        ├── phase_force.py     # Two-phase test force (DO NOT MODIFY)
        ├── events.py          # apply_timed_force
        ├── rewards.py         # hand_on_wall, hand_height
        └── contact_force_vis.py  # Debug visualization

scripts/evaluate_microspine.py   # Headless evaluator → scores.json + verdict
.microspine-tuning/
├── eval-criteria.md             # Shared evaluation criteria (H1-H4, S1-S7)
└── mujoco-prior-knowledge.md    # MuJoCo contact model reference
```

## Constraints (ALL agents must follow)

- Phase 1 (0-3s): 20N +x push — DO NOT MODIFY
- Phase 2 (3s+): 300N -z shear — DO NOT MODIFY
- `mdp/phase_force.py` is the test harness — DO NOT MODIFY
- No velocity freeze, no weld constraints, no artificial counter-forces
- Keep code GPU-friendly (batched tensor ops)
- Cannot read git history

## Evaluation

Run `python scripts/evaluate_microspine.py` after each code change. It checks:
- **Hard constraints** H1-H4: must all pass (phase unchanged, no cheat, z < 0.5m, no vibration)
- **Soft targets** S1-S7: need >= 5/7 to PASS
- **Verdict**: PASS / FAIL / PIVOT

## Code Style

- Python, no type stubs
- Use existing patterns (see contact_force_vis.py for class-based step events)
- Imports: `from __future__ import annotations` at top
