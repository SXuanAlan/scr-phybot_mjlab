# Stanford Climbing Robot & PHYBot MJLab

Reinforcement-learning environments for the PHYBot C2 humanoid on MJLab
(MuJoCo Warp, RSL-RL): wall climbing on four microspine pads, the mount onto
an 80-degree wall, a two-handed brick carry, and walking, standing and
squatting for the deployed robot.

## Setup

```bash
conda activate env_mjlab15   # Python 3.11, MuJoCo 3.10, mjlab 1.5, RSL-RL 5.4
pip install -e ".[dev]"
pre-commit install
```

`environment.yml` records the full environment.

## Tasks

| Task | What it trains | Details |
| --- | --- | --- |
| `Mjlab-Climb-Wall-Phybot-C2` | Climbing with all 12 spine units per pad; wall-angle curriculum 45° → 90° | [climb_wall](docs/tasks/climb_wall.md) |
| `Mjlab-Climb-Wall-Phybot-C2-Corners-A60` | The four corner units per pad; curriculum 45° → 60° | [climb_wall](docs/tasks/climb_wall.md) |
| `Mjlab-Climb-Wall-Phybot-C2-Corners-A80` | The four corner units on a fixed 80° wall | [climb_wall](docs/tasks/climb_wall.md) |
| `Mjlab-Climb-Transition80-Phybot-C2` | Mounting from standing onto the 80° wall along the a80 clip, handing off to the climb cycle | [env_cfgs.py](src/scr_phybot_mjab/tasks/climb_transition80/config/phybot_c2/env_cfgs.py) |
| `Mjlab-Microspine-Brick-Phybot-C2` | Picking up, carrying and placing a brick with a pad on each hand | [env_cfgs.py](src/scr_phybot_mjab/tasks/brick/config/phybot_c2/env_cfgs.py) |
| `Mjlab-Microspine-Brick-Phybot-C2-DR` | The same with domain randomization | [env_cfgs.py](src/scr_phybot_mjab/tasks/brick/config/phybot_c2/env_cfgs.py) |
| `Mjlab-Microspine-Brick-Phybot-C2-Hold`, `Mjlab-Microspine-Brick-Phybot-C2-Engage`, `Mjlab-Microspine-Brick-Phybot-C2-Place` | Training stages that start from the hold, engage or place phase | [env_cfgs.py](src/scr_phybot_mjab/tasks/brick/config/phybot_c2/env_cfgs.py) |
| `Mjlab-Microspine-Brick-Phybot-C2-NoGrip` | The ablation with the pads unable to grip | [env_cfgs.py](src/scr_phybot_mjab/tasks/brick/config/phybot_c2/env_cfgs.py) |
| `Mjlab-Walk-Phybot-C2` | Velocity tracking with plain PPO | [variants.py](src/scr_phybot_mjab/tasks/walk/config/phybot_c2/variants.py) |
| `Mjlab-Stand-Default-Phybot-C2` | Recovering to and holding the standing pose (the deployed "prepare" policy) | [variants.py](src/scr_phybot_mjab/tasks/walk/config/phybot_c2/variants.py) |
| `Mjlab-Walk-Amp-Phybot-C2` | Preset gaits with an AMP style reward and the deployed PD damping | [variants.py](src/scr_phybot_mjab/tasks/walk/config/phybot_c2/variants.py) |
| `Mjlab-Squat-Phybot-C2` | Tracking one full-body squat clip | [env_cfgs.py](src/scr_phybot_mjab/tasks/squat/config/phybot_c2/env_cfgs.py) |

Transition80 checkpoints trained before it moved to the empirical pad model
(3e155ad) do not apply to the current task.

## Common commands

```bash
python scripts/list_envs.py
python scripts/train.py Mjlab-Climb-Wall-Phybot-C2 --gpu-ids "[1]" --env.scene.num-envs 4096
python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 --checkpoint-file logs/rsl_rl/<experiment>/<run>/model_<iter>.pt
python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 --agent zero
python scripts/export_policy.py Mjlab-Walk-Phybot-C2 --checkpoint-file logs/rsl_rl/phybot_c2_walk/<run>/model_<iter>.pt
tensorboard --logdir logs/rsl_rl
```

- Resume training: add `--agent.resume --agent.load-run <run> --agent.load-checkpoint model_<iter>.pt`.
- Play picks the climb wall angle that the checkpoint's iteration reached;
  `--wall-angle-deg 90` overrides it. `--video` records the rollout.
- Analyses of a checkpoint (play videos, the climb exam, torque and pad
  wrench plots) land beside it, in `<run>/eval/model_<iter>/`.
- Run long jobs in `tmux` (`tmux new -s train`, detach with `Ctrl-b d`,
  return with `tmux attach -t train`).
- Every script prints its options with `--help`. `scripts/` holds the four
  commands above at the top and the rest by purpose: `train/`, `eval/`,
  `reference/`, `validate/`, `microspine/`, `deploy/` and `dev/`.

## Tests and checks

```bash
pytest                                    # CPU tests
CUDA_VISIBLE_DEVICES=1 pytest --gpu       # also the GPU integration tests
ruff check . && ruff format --check .
pre-commit run --files $(git ls-files)    # every hook (--all-files needs git >= 2.31)
```

Refactors are checked against recorded behaviour snapshots with
`scripts/dev/equivalence.py` (see [AGENTS.md](AGENTS.md)).

## Layout

```text
src/scr_phybot_mjab/
  assets/       the C2 robot (MJCF, meshes), wall, brick, motion clips and their recipes
  microspine/   the pad plant: contact features, engage gate, hold judge, equality anchors
  tasks/        one package per task: config/phybot_c2/ (what the task is), mdp/ (per-step terms)
    common/     pieces the tasks share
  tools/        offline code: reference generation, microspine analysis, evaluation, the equivalence harness
  utils/        helpers for scripts: building envs, loading policies, CLI options
scripts/        thin command-line wrappers
tests/          microspine/, tasks/<task>/, reference/, tools/, conventions/
docs/           see docs/README.md; docs/figures/ holds the figures for the paper
data/           inputs: the CAD model and bench data (raw trials are not in git)
logs/rsl_rl/    training runs (ignored); each checkpoint's analyses in <run>/eval/<checkpoint>/
artifacts/      other generated outputs (ignored)
```

## Data and assets

- Raw microspine trials live in `data/microspine/raw/` and are not in git;
  `data/microspine/empirical_unit_v1/` records how the packaged unit model
  was fitted.
- Each task keeps its reference clips in `tasks/<task>/motions/`. The climb
  and mount clips carry a `<clip>.recipe.txt` with the command that
  regenerates them; preview videos (`<clip>.mp4`) sit beside the clips, out
  of git.
- The clips record the hash of `assets/phybot_c2/xml/phybot_c2.xml`; change
  that file only together with regenerating every clip.
- Logs, checkpoints, W&B runs and videos stay out of git.

Rules for changing the code, for people and coding agents alike, are in
[AGENTS.md](AGENTS.md).
