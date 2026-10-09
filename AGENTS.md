# AGENTS.md

Rules for changing this repository, for people and coding agents alike.
[README.md](README.md) covers setup, the tasks and everyday commands.

## Map

- `src/scr_phybot_mjab/tasks/<task>/`: `config/phybot_c2/` says what the task
  is, one module per section, each built in its final form; `env_cfgs.py`
  composes them and holds `apply_play_overrides`. `mdp/` holds the per-step
  terms, `motions/` the task's reference clips with their recipes and preview
  videos. `tasks/common/` holds what several tasks share.
- `src/scr_phybot_mjab/microspine/` is the pad plant; `assets/` the robot,
  wall and brick; `tools/` offline code, the clip generators among it;
  `utils/` helpers for scripts.
- `scripts/` are thin command-line wrappers; `tests/` is split by domain, with
  the repository rules checked in `tests/conventions/`.
- Retired code lives on in the local `archive/*` git tags (`git tag -l 'archive/*'`).

## Commands

- Python: `conda activate env_mjlab15`
  (`/home/spider/miniconda3/envs/env_mjlab15/bin/python`).
- Tests: `pytest`; GPU tests: `CUDA_VISIBLE_DEVICES=1 pytest --gpu`. GPU0 often
  runs other jobs; ask before using it.
- Lint: `ruff check . && ruff format --check .` (2-space indent, 120 columns),
  or `pre-commit run --files <files>`.
- Behaviour snapshots: `python scripts/dev/equivalence.py record <dir> --tiers cfg,model,clips --strict-tasks`,
  then rollouts with `--tiers rollout --jobs 6` on GPU1, and `compare <base> <new>`.
  Record the base from the commit you start from, under the ignored `artifacts/equivalence/`.

## Rules

1. Build a task's config directly: variants are parameters of one builder,
   play changes live in `apply_play_overrides`, nothing is built and then
   popped, and no environment variable switches behaviour. Delete switches
   no registered task flips.
2. Manager keys, observation term order and config field names are
   checkpoint and command-line contracts; changing one is a behaviour change.
3. Across packages, import public names only; a module that is star-imported
   declares `__all__`.
4. Comments give the reason in the present tense. Experiment history, run
   names, versions and dates belong in commit messages.
5. Scripts are thin: `main(argv=None)`, nothing runs on import beyond
   choosing the GL or plot backend, and logic another script or a test needs
   lives in `src/`. Keep new scripts under 150 lines; long single-purpose
   analysis and figure scripts are the exception. One-off probes go in the
   ignored `scratch/`.
6. Commit runtime assets with the recipe that regenerates them; never commit
   raw data, videos, logs or checkpoints. The clips record the hash of
   `assets/phybot_c2/xml/phybot_c2.xml`: change it only while regenerating
   every clip.
7. Generated files go to one of three places, taken from `tools/paths.py`,
   never from a path relative to the working directory: analyses of a
   checkpoint (figures, metrics, rollout videos) beside it in
   `<run>/eval/<checkpoint>/` (`checkpoint_eval_dir`); figures for the docs
   and the paper in `docs/figures/` (`FIGURES_DIR`), with a caption; anything
   else in the ignored `artifacts/<area>/`.

## Commits

- Work on a branch, and never push without the owner's go-ahead.
- Tag each commit `[behavior-preserving]` or `[behavior-change]`. A
  preserving commit passes the behaviour snapshots (config, model, clips and
  rollouts identical) or explains every difference; a changing one lists the
  differences it expects. Drawing a different number of random values is a
  behaviour change.
- Read `git diff --cached --stat` before every commit.
