# Four-corner 60-degree checkpoint evaluation

`scripts/evaluate_climb_checkpoint.py` evaluates the registered
`Mjlab-Climb-Wall-Phybot-C2-Corners-A60` PPO actor. It verifies every loaded
actor tensor, including observation normalization, and checks the actual
wall angle and four active empirical units per pad at runtime.

```sh
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/spider/miniconda3/envs/env_mjlab15/bin/python \
  scripts/evaluate_climb_checkpoint.py \
  --checkpoint logs/rsl_rl/phybot_c2_climb_wall_corners60/RUN/model_250.pt \
  --angles 40 45 50 --num-envs 32 --duration-s 30 \
  --profile canonical --output-dir logs/eval/RUN/model_250_canonical
```

`--angle-deg` is an alias for `--angles`. Angles must be explicit: play's
default final 60-degree angle is never silently used for an early-stage
exam. Use the current band's low, center and high angles; repeat previously
mastered angles to expose regression. The supervisor owns when to test and
when to extend or advance the curriculum.

`canonical` starts from reference frame 0, with ordinary domain
randomization off. `training-rsi` retains training observation noise,
friction, mass, encoder bias and height offsets, and uses uniformly sampled
full-cycle reference phases for coverage. Empirical per-unit capacity and
early-loading draws remain enabled in both profiles. `--seed` controls the
reproducible draws. Neither profile modifies rewards, capacity, controllers,
nor failure thresholds. The 30-second training timeout is retained.

Automatic reset is disabled only so each terminal state can be measured
and rendered before explicitly resetting done worlds. Initial seeded grips
and physical new latches are separate fields. The trajectory includes every
reset, release reason (including empirical reasons 7 and 8), seed provenance,
per-unit loads, phase and reference/actual along-wall altitude. Statistics
used for curriculum decisions are computed from the **first continuous
episode of each world**. Later resets cannot assemble a successful climb.

For each angle the output directory contains:

- `metrics.json`: checkpoint SHA/iteration, hardware, runtime angles,
  per-environment first-episode metrics and all episode endings.
- `trajectory.npz`: time-aligned arrays with shape `(step, env, ...)`.
- `engage_schedule.png`: environment 0 reference/actual engagement; yellow
  means an initial seed, green a policy-created grip. Red reset lines and
  along-wall altitude expose falling and respawning.
- `rollout.mp4`: the same physical environment 0 rollout, with timestamp,
  episode, phase, actual/reference engagement and seed state overlaid.
- `video_contact_sheet.png` and `event_*.png`: regular rollout samples and
  the first release, first real new latch, and first termination if present.
  Event stills may fall between the regular 25 fps video samples; their
  labels show exact simulation time.

The root `summary.json` is atomically published with `complete: false`
until all requested angles finish. Its `results[]` entries hold metrics
directly (there is no nested `metrics` object). The supervisor can use:

| Field | Meaning |
|---|---|
| `engage_match_min` | Worst first-episode per-world raw reference agreement |
| `first_episode_falls` | Worlds whose first episode has a non-timeout termination |
| `full_30s_survival_rate` | Fraction that survives a complete 30 seconds |
| `pad_occupancy_excess_max_per_pad` | Worst per-world actual minus reference occupancy for LH/RH/LF/RF |
| `progress_ratio_min` | Worst along-wall robot/reference progress ratio after the same 0.5-second settling interval; null means insufficient valid evidence |
| `reference_cycles_completed_min` | Fewest complete 9-second reference cycles in a first episode |
| `true_regrip_cycles_min` | Fewest physical new latches on any pad in any first episode |
| `per_env` | Individual-world values underlying these conservative reductions |

The automatic candidate check requires 30 seconds, zero first-episode
failures, at least two complete reference cycles, at least two physical new
latches per pad per world, progress ratio at least 0.8, worst-world engage
match at least 0.95, and every pad's occupancy excess below 0.04. A normal
timeout at 30 seconds counts as survival; a short smoke test cannot pass.
Visual review remains required. The supervisor separately requires repeated
passing exams and reviews bound to the checkpoint SHA before promotion.

Validation: the CPU tests in
`tests/microspine/test_climb_checkpoint_evaluation.py` cover reset exclusion,
single-use event matching, static seeded holds, timeout versus short exams,
worst-world reductions, and numeric divergence reporting. The GPU smoke
at `logs/eval/climb60_eval_smoke_model0/` strictly loaded the new run's
`model_0.pt`, evaluated 4 worlds for 1 second at 45 degrees, and produced all
artifacts. All 4 first episodes failed at 0.46 seconds, with zero new latches;
the examined raster and contact sheet show the four seeds release at 0.06
seconds and explicit resets at 0.46 and 0.92 seconds. This verifies the
evaluation pipeline and records the initial checkpoint baseline (after the
first PPO update), not climbing success.
