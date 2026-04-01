# Iteration 5 Report: 1024-env Microspine Evaluation

## Changes Made

### Problem 1: OOM Fix (microspine_test_env_cfg.py)
Reduced memory/performance parameters only (no physics changes):
- `ccd_iterations`: 2000 -> 100
- `njmax`: 1000 -> 200
- `nconmax`: 500 -> 100

### Problem 2: RT Measurement (evaluate_microspine.py)
Added wall-clock timing around the simulation loop:
- CUDA sync before/after loop for accurate timing
- Computes `rt_ratio = simulated_time / wall_clock_time`
- Reports `rt_ratio`, `wall_clock_s`, `simulated_time_s` in mean metrics

## Results (1024 envs, cuda:0, RTX 5070 Ti)

| Metric | Value |
|--------|-------|
| final_z_displacement_m | 0.0 |
| final_x_displacement_m | 0.0 |
| max_z_drift_m | 0.0 |
| max_x_drift_m | 0.0 |
| phase2_survival_s | 7.0 |
| z_stability_std | 0.0 |
| pct_survived_full | 1.0 (100%) |
| rt_ratio | 0.844 (84.4%) |
| wall_clock_s | 11.84 |
| simulated_time_s | 10.0 |

## Eval Criteria Assessment

### Hard Targets (all must pass)

| ID | Criterion | Result |
|----|-----------|--------|
| H1 | Phase 1 unchanged | PASS - no physics params changed |
| H2 | No anti-cheat | PASS - only reduced solver buffers |
| H3 | Final z displacement < 0.5m | PASS (0.0m) |
| H4 | Hand stays stable, no vibration | PASS (z_stability_std = 0.0) |

### Soft Targets (need >= 5/7)

| ID | Criterion | Threshold | Value | Result |
|----|-----------|-----------|-------|--------|
| S1 | Final z displacement | < 0.1m | 0.0m | PASS |
| S2 | Max z drift | < 0.15m | 0.0m | PASS |
| S3 | Phase 2 survival | >= 7.0s | 7.0s | PASS |
| S4 | Z stability std | < 0.02m | 0.0m | PASS |
| S5 | Max x drift | < 0.01m | 0.0m | PASS |
| S6 | Envs survive full phase 2 | >= 100% | 100% | PASS |
| S7 | RT ratio | >= 70% | 84.4% | PASS |

**Soft targets passed: 7/7**

### Overall: PASS (4/4 hard + 7/7 soft)
