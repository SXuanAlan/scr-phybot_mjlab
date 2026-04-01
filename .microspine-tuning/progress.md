# Progress Log

## Iteration 1 — Step 1 XML-only (friction=1.0)
- **Changes**: `impratio=10`, `integrator=implicitfast`, `condim=6`, `priority=1`, `friction=(1.0, 0.01, 0.01)`
- **Result**: FAIL — hand falls off immediately (survival ~0.2s). Root cause: no normal force in phase 2 → friction=1.0 × 0 normal = 0 friction.

## Iteration 2 — Restore high sliding friction (friction=30)
- **Changes**: Sliding friction back to 30.0, kept condim=6 + impratio=10 + priority=1
- **Result**: FAIL — still falls off (survival ~0.2s). Root cause: even with μ=30, no normal force during phase 2 because the phase 1 +x push stops and contact spring pushes hand away from wall.

## Iteration 3 — Adhesion controller
- **Changes**: Implemented MicrospineController to apply 50N adhesion force (+x toward wall) when spines are engaged. Uses hysteresis: engage when distance < 1mm, disengage when distance > 10mm.
- **Result**: PASS (32 envs)
  - z_displacement: 0.0m (S1 ✓)
  - max_z_drift: 0.0m (S2 ✓)
  - phase2_survival: 7.0s (S3 ✓, after fixing eval fencepost)
  - z_stability_std: 0.0m (S4 ✓)
  - max_x_drift: ~3e-9m (S5 ✓)
  - pct_survived_full: 1.0 (S6 ✓)
- **Verdict**: H1-H4 PASS, 6/6 soft targets → **PASS**
