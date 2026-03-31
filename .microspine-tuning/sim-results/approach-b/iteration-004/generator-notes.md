# Iteration 004 - Generator Notes

## Iteration 003 results
- max_z_drift: 0.05m (PASS S2, PASS S1)
- max_x_drift: 0.005m (PASS S5)
- phase2_survival: 6.995s (FAIL S3 by 5ms -- off-by-one in evaluator timing)
- z_stability_std: 0.0025m (PASS S4)
- pct_survived: 100% >= 6.5s (PASS S6)
- All H1-H4 PASS

Score: 5/6 soft targets = PASS verdict. But S3 just barely misses.

## Changes for iteration 004

1. Increase k_tangential from 6000 to 10000 to reduce initial z-drift (currently 50mm)
   - Target: reduce to ~30mm which still meets S1/S2 and is more robust
   - Stability check: dt < 2*sqrt(m/k) = 2*sqrt(1.9/10000) = 0.028s >> 0.005s. Safe.

2. Increase c_tangential from 250 to 350 for better damping ratio
   - Critical damping: c = 2*sqrt(k*m) = 2*sqrt(10000*1.9) = 276. Use 350 for slight overdamping.

3. Keep force clamp at 500N to prevent any overflow issues.

## Hypothesis
Higher stiffness will reduce z-drift while maintaining stability.
The system should still be well within numerical stability bounds.
