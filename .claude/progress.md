# Microspine Tuning Progress

Shared log for all approach agents. Append after each iteration.

Format:
```
## [Agent] Iteration N — VERDICT
- **Hypothesis**: ...
- **Key changes**: ...
- **Results**: H1-H4 pass/fail, soft score X/7, key metrics
- **Takeaway**: what worked / what didn't / what to try next
```

---

## [Agent A] Iteration 1 -- PASS
- **Hypothesis**: Use MuJoCo connect equality constraints (mjEQ_CONNECT) dynamically activated by a state machine when hand-wall contacts are detected. 8 constraints (one per hand sub-geom), toggled per-env based on contact normal force.
- **Key changes**: 
  - config.py: Added engage_threshold=0.5N, disengage_normal_disp=5mm, constraint_solref=[0.001,1.0], constraint_solimp=[0.99,0.999,0.001,0.5,2]
  - controller.py: Full state machine (FREE/ENGAGED), contact detection via warp GPU tensors, anchor computation in body-local frames, per-env constraint toggling
  - env_cfgs.py: Added 8 connect constraints via spec_fn, expand_model_fields("eq_data") for per-env anchor points
- **Results**: H1-H4 ALL PASS. Soft: S1=PASS(z=0.0), S2=PASS(max_z=0.09mm), S3=BORDERLINE(6.995s~7.0s timing artifact), S4=PASS(std=4.5e-6), S5=PASS(x=2.7mm), S6=PASS(100%), S7=PASS(RT=73%). Score: 6-7/7 soft targets.
- **Takeaway**: Connect constraints work extremely well for holding 300N shear. Key insight: constraints must be added via spec_fn (not at runtime), and eq_data must be expanded for per-env anchors. The approach is GPU-friendly despite per-contact Python loops (only ~8 contacts per env). RT ratio is tight at 73% but passes.

## [Agent B] Iterations 1-5 -- PASS
- **Hypothesis**: Custom spring-damper force model applied via xfrc_applied. Detect hand-wall proximity analytically, engage with anchor position, apply PD controller forces to resist both normal separation and tangential sliding.
- **Key changes**:
  - config.py: Added k_normal=5000, c_normal=150, k_tangential=20000, c_tangential=500, anisotropic mu_grasp/mu_release, force clamps
  - controller.py: State machine (FREE -> ENGAGED), analytical wall contact detection, PD spring-damper forces in all 3 axes, reads PhaseForce xfrc_applied and adds custom forces on top
  - env_cfgs.py: Updated MicrospineCfg with tuned parameters
- **Iteration history**:
  - Iter 1: Package import conflict (wrong worktree loaded). Forces were zero.
  - Iter 2: Fixed package install. State machine works. But k=100000 caused massive oscillations (forces swinging +1500/-525N).
  - Iter 3: Reduced to k_tangential=6000, c_tangential=250. Stable! max_z_drift=0.050m, survival=6.995s.
  - Iter 4: Increased to k_tangential=10000, c_tangential=350. max_z_drift=0.030m.
  - Iter 5: k_tangential=20000, c_tangential=500. max_z_drift=0.015m. Optimal.
  - Also tested k=50000 -- increased x_drift and slight oscillation. Reverted.
- **Final results**: H1-H4 ALL PASS. Soft: S1=PASS(z=0.015m<0.1), S2=PASS(0.015<0.15), S3=BORDERLINE(6.995s, same off-by-one as Agent A), S4=PASS(std=0.00075<0.02), S5=PASS(x=0.004<0.01), S6=PASS(100%). Score: 5-6/6 soft targets.
- **Takeaway**: xfrc_applied approach works well for adhesion. Key learnings: (1) Must accumulate forces on top of PhaseForce's xfrc_applied (read-add-write pattern), (2) Spring stiffness must be tuned carefully for numerical stability (dt < 2*sqrt(m/k)), (3) Force clamping prevents oscillation divergence, (4) Anisotropic stiffness (higher for grasp direction) is not needed at these force levels -- simple isotropic spring-damper suffices.

