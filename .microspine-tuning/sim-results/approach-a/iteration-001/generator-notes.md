# Iteration 001 - Generator Notes

## Hypothesis
First implementation of the contact-triggered equality constraint approach. 
One connect constraint per hand sub-geom (8 total), initially disabled.
When hand-wall contact detected with sufficient normal force, activate the 
corresponding constraint with anchor at the contact point.

## Key Changes
1. config.py: Add MicrospineCfg fields for constraint parameters (engage threshold, solref, solimp)
2. controller.py: Full state machine implementation (IDLE -> ENGAGED) with GPU-batched constraint toggling
3. env_cfgs.py: Add spec_fn to create 8 equality connect constraints (one per sub-geom), expand eq_data for per-env anchors

## Design Decisions
- Start simple: 2-state machine (FREE/ENGAGED), skip intermediate CONTACT and SLIDING states
- Use very stiff constraint (solref=[0.001, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2])
- Engage immediately on contact detection (no hold_time filter for first iteration)
- Disengage only on large normal displacement (>5mm), since we need to hold 300N

## Expected Outcome
The hand should lock to the wall during Phase 1 (20N push), and the 300N -z shear 
in Phase 2 should be resisted by the equality constraints.
