# Approach Contract

## Goal
Make the forearm stay stable at its original position when 300N -z force is applied
in Phase 2 (after 3s) of the `Mjlab-Microspine-Test-Forearm-Hand` task.

## Constraints
- Phase 1 (0-3s): 20N +x push — DO NOT MODIFY
- Phase 2 (3s+): 300N -z shear — hand must resist this
- No velocity freeze, no weld constraints, no artificial counter-forces
- Can't change the phase 1 and phase 2 force
- Keep code GPU-friendly
- Can't read the git history

## Three Parallel Approaches

### Path A: MJCF Contact Parameter Tuning
Tune friction, solref, solimp, adhesion gain, and solver parameters in the contact
pairs and MicrospineCfg to maximize holding force through MuJoCo's built-in contact model.

### Path B: Enhanced Spine Physics + Directional Damping
Modify the MicrospineController to apply directional contact damping (heavy damping
in -z when spines are engaged) and increase spine count/force capacity to physically
resist the 300N load.

### Path C: Solver + Adhesion Overhaul
Increase MuJoCo solver iterations, switch to more aggressive impedance parameters,
maximize adhesion actuator gain, and tune the preload/engagement thresholds for
faster and stronger spine engagement.
