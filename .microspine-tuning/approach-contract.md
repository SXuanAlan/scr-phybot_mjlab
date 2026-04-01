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

## Three-Step Approach (from microspine-method.md)

Based on MuJoCo official XML examples (gripper.xml, touch_grid.xml, etc.), the new
approach replaces the old Path A/B/C with a progressive 3-step method:

### Step 1: XML-Only Contact Parameter Overhaul (Current)
- `cone="elliptic"` + `impratio=10` + `integrator="implicitfast"` in `<option>`
- `condim=6` (full 6D contact: normal + 2 sliding + torsional + 2 rolling)
- `priority=1` on hand sub-geoms (spine parameters dominate over wall)
- Friction `(1.0, 1.0, 0.01, 0.01, 0.01)` — impratio amplifies torsional/rolling 10x
- No Python control logic changes needed

### Step 2: Flexcomp Deformable Contact Pads (If Step 1 insufficient)
- Replace rigid box sub-geoms with `flexcomp` trilinear deformable pads
- `dof="trilinear"`, `young=5000`, `poisson=0.1`, `damping=0.1`
- Deformable surface conforms to wall, increasing effective contact area
- Pin back vertices to hand body, allow front face to deform

### Step 3: Touch Grid Sensors (For RL/control)
- Add `mujoco.sensor.touch_grid` plugin to each spine array center
- 5x5 or 7x7 grid, 3-channel (normal + 2 tangential forces)
- Provides spatial force distribution for engage/disengage detection
- Direct input to RL observation space
