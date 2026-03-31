# Iteration 003 - Generator Notes

## What failed in iteration 002

1. The state machine works -- controller engages at step 4 and stays ENGAGED.
2. BUT massive oscillations in all force components due to:
   - k_tangential = 100000 N/m is too stiff for dt=0.005s
   - velocity estimation via finite differences amplifies noise
   - Friction cone limit of mu_grasp * f_normal = 12 * 150 = 1800N is way too high
3. Forces swing between +1500 and -525 in z-direction instead of stabilizing at ~300N.

## Root cause analysis
Total mass is ~1.9 kg (elbow_pitch 0.85 + elbow_yaw 0.75 + hand 0.3).
For spring-damper stability: dt < 2*sqrt(m/k).
k=100000, m=1.9: critical dt = 0.0087s. step_dt=0.005s is near the boundary.
But the real issue is that the FORCE MAGNITUDE is too large. We need ~300N to hold,
but the spring generates thousands of N, creating overshoot.

## Changes for iteration 003

Strategy: Use much softer springs that produce forces in the right range (~300N).

For z-direction: We need F = 300N at equilibrium. With k_z and the hand drifting
some distance delta_z before the spring force catches up:
- If we want F=300N at delta_z=0.003m (3mm drift): k_z = 300/0.003 = 100000 N/m (too stiff)
- If we want F=300N at delta_z=0.01m (10mm drift): k_z = 300/0.01 = 30000 N/m (still stiff)
- If we want F=300N at delta_z=0.05m (50mm drift): k_z = 300/0.05 = 6000 N/m (moderate)

For stability: dt < 2*sqrt(m/k) = 2*sqrt(1.9/6000) = 0.036s. Much safer.

Critical damping: c = 2*sqrt(k*m) = 2*sqrt(6000*1.9) = 213 N*s/m

Actually, a better approach: make the tangential force independent of spring stiffness.
Use a PD controller that targets the anchor position directly:
  F_z = k_z * (anchor_z - hand_z) + c_z * (0 - v_z)

But clamp the maximum force to a reasonable value (e.g., 400N).
This way, even if displacement gets large, force stays bounded.

For x-direction: lighter spring to keep hand near wall.
k_x = 5000, c_x = 150. Target force ~20N at 4mm displacement.

Parameters:
- k_tangential = 6000 N/m (was 100000)
- c_tangential = 250 N*s/m (was 500)
- k_normal = 5000 N/m (was 50000)
- c_normal = 150 N*s/m (was 200)
- Remove friction cone clamping entirely for now -- just use direct spring force
- Add force magnitude clamp: max 400N per axis

## Expected outcome
Much smoother force response. Small z-drift (maybe 50mm) as spring builds up to 300N.
Should survive much longer in Phase 2.
