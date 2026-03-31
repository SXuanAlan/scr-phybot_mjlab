# Iteration 002 - Generator Notes

## What failed in iteration 001

1. Geometry mismatch: hand body origin is at x=-0.002, sub-geoms at local +0.08.
   contact_x = -0.002 + 0.08 = 0.078. Wall face at 0.08. Gap = 2mm. Never enters CONTACT.
2. Sub-geom box half-size is 0.002m in x. So the actual contact surface ranges from 0.076 to 0.080.
   Need to use the sub-geom surface position (local_x + half_size_x), not just the center.
3. Even if penetration were detected, the model only applied friction forces and relied on
   MuJoCo contact for normal force. But once Phase 2 starts (no +x push), the hand separates.
4. The model needs to HOLD the hand to the wall once engaged (adhesion in normal direction too).

## Changes for iteration 002

1. Fix contact detection: Use proper penetration calculation accounting for sub-geom box size.
   Actual contact surface x = hand_body_x + geom_local_x + geom_half_size_x
   = hand_body_x + 0.08 + 0.002 = hand_body_x + 0.082
   Penetration = contact_surface_x - wall_face_x

2. Model engagement as a mechanical interlock:
   - Once engaged (penetration exceeded engage_depth), apply BOTH:
     a) Normal adhesion force (+x, toward wall) to keep hand pressed
     b) Tangential friction force (+z, resisting shear) based on engagement strength
   - The engagement force should be proportional to how well the spines are embedded

3. Simplify the model: Instead of purely friction-based, use a direct spring-damper
   in the z-direction anchored at the engagement position. This models the mechanical
   interlock of microspine hooks.

## Hypothesis
The mechanical interlock model (spring-damper anchor in all 3 axes after engagement)
should prevent the hand from both separating from the wall AND sliding down.
The key force is ~300N in +z direction to counteract the 300N -z shear.
