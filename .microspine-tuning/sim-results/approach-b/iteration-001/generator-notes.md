# Iteration 001 - Generator Notes

## Hypothesis
First implementation of the custom spring-damper-friction force model via xfrc_applied.
The hand body (right_hand) has its +x face pressing against the wall inner surface.
The wall inner face is at x = wall_center_x - wall_half_size_x.

The key insight: PhaseForce writes to xfrc_applied first (overwriting), then our controller
runs second. We need to READ the current xfrc_applied after PhaseForce has set it, then
ADD our adhesion forces and write the combined result back.

## Force Model Design

### Wall geometry
- Wall center at (0.58, 0, 2.5) with half_size [0.5, 1.5, 2.5]
- Wall inner face (facing -x) at x = 0.58 - 0.5 = 0.08
- Wall outward normal (facing robot) is (-1, 0, 0) (pointing in -x direction)
- Robot hand at x ~0.08 with sub-geoms at local pos (0.08, y, z)

### Contact detection
- Use hand body x-position vs wall inner face x-position
- Penetration depth = hand_pos_x + hand_geom_offset_x - wall_face_x
  (positive = penetrating wall)

### Force components
1. Normal (spring-damper toward wall in +x when engaged): keeps hand pressed to wall
2. Tangential friction (in z-direction): resists the 300N -z shear force
   - Anisotropic: higher mu for -z direction (grasp), lower for +z (release)
3. Adhesion: small pull-back force when just separated

### State machine
- FREE -> CONTACT -> ENGAGED_STICK / ENGAGED_SLIP -> ADHESION -> FREE

### Initial parameters (conservative)
- k_normal = 50000 N/m
- c_normal = 200 N*s/m
- engage_depth = 0.003 m (3mm)
- k_tangential = 100000 N/m
- c_tangential = 500 N*s/m
- mu_grasp = 12.0 (high friction along grasp/down direction)
- mu_release = 1.5 (low friction against grasp/up direction)
- adhesion_depth = 0.001 m
- k_adhesion = 10000 N/m
- F_adhesion_max = 50 N

### Expected outcome
The high anisotropic friction in -z direction should generate enough tangential
resistance to counteract the 300N shear force. With mu_grasp=12 and even moderate
normal force (25N from penetration), friction capacity = 12 * 25 = 300N.

## Changes
- config.py: Add all force model parameters to MicrospineCfg
- controller.py: Full spring-damper-friction model with state machine, xfrc_applied accumulation
- env_cfgs.py: Update MicrospineCfg instantiation with new parameters
