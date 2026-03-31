# MuJoCo Prior Knowledge for Microspine Tuning

Source: https://mujoco.readthedocs.io/en/stable/computation/index.html

## 1. Soft Contact Model

MuJoCo uses a **convex optimization** formulation for contacts (not LCP). All constraints are inherently soft: pushing harder against a constraint always results in larger acceleration. This means **gradual slip is unavoidable** — it's a fundamental property, not a solver limitation.

### Core constraint equation:
```
a_c + d * (b*v + k*r) = (1 - d) * a_u
```
- `a_c` = constraint acceleration
- `v` = velocity, `r` = position residual/violation
- `k` = stiffness, `b` = damping
- `d` = impedance (constraint force generation ability, range [0.0001, 0.9999])
- `a_u` = unforced acceleration

## 2. solref — Reference Acceleration

`solref` real(2), default: `0.02 1`

**Standard format** (positive values): `(timeconst, dampratio)`
- `b = 2 / (d_width * timeconst)`
- `k = d(r) / (d_width^2 * timeconst^2 * dampratio^2)`
- Penetration at rest: `r = a_u * (1-d) * timeconst^2 * dampratio^2`
- timeconst >= 2x timestep (enforced unless refsafe=false)
- dampratio=1: critical damping; <1: bouncy; >1: over-damped

**Direct format** (negative values): `(-stiffness, -damping)`
- `b = damping / d_width`
- `k = stiffness * d(r) / d_width^2`
- Penetration at rest: `r = a_u*(1-d) / stiffness`

## 3. solimp — Impedance

`solimp` real(5), default: `0.9 0.95 0.001 0.5 2`

Parameters: `(d0, d_width, width, midpoint, power)`
- `d0` = impedance at zero violation
- `d_width` = impedance at violation = width
- Sigmoidal shape controlled by midpoint and power
- Small d = weak constraint, large d = strong constraint
- **For friction constraints: always uses d0** (position violation = 0)

## 4. condim — Contact Dimensionality

- 1: normal only (frictionless)
- 3: normal + 2 tangential (standard friction)
- 4: normal + 2 tangential + torsional
- 6: full (normal + 2 tangential + torsional + 2 rolling)

Mixing rule: higher-priority condim wins; equal priority uses maximum.

## 5. Friction Parameters

Each contact has 5 friction coefficients; each geom specifies 3:
1. Tangential (replicated for both axes)
2. Torsional
3. Rolling (replicated for both axes)

Mixing: higher-priority geom's friction wins; equal priority uses element-wise max.

**Friction constraints** have zero position violation (r=0):
- Impedance always = d0
- First-order only (k=0)
- Velocity decays exponentially via damping

## 6. Reducing Slip

Recommended approach: **Newton solver + elliptic friction cones + large impratio**.

- `impratio`: global setting in mjModel.opt, influences overall constraint impedance
- Increasing impratio + first two elements of solimp reduces slip
- Requires smaller timesteps for numerical stability

**Noslip solver**: modified PGS post-processor that updates friction forces ignoring regularization. Suppresses drift but is no longer a well-defined optimization problem.

## 7. Solver Options

- **Newton**: quadratic convergence, 5-20 iterations typical. Best with tolerance ~1e-10.
- **CG**: linear convergence, faster per iteration, good for large models.
- **PGS**: sub-linear, best when DOF > constraints.
- **Friction cones**: elliptic (physically accurate) vs pyramidal (cheaper).

## 8. Contact Margin and Gap

- `margin`: distance threshold before contact activates
- `gap`: applied distance offset
- Mixing: sum of two geom values
- Contact active when r < 0 (penetration)

## 9. Python Callbacks (MuJoCo 3.6.0)

### mjcb_control
```python
mujoco.set_mjcb_control(callback_fn)
# callback_fn(model, data) -> None
```
Most commonly used callback. Writes to `data.ctrl`, `data.qfrc_applied`, `data.xfrc_applied`. Executes within `mj_forward`/`mj_step` before controls needed. **Cannot access contact forces** (computed later in pipeline).

### mjcb_contactfilter
```python
mujoco.set_mjcb_contactfilter(callback_fn)
# callback_fn(model, data, geom1, geom2) -> int
# Return 0: allow contact, 1: discard contact
```
Replaces MuJoCo's default collision filtering entirely. Processes geom pairs that passed broadphase. When installed, must manually reimplement desired default behaviors (contype/conaffinity).

## 10. Key Implications for Microspine

1. **Slip is fundamental** in MuJoCo's soft contact model — can be minimized but not eliminated
2. **High friction alone won't prevent sliding** under strong shear loads; impedance (solimp) matters more
3. **Adhesion actuators** exist natively (set_to_adhesion on MjSpec) but require ctrlrange and body transmission
4. **impratio** is the single most important global parameter for reducing slip
5. **Noslip solver** can suppress residual drift as a post-processor
6. **Contact filter callback** can implement directional contact acceptance/rejection
7. **Control callback** can inject forces to qfrc_applied/xfrc_applied but runs before contact forces are known
