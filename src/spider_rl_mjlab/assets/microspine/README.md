# Microspine Adhesion Simulation

This module simulates microspine adhesion for a wall-climbing robot. Microspines are small hooks that catch on surface asperities (microscopic bumps) to generate shear forces against a rough wall. The simulation is based on the stochastic model from **Jiang et al. 2018** ("Stochastic models of compliant spine arrays for rough surface grasping").

## High-Level Architecture

The simulation uses a **dual-force architecture** that decouples shear and normal forces, matching the paper's physics:

- **Shear** (wall-parallel) is the primary microspine force, computed from engaged spine states and applied via `xfrc_applied`
- **Normal** (pull-off resistance) is capped by the loading angle limit `F_n ≤ F_s × tan(θ_th)`, applied via MuJoCo adhesion actuator

Shear does NOT depend on friction. Both forces come directly from the stochastic spine engagement model.

```
Wall contact detected
        ↓
Stochastic spine engagement (Jiang et al. 2018)
        ↓
Per-spine forces F(x) = K·(d_stroke - x) summed over engaged spines
        ↓
┌───────────────────────┬──────────────────────────────┐
│ Shear force (primary) │ Normal force limit (derived) │
│ = Σ F(xᵢ) engaged    │ = shear × tan(θ_th)          │
│ via xfrc_applied      │ via adhesion actuator ctrl    │
└───────────────────────┴──────────────────────────────┘
```

| Component | File | Role |
|-----------|------|------|
| `MicrospineController` | `controller.py` | Detects contact, runs spine model, applies shear + normal forces |
| MuJoCo adhesion actuator | `actuators.py` | Constraint-based normal force (pull-off resistance only) |
| Physics functions | `physics.py` | Pure vectorized Jiang et al. equations |

---

## Part 1 — Force Application

### Shear force (via `xfrc_applied`)

Shear is the primary force microspines generate — spines hook into asperities and resist sliding along the wall surface. The total shear per EE equals the sum of individual spine forces:

```python
# physics.py
total_shear = Σ F(xᵢ) for all engaged spines  # (N, E) in Newtons
```

The shear direction opposes the EE's tangential velocity (when moving) or gravity's projection onto the wall plane (when stationary). This works for any wall orientation.

```python
# controller.py — applied via write_external_wrench_to_sim
shear_force_3d = shear_direction × total_shear  # (N, E, 3)
```

### Normal force (via adhesion actuator)

MuJoCo's `adhesion` actuator resists contact separation (pull-off). The normal force is limited by the loading angle constraint (Eq. 8):

```
F_normal ≤ F_shear × tan(θ_th)
```

With `θ_th ≈ 0.124 rad (7.1°)`, normal is only ~12.6% of shear — spines are much stronger in shear than in pull-off.

```python
# controller.py
normal_limit = total_shear * tan(theta_threshold)
adhesion_ctrl = normal_limit / max_adhesion_gain  # → data.ctrl ∈ [0, 1]
```

**Force limits (per EE, 20 spines at full engagement):**

| Force type | Source | Max |
|------------|--------|-----|
| Shear | `xfrc_applied` from Σ F(xᵢ) | **50 N** (20 × 2.5 N) |
| Normal | Adhesion actuator, capped by loading angle | **~6.3 N** (50 × tan(7.1°)) |

Both forces are written to `mjData` in the same step event and take effect in the **same simulation step** (zero delay).

---

## Part 2 — Stochastic Spine Physics

### State Variables

The controller maintains per-spine state for all environments in parallel:

```python
# controller.py __init__
N, E, S = self._num_envs, self._n_ee, self._n_spines   # e.g. (4096, 4, 20)

self._spine_engaged = torch.zeros(N, E, S, dtype=torch.bool)   # is spine hooked?
self._spine_travel  = torch.zeros(N, E, S, dtype=torch.float32) # travel distance x [m]
self._ee_was_in_contact = torch.zeros(N, E, dtype=torch.bool)  # contact from last step
```

Surface asperity parameters are stored per-environment (randomized on reset):

```python
self._surface_alpha  = torch.full((N,), mcfg.alpha)    # α: immediate-engagement prob
self._surface_lambda = torch.full((N,), mcfg.lambda_)  # λ: asperity density [/mm]
self._surface_eta    = torch.full((N,), mcfg.eta)      # η: strong asperity fraction
```

---

### Step 1 — Contact Detection

Each step, the controller computes the distance from every EE body to the wall face and thresholds it:

```python
# controller.py __call__
ee_pos      = self._asset.data.body_link_pos_w[:, self._ee_body_ids, :]  # (N, E, 3)
wall_face_x = env_origin_x + self._wall_x_offset                         # (N,)
dist_to_wall = (wall_face_x.unsqueeze(1) - ee_pos[:, :, 0]).abs()        # (N, E)

in_contact       = dist_to_wall < cfg.contact_threshold   # (N, E) bool
newly_in_contact = in_contact & ~self._ee_was_in_contact  # first contact this step
lost_contact     = ~in_contact & self._ee_was_in_contact  # just left the wall
```

`contact_threshold = 0.05 m` is intentionally large to account for the sim-to-real gap between simulated geometry and physical spine reach.

---

### Step 2 — New Contact: Spine Engagement Sampling

When an EE first touches the wall, each spine independently samples whether and where it catches an asperity. This implements the **two-phase asperity distribution (Eq. 2)**:

```
p(α, λ; x) = (1 - α)·δ(x)  +  α·λ·e^(-λx)
```

- `(1 - α)` of spines engage **immediately** at `x = 0` (delta term)
- `α` of spines require some travel before finding an asperity (exponential term)

After this step, each engaged spine has a sampled travel position `x ∈ [0, d_stroke]` that represents where on the stroke it caught an asperity.

---

### Step 3 — Ongoing Contact: Failure Checks

Every step, engaged spines are tested against two independent failure conditions.

#### 3a. Asperity strength failure (Eq. 3)

A spine disengages if its current force exceeds what the asperity can sustain.

**Spine force (Eq. 5)** — linear spring, maximum at `x = 0`:

```
F(x) = K · (d_stroke - x),   K = F_max / d_stroke
```

**Asperity hold probability (Eq. 3)** — piecewise linear:

```
Pr(F) = 1.0                            if F ≤ 0.5 N   (all asperities hold)
Pr(F) = η + (1-η)·(3 - F) / 2.5       if 0.5 < F < 3.0 N
Pr(F) = η                              if F ≥ 3.0 N   (only strong asperities hold)
```

#### 3b. Loading angle failure (Eq. 1 + 8) — pull-out condition

A spine disengages if the loading direction tilts too far from the wall surface:

```
tan(θ_m) = h_mis / d_unit                     (Eq. 1)
θ_th = arctan(h_mis / d_unit) ≈ 0.124 rad (7.1°)
```

Pull-out fails when (Eq. 8):

```
arctan(F_n / F_s) ≥ θ_th
```

`theta_threshold` is a **derived property** in `MicrospineCfg`:

```python
# config.py
@property
def theta_threshold(self) -> float:
    return math.atan(self.h_mis / self.d_unit)   # Eq. 1
```

This threshold also governs the **normal force cap** in force application (see Part 1).

---

### Step 4 — Lost Contact: Full Disengagement

When an EE leaves the wall, all its spines reset and both shear (`xfrc_applied`) and normal (`adhesion ctrl`) forces are zeroed.

---

### Step 5 — Force Computation and Application

```python
# controller.py __call__
# Shear = sum of engaged spine forces
total_shear = compute_total_shear_force(travel, engaged, K, d_stroke, F_max)  # (N, E)

# Normal = shear × tan(θ_th), capped by adhesion gain
normal_limit = total_shear * tan(theta_threshold)  # (N, E)

# Apply shear via xfrc_applied (wall-tangential direction)
shear_dir = compute_shear_direction(ee_vel, wall_normal, gravity_dir)
self._asset.write_external_wrench_to_sim(forces=shear_dir * total_shear, ...)

# Apply normal via adhesion actuator
self._sim_data.ctrl[:, adhesion_ctrl_ids] = (normal_limit / max_adhesion_gain).clamp(0, 1)
```

---

## Domain Randomization

Surface parameters are re-sampled per-environment on each reset to simulate wall-to-wall variability. This is the primary mechanism for sim-to-real transfer.

| Parameter | Meaning | Default | Range |
|-----------|---------|---------|-------|
| `α` | Fraction needing travel before engagement | 0.25 | [0.20, 0.31] |
| `λ` [/mm] | Asperity spatial density | 0.40 | [0.33, 0.49] |
| `η` | Strong-asperity fraction (hold at ≥ 3 N) | 0.20 | [0.027, 0.53] |

The wide range of `η` (0.027–0.53) produces surfaces ranging from very slippery to very grippy, forcing the policy to adapt through the RMA adaptation module.

---

## Privileged Observations (for RMA)

The controller caches a `(num_envs, n_ee × 5)` tensor. The teacher policy observes these directly; the student must infer them from proprioception.

```python
# controller.py _update_privileged_obs()
obs = torch.stack([
    engaged_ratio,       # n_engaged / n_spines
    normalized_shear,    # total shear / (n_spines × F_max)
    normalized_normal,   # normal limit / max_adhesion_gain
    in_contact.float(),  # 1.0 if EE touching wall
    surface_quality,     # η·λ / 0.49
], dim=-1)  # (N, E, 5)
self._privileged_obs = obs.reshape(self._num_envs, -1)
```

---

## File Structure

```
microspine/
├── config.py       — MicrospineCfg dataclass (all parameters with defaults)
├── physics.py      — Pure vectorized physics functions (Jiang et al. equations)
├── actuators.py    — Adds MuJoCo adhesion actuators to MjSpec at build time
└── controller.py   — MicrospineController: step event, ties everything together
```

## Reference

Jiang, H., Wang, S., and Cutkosky, M. R. "Stochastic models of compliant spine arrays for rough surface grasping." *The International Journal of Robotics Research*, 37(7):669–687, 2018.
