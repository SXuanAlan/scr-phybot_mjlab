# Spider RL MJLab

A wall-climbing reinforcement learning environment for the PHYBot C2 humanoid robot, built on [mjlab](https://mujocolab.github.io/mjlab/main/index.html) — a lightweight, GPU-accelerated robot learning framework using MuJoCo Warp physics.

Migrated from an [IsaacLab](https://isaac-sim.github.io/IsaacLab/)-based implementation (`spider_rl_lab`) to leverage mjlab's native MuJoCo integration and manager-based RL environment API.

## Project Structure

```
spider_rl_mjlab/
├── pyproject.toml                              # Package metadata, dependencies, entry points
├── scripts/
│   ├── train.py                                # Train an RL agent
│   ├── play.py                                 # Evaluate / visualize a trained agent
│   └── list_envs.py                            # List all registered environments
└── spider_rl_mjlab/                            # Python package
    ├── __init__.py
    ├── assets/
    │   ├── __init__.py                         # Re-exports: get_phybot_c2_robot_cfg, ACTION_SCALE
    │   └── phybot_c2/
    │       ├── __init__.py                     # Re-exports from phybot_c2_constants
    │       ├── phybot_c2_constants.py          # Robot definition: actuators, keyframe, collision
    │       └── xmls/
    │           ├── phybot_c2.xml               # Cleaned MJCF model (no scene elements)
    │           └── assets/                     # 25 STL mesh files
    ├── microspine/
    │   ├── __init__.py                         # Re-exports: MicrospineController, MicrospineCfg
    │   ├── config.py                           # MicrospineCfg dataclass (spine mechanics, surface params)
    │   ├── actuators.py                        # add_adhesion_actuators() — adds MuJoCo adhesion actuators
    │   ├── physics.py                          # Jiang et al. 2018 equations (force, engagement, activation)
    │   └── controller.py                       # MicrospineController — stochastic step-mode event
    └── tasks/
        ├── __init__.py                         # Imports climb_wall → triggers task registration
        └── climb_wall/
            ├── __init__.py                     # Imports config.phybot_c2 → triggers registration
            ├── climb_wall_env_cfg.py           # Task factory: observations, rewards, scene, etc.
            ├── mdp/
            │   ├── __init__.py                 # Re-exports custom MDP functions
            │   ├── observations.py             # Microspine privileged observations
            │   └── rewards.py                  # Custom reward functions (incl. adhesion rewards)
            └── config/
                ├── __init__.py
                └── phybot_c2/
                    ├── __init__.py             # Calls register_mjlab_task()
                    ├── env_cfgs.py             # Robot-specific env config
                    └── rl_cfg.py               # PPO hyperparameters
```

## How It Works

### Task Registration

mjlab discovers tasks via the `mjlab.tasks` entry point declared in `pyproject.toml`:

```toml
[project.entry-points."mjlab.tasks"]
spider_rl_mjlab = "spider_rl_mjlab.tasks"
```

When mjlab (or our scripts) imports `spider_rl_mjlab.tasks`, the following import chain fires:

```
spider_rl_mjlab.tasks
  → .climb_wall
    → .config.phybot_c2
      → register_mjlab_task("Mjlab-Climb-Wall-Phybot-C2", ...)
```

This registers the task ID so that `python scripts/train.py Mjlab-Climb-Wall-Phybot-C2` just works.

### Manager-Based Environment

mjlab uses a manager-based architecture where the environment is configured via a `ManagerBasedRlEnvCfg` containing plain Python dicts for each manager:

| Manager | Purpose | Configuration |
|---------|---------|---------------|
| **Observation** | Collects observations into flat tensors | policy (72-dim) + privileged (20-dim) |
| **Action** | Applies joint position targets | 21-dim (all actuated joints) |
| **Reward** | Computes per-step rewards | 9 weighted terms (incl. adhesion rewards) |
| **Termination** | Checks episode-ending conditions | timeout + fall detection |
| **Event** | Handles resets + microspine adhesion | step: microspine controller; reset: scene/joint/base |

### Parallel Simulation (nworld)

mjlab uses MuJoCo Warp's `nworld` mechanism: one compiled `MjModel` is simulated N times in parallel with independent state (`qpos`, `qvel`, contacts). Each "world" is a self-contained environment instance.

Key consequence: static bodies (like walls) must exist in the shared model at the correct per-env positions. The env config achieves this by computing a grid of `env_origins` and placing one wall body per env in the `MjSpec` before compilation (see `_make_wall_spec_fn`).

---

## File Reference

### `pyproject.toml`

Package configuration. Key sections:

- **Dependencies**: requires `mjlab`
- **Entry points**: `mjlab.tasks` → enables auto-discovery of tasks
- **Console scripts**: `spider-train`, `spider-play`, `spider-list-envs` → delegate to mjlab's CLI

### Scripts

All scripts follow the same pattern: import `spider_rl_mjlab.tasks` (to trigger task registration), then delegate to a mjlab script's `main()`. mjlab uses [tyro](https://brentyi.github.io/tyro/) for CLI argument parsing.

| Script | Delegates to | Purpose |
|--------|-------------|---------|
| `scripts/train.py` | `mjlab.scripts.train:main` | Train with PPO via RSL-RL |
| `scripts/play.py` | `mjlab.scripts.play:main` | Run zero / random / trained agent |
| `scripts/list_envs.py` | `mjlab.scripts.list_envs:main` | Print registered task IDs |

### `assets/phybot_c2/phybot_c2_constants.py`

Defines the PHYBot C2 robot for mjlab. This is the central robot configuration file.

**MJCF loading:**

- `PHYBOT_C2_XML` — path to `xmls/phybot_c2.xml`
- `get_assets(meshdir)` — loads STL mesh files into a `dict[str, bytes]` via `mjlab.utils.os.update_assets`
- `get_spec()` — returns a `mujoco.MjSpec` loaded from the XML with mesh assets and 4 adhesion actuators (one per end-effector) attached

**Actuator configuration** — 11 `BuiltinPositionActuatorCfg` instances, one per unique gain group:

| Config | Joints Matched | Stiffness | Damping | Effort Limit |
|--------|---------------|-----------|---------|--------------|
| `HIP_PITCH` | `.*_hip_pitch` | 100 | 10 | 200 |
| `HIP_ROLL` | `.*_hip_roll` | 100 | 10 | 80 |
| `HIP_YAW` | `.*_hip_yaw` | 100 | 10 | 58 |
| `KNEE` | `.*_knee` | 100 | 10 | 80 |
| `FEET` | `.*_ankle_pitch`, `.*_ankle_roll` | 50 | 5 | 58 |
| `WAIST` | `waist_yaw` | 100 | 10 | 58 |
| `SHOULDER_PITCH` | `.*_shoulder_pitch` | 50 | 5 | 58 |
| `SHOULDER_ROLL` | `.*_shoulder_roll` | 50 | 5 | 58 |
| `SHOULDER_YAW` | `.*_shoulder_yaw` | 5 | 5 | 58 |
| `LEFT_ELBOW` | `left_elbow_pitch` | 50 | 5 | 9 |
| `RIGHT_ELBOW` | `right_elbow_pitch` | 50 | 5 | 58 |

Each config creates MuJoCo position actuators via `BuiltinPositionActuatorCfg(target_names_expr=(...), stiffness=..., damping=..., effort_limit=...)`. The `target_names_expr` are regex patterns matched against joint names.

**Adhesion actuators** — `get_spec()` also adds 4 MuJoCo adhesion actuators (one per end-effector: left/right ankle_roll, left/right elbow_pitch) via `add_adhesion_actuators()`. These are constraint-based adhesion actuators with gain=50N and ctrl range [0,1], controlled by the `MicrospineController` step event at runtime. Total `model.nu = 25` (21 joint + 4 adhesion).

**Initial state** — `CLIMB_WALL_KEYFRAME` sets:
- Root position: `(0.0, 0.0, 0.66)` (standing height)
- 21 joint positions for a slightly crouched stance (hips bent, knees flexed, elbows slightly bent)

**Collision** — `PHYBOT_C2_COLLISION` applies `condim=3` and friction `(1.0, 0.1, 0.1)` to all geoms matching `.*_collision.*`.

**Final assembly** — `get_phybot_c2_robot_cfg()` returns an `EntityCfg` combining:
- `spec_fn=get_spec` (MJCF loader)
- `init_state=CLIMB_WALL_KEYFRAME`
- `collisions=(PHYBOT_C2_COLLISION,)`
- `articulation=PHYBOT_C2_ARTICULATION` (all 11 actuator configs + `soft_joint_pos_limit_factor=0.9`)

### `assets/phybot_c2/xmls/phybot_c2.xml`

Cleaned MJCF model containing only the robot's kinematic tree. Stripped of scene-level elements (lights, ground, skybox, actuators, sensors) since mjlab manages these programmatically.

Key structure:
- **Defaults**: `phybot_c2` class with sub-classes `visual` (group 1, no collision) and `collision` (group 3, hidden)
- **Assets**: 25 mesh references for body part STL files
- **Body tree**: `base_link` with free joint → legs (6 DOF each) + waist → torso → neck + arms (4 DOF each)
- **21 actuated joints**: 12 leg + 2 ankle per leg + 1 waist + 4 arm joints
- **Collision geoms**: All tagged `class="collision"` → `group=3` (not visible in default rendering)

### `tasks/climb_wall/climb_wall_env_cfg.py`

Factory function `make_climb_wall_env_cfg()` that creates the base task configuration. This is robot-agnostic — the robot entity is injected later by robot-specific configs.

**Scene setup:**

The scene uses `TerrainEntityCfg(terrain_type="plane")` for a flat ground plane with checker texture, plus `SceneCfg.spec_fn` to add per-env walls:

```python
spec_fn=_make_wall_spec_fn(num_envs=128, env_spacing=4.0)
```

`_make_wall_spec_fn` computes the same grid layout as `TerrainEntity._compute_env_origins_grid` and places a static box body (no joints) at each `env_origin + (2.0, 0.0, 2.5)`. The wall inner face is at `x = env_origin_x + 1.5`, 1.5m from the robot's starting position.

**Observations:**

*Policy group* (72-dim concatenated vector):

| Term | Function | Dims | Description |
|------|----------|------|-------------|
| `base_lin_vel` | `envs_mdp.base_lin_vel` | 3 | Body-frame linear velocity |
| `base_ang_vel` | `envs_mdp.base_ang_vel` | 3 | Body-frame angular velocity |
| `projected_gravity` | `envs_mdp.projected_gravity` | 3 | Gravity vector in body frame |
| `joint_pos_rel` | `envs_mdp.joint_pos_rel` | 21 | Joint positions relative to default |
| `joint_vel_rel` | `envs_mdp.joint_vel_rel` | 21 | Joint velocities |
| `last_action` | `envs_mdp.last_action` | 21 | Previous action |

*Privileged group* (20-dim, for RMA training):

| Term | Function | Dims | Description |
|------|----------|------|-------------|
| `microspine_state` | `mdp.microspine_privileged_obs` | 20 | Per-EE: engaged_ratio, shear/normal capacity, contact, surface_quality (4 EEs x 5) |

**Actions** (21-dim):

`JointPositionActionCfg` with `use_default_offset=True` — actions are deltas from the keyframe joint positions, scaled by `PHYBOT_C2_ACTION_SCALE = 0.25`.

**Rewards** (9 terms):

| Term | Function | Weight | Purpose |
|------|----------|--------|---------|
| `base_height` | `mdp.base_height` | +5.0 | Climb higher |
| `torso_orientation` | `mdp.torso_orientation` | +40.0 | Stay upright |
| `microspine_adhesion` | `mdp.microspine_adhesion_reward` | +10.0 | Maximize adhesion force capacity |
| `spine_engagement` | `mdp.spine_engagement_reward` | +5.0 | Engage spines on wall |
| `torso_ang_vel_xy` | `mdp.torso_ang_vel_xy` | -2.0 | Don't shake |
| `lin_vel_xy` | `mdp.lin_vel_xy` | -2.0 | Don't drift laterally |
| `action_rate` | `envs_mdp.action_rate_l2` | -0.6 | Smooth actions |
| `joint_torques` | `envs_mdp.joint_torques_l2` | -1e-6 | Energy efficiency |
| `dof_pos_limits` | `envs_mdp.joint_pos_limits` | -10.0 | Stay in joint limits |

**Events:**

| Event | Function | Mode | Purpose |
|-------|----------|------|---------|
| `microspine_adhesion` | `MicrospineController` | step | Stochastic spine engagement + adhesion ctrl writes |
| `reset_scene_to_default` | `envs_mdp.reset_scene_to_default` | reset | Reset all entities to keyframe + env_origins |
| `reset_robot_joints` | `envs_mdp.reset_joints_by_offset` | reset | Add small velocity noise to joints |
| `reset_robot_base` | `envs_mdp.reset_root_state_uniform` | reset | Randomize y-position ±0.3m |

**Terminations:**

| Term | Function | Condition |
|------|----------|-----------|
| `time_out` | `envs_mdp.time_out` | Episode exceeds 10s |
| `base_height` | `envs_mdp.root_height_below_minimum` | Robot z < 0.3m (fallen) |

**Simulation**: timestep=0.002s, decimation=10 → 50 Hz control frequency.

### `microspine/`

Physics-based microspine adhesion model for wall climbing, combining MuJoCo's native adhesion actuators with the stochastic spine-asperity interaction model from Jiang et al. 2018.

**Architecture:** Two layers work together:

1. **MuJoCo adhesion actuators** (4 total, one per EE) — constraint-based adhesion solved within the contact solver (zero delay). Added to the MjSpec at model build time via `add_adhesion_actuators()`. The `ctrl` signal (0-1) controls adhesion activation strength (gain=50N max per EE).

2. **Stochastic microspine controller** (`MicrospineController`, step-mode event) — implements the Jiang et al. model to compute per-EE adhesion activation based on spine engagement state, surface properties, and loading. Writes computed activation to `data.ctrl` for the adhesion actuators each env step.

**Key files:**

- `config.py` — `MicrospineCfg` dataclass: spine mechanics (n_spines=20, F_max=2.5N, d_stroke=6mm), surface asperity parameters (alpha, lambda, eta) with domain randomization ranges, adhesion gain (50N), contact threshold (5cm)
- `actuators.py` — `add_adhesion_actuators(spec, cfg)`: adds `<adhesion>` actuators to the MjSpec for each EE body (left/right ankle_roll and elbow_pitch)
- `physics.py` — Pure functions implementing Jiang et al. equations: `compute_spine_force` (Eq. 5), `compute_engagement_probability` (Eq. 3), `compute_adhesion_activation` (maps engagement state to ctrl [0,1])
- `controller.py` — `MicrospineController`: stateful step event that determines wall contact per EE, runs stochastic Bernoulli engagement/disengagement per spine, computes adhesion activation, writes to `data.ctrl`, and caches 20-dim privileged observations for future RMA training

**Privileged observations** (20-dim = 4 EEs x 5 values per EE):

| Index | Value | Description |
|-------|-------|-------------|
| 0 | `engaged_ratio` | Fraction of spines engaged (0-1) |
| 1 | `shear_capacity` | Normalized adhesion strength |
| 2 | `normal_capacity` | Normal force capacity (activation x preload_ratio) |
| 3 | `is_in_contact` | Binary wall contact flag |
| 4 | `surface_quality` | Normalized f(eta, lambda) |

### `tasks/climb_wall/mdp/rewards.py`

Custom reward functions for wall climbing. Includes both robot-state rewards and microspine-aware rewards. All follow the signature:

```python
def reward_fn(env: ManagerBasedRlEnv, ...) -> torch.Tensor:
```

Microspine rewards (`microspine_adhesion_reward`, `spine_engagement_reward`) access the controller via `env.event_manager.get_term_cfg("microspine_adhesion").func` to read privileged observations.

Key mjlab entity data fields used (vs IsaacLab equivalents):

| mjlab | IsaacLab | Description |
|-------|----------|-------------|
| `entity.data.root_link_pos_w` | `asset.data.root_pos_w` | Root body world position |
| `entity.data.root_link_lin_vel_w` | `asset.data.root_lin_vel_w` | Root body world linear velocity |
| `entity.data.root_link_ang_vel_w` | `asset.data.root_ang_vel_w` | Root body world angular velocity |
| `entity.data.projected_gravity_b` | `asset.data.projected_gravity_b` | Gravity in body frame (same name) |
| `env.scene.env_origins` | `env.scene.env_origins` | Per-env origin offsets (same name) |

### `tasks/climb_wall/config/phybot_c2/env_cfgs.py`

`phybot_c2_climb_wall_env_cfg(play=False)` — creates the base climb wall config and injects the PHYBot C2 robot:

```python
cfg.scene.entities = {"robot": get_phybot_c2_robot_cfg(), "wall": get_wall_cfg()}
cfg.actions["joint_pos"].scale = PHYBOT_C2_ACTION_SCALE  # 0.25
```

In play mode, sets episode length to infinity and disables observation corruption.

### `tasks/climb_wall/config/phybot_c2/rl_cfg.py`

`phybot_c2_ppo_runner_cfg()` — PPO training hyperparameters using RSL-RL:

- **Networks**: actor and critic both use `[256, 128, 64]` hidden dims with ELU activation and observation normalization
- **Actor distribution**: Gaussian with learnable scalar std (init 1.0)
- **PPO**: lr=3e-4 (adaptive schedule), gamma=0.99, lambda=0.95, clip=0.2, entropy=0.003
- **Training**: 24 steps/env, 5 epochs, 4 mini-batches, up to 100k iterations

### `tasks/climb_wall/config/phybot_c2/__init__.py`

Registers the task with mjlab's task registry:

```python
register_mjlab_task(
    task_id="Mjlab-Climb-Wall-Phybot-C2",
    env_cfg=phybot_c2_climb_wall_env_cfg(),
    play_env_cfg=phybot_c2_climb_wall_env_cfg(play=True),
    rl_cfg=phybot_c2_ppo_runner_cfg(),
)
```

---

## mjlab API Reference

### Core Concepts

| Concept | mjlab | IsaacLab Equivalent |
|---------|-------|-------------------|
| Environment | `ManagerBasedRlEnv` | `ManagerBasedRLEnv` |
| Robot / object | `Entity` | `Articulation` |
| Robot config | `EntityCfg` | `ArticulationCfg` |
| Scene | `SceneCfg` (plain dataclass) | `InteractiveSceneCfg` (`@configclass`) |
| Actuators | `BuiltinPositionActuatorCfg` | `ImplicitActuatorCfg` |
| Physics | `MujocoCfg` (MuJoCo Warp) | PhysX |
| Config style | Plain Python dicts | `@configclass` nested dataclasses |
| Asset format | MJCF (`.xml`) | USD / URDF |
| Action field | `entity_name` | `asset_name` |
| Action joints | `actuator_names` | `joint_names` |

### Key API Classes

**`EntityCfg`** — Robot/object configuration:
- `spec_fn: Callable[[], MjSpec]` — returns a loaded MJCF spec
- `init_state: InitialStateCfg` — default root pose + joint positions
- `articulation: EntityArticulationInfoCfg` — actuator configs + joint limits
- `collisions: tuple[CollisionCfg, ...]` — collision geom properties

**`BuiltinPositionActuatorCfg`** — Creates MuJoCo position actuators:
- `target_names_expr: tuple[str, ...]` — regex patterns to match joint names
- `stiffness: float` — PD position gain (must be scalar, not dict)
- `damping: float` — PD velocity gain
- `effort_limit: float` — maximum actuator force

**`SceneCfg`** — Scene configuration:
- `num_envs: int` — number of parallel environments
- `env_spacing: float` — grid spacing between env origins
- `terrain: TerrainEntityCfg` — ground plane / procedural terrain
- `entities: dict[str, EntityCfg]` — named entities to add
- `spec_fn: Callable[[MjSpec], None]` — modify spec after entities are added (used for walls)

**`ManagerBasedRlEnvCfg`** — Full environment config:
- `scene: SceneCfg`
- `sim: SimulationCfg`
- `observations: dict` — observation group configs
- `actions: dict` — action term configs
- `rewards: dict` — reward term configs
- `events: dict` — reset/interval event configs
- `terminations: dict` — termination condition configs
- `decimation: int` — physics steps per control step
- `episode_length_s: float` — episode timeout in seconds

---

## Usage

### Installation

```bash
conda activate env_mjlab
cd /home/sxuan/projects/spider_rl_mjlab
pip install -e .
```

### Training

```bash
# Default training (128 envs, 100k iterations)
python scripts/train.py Mjlab-Climb-Wall-Phybot-V8

# Custom settings
python scripts/train.py Mjlab-Climb-Wall-Phybot-V8 \
  --env.scene.num-envs 256 \
  --agent.max-iterations 50000

# Multi-GPU
python scripts/train.py Mjlab-Climb-Wall-Phybot-V8 --gpu-ids "[0, 1]"
```

### Evaluation

```bash
# Zero agent (all-zeros action)
python scripts/play.py Mjlab-Climb-Wall-Phybot-V8 --agent zero --num-envs 4

# Random agent
python scripts/play.py Mjlab-Climb-Wall-Phybot-V8 --agent random --num-envs 4

# Trained policy
python scripts/play.py Mjlab-Climb-Wall-Phybot-V8 \
  --checkpoint-file logs/rsl_rl/phybot_climb_wall/*/model_*.pt

# With video recording
python scripts/play.py Mjlab-Climb-Wall-Phybot-V8 \
  --checkpoint-file path/model.pt --video
```

### List Registered Tasks

```bash
python scripts/list_envs.py
```

### Standalone Robot Viewer

```bash
python -m spider_rl_mjlab.assets.phybot_c2.phybot_c2_constants
```

---

## Adding a New Robot

To add a different robot to the wall-climbing task:

1. **Create the robot asset** in `spider_rl_mjlab/assets/<robot_name>/`:
   - Clean MJCF XML (strip scene elements, actuators, sensors)
   - `<robot>_constants.py` with `EntityCfg`, actuators, keyframe, collision
   - Export `get_<robot>_robot_cfg()` and action scale

2. **Create robot-specific config** in `tasks/climb_wall/config/<robot_name>/`:
   - `env_cfgs.py` — inject robot entity into base config
   - `rl_cfg.py` — PPO hyperparameters (may differ per robot)
   - `__init__.py` — call `register_mjlab_task()` with a new task ID

3. **Wire the imports** in `tasks/climb_wall/__init__.py` to trigger registration.
