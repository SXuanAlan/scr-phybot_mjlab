# Spider RL MJLab

A wall-climbing reinforcement learning environment for the PHYBot Big V2 humanoid robot, built on [mjlab](https://mujocolab.github.io/mjlab/main/index.html) — a lightweight, GPU-accelerated robot learning framework using MuJoCo Warp physics.

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
    │   ├── __init__.py                         # Re-exports: get_phybot_big_v2_robot_cfg, ACTION_SCALE
    │   ├── phybot_big_v2/
    │   │   ├── __init__.py                     # Re-exports from phybot_big_v2_constants
    │   │   ├── phybot_big_v2_constants.py      # Robot definition: actuators, keyframe, collision
    │   │   ├── xml/
    │   │   │   └── phybot_big_v2_24dof.xml     # MuJoCo MJCF model (24 DOF)
    │   │   └── meshes/                         # 30 STL mesh files
    │   └── wall/
    │       ├── __init__.py
    │       └── wall_constants.py               # Wall asset configuration
    └── tasks/
        ├── __init__.py                         # Imports climb_wall → triggers task registration
        └── climb_wall/
            ├── __init__.py                     # Imports config.phybot_big_v2 → triggers registration
            ├── climb_wall_env_cfg.py           # Task factory: observations, rewards, scene, etc.
            ├── mdp/
            │   ├── __init__.py                 # Re-exports custom MDP functions
            │   ├── observations.py             # Custom observations
            │   └── rewards.py                  # Custom reward functions
            └── config/
                ├── __init__.py
                └── phybot_big_v2/
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
    → .config.phybot_big_v2
      → register_mjlab_task("Mjlab-Climb-Wall-Phybot-Big-V2", ...)
```

This registers the task ID so that `python scripts/train.py Mjlab-Climb-Wall-Phybot-Big-V2` just works.

### Manager-Based Environment

mjlab uses a manager-based architecture where the environment is configured via a `ManagerBasedRlEnvCfg` containing plain Python dicts for each manager:

| Manager | Purpose | Configuration |
|---------|---------|---------------|
| **Observation** | Collects observations into flat tensors | policy (concatenated vector) |
| **Action** | Applies joint position targets | 24-dim (all actuated joints) |
| **Reward** | Computes per-step rewards | 7 weighted terms |
| **Termination** | Checks episode-ending conditions | timeout + fall detection |
| **Event** | Handles resets | reset: scene/joint/base |

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

### `assets/phybot_big_v2/phybot_big_v2_constants.py`

Defines the PHYBot Big V2 robot for mjlab. This is the central robot configuration file.

**MJCF loading:**

- `PHYBOT_BIG_V2_XML` — path to `xml/phybot_big_v2_24dof.xml`
- `get_assets(meshdir)` — loads STL mesh files into a `dict[str, bytes]` via `mjlab.utils.os.update_assets`
- `get_spec()` — returns a `mujoco.MjSpec` loaded from the XML with mesh assets

**Actuator configuration** — 13 `BuiltinPositionActuatorCfg` instances, one per unique gain group:

| Config | Joints Matched | Stiffness | Damping | Effort Limit |
|--------|---------------|-----------|---------|--------------|
| `HIP_PITCH` | `.*_hip_pitch` | 100 | 10 | 600 |
| `HIP_ROLL` | `.*_hip_roll` | 100 | 10 | 200 |
| `HIP_YAW` | `.*_hip_yaw` | 100 | 10 | 250 |
| `KNEE` | `.*_knee` | 100 | 10 | 600 |
| `ANKLE_PITCH` | `.*_ankle_pitch` | 50 | 5 | 200 |
| `ANKLE_ROLL` | `.*_ankle_roll` | 50 | 5 | 89 |
| `WAIST_YAW` | `waist_yaw` | 100 | 10 | 250 |
| `WAIST_ROLL` | `waist_roll` | 100 | 10 | 200 |
| `SHOULDER_PITCH` | `.*_shoulder_pitch` | 50 | 5 | 200 |
| `SHOULDER_ROLL` | `.*_shoulder_roll` | 50 | 5 | 135 |
| `SHOULDER_YAW` | `.*_shoulder_yaw` | 5 | 5 | 89 |
| `ELBOW_PITCH` | `.*_elbow_pitch` | 50 | 5 | 89 |
| `ELBOW_YAW` | `.*_elbow_yaw` | 50 | 5 | 58 |

Each config creates MuJoCo position actuators via `BuiltinPositionActuatorCfg(target_names_expr=(...), stiffness=..., damping=..., effort_limit=...)`. The `target_names_expr` are regex patterns matched against joint names.

**Initial state** — `CLIMB_WALL_KEYFRAME` sets:
- Root position: `(0.0, 0.0, 1.1)` (standing height)
- 24 joint positions for a slightly crouched stance (hips bent, knees flexed)

**Collision** — `PHYBOT_BIG_V2_COLLISION` applies `condim=3` and friction `(1.0, 0.1, 0.1)` to all geoms matching `.*_collision.*`.

**Final assembly** — `get_phybot_big_v2_robot_cfg()` returns an `EntityCfg` combining:
- `spec_fn=get_spec` (MJCF loader)
- `init_state=CLIMB_WALL_KEYFRAME`
- `collisions=(PHYBOT_BIG_V2_COLLISION,)`
- `articulation=PHYBOT_BIG_V2_ARTICULATION` (all 13 actuator configs + `soft_joint_pos_limit_factor=0.9`)

### `assets/phybot_big_v2/xml/phybot_big_v2_24dof.xml`

MuJoCo MJCF model for the PHYBot Big V2 24-DOF humanoid.

Key structure:
- **Defaults**: `phybot` class with sub-classes `visual` (group 2, no collision) and `collision` (group 3, hidden)
- **Assets**: 30 mesh references for body part STL files
- **Body tree**: `base_link` with free joint → legs (6 DOF each) + waist (2 DOF) → torso → head (fixed) + arms (5 DOF each)
- **24 actuated joints**: 12 leg + 2 waist + 10 arm joints (no hand or head actuation)
- **Collision geoms**: All tagged `class="collision"` → `group=3` (not visible in default rendering)

### `tasks/climb_wall/climb_wall_env_cfg.py`

Factory function `make_climb_wall_env_cfg()` that creates the base task configuration. This is robot-agnostic — the robot entity is injected later by robot-specific configs.

**Scene setup:**

The scene uses `TerrainEntityCfg(terrain_type="plane")` for a flat ground plane with checker texture.

**Observations:**

*Policy group* (concatenated vector):

| Term | Function | Dims | Description |
|------|----------|------|-------------|
| `base_lin_vel` | `envs_mdp.base_lin_vel` | 3 | Body-frame linear velocity |
| `base_ang_vel` | `envs_mdp.base_ang_vel` | 3 | Body-frame angular velocity |
| `projected_gravity` | `envs_mdp.projected_gravity` | 3 | Gravity vector in body frame |
| `joint_pos_rel` | `envs_mdp.joint_pos_rel` | 24 | Joint positions relative to default |
| `joint_vel_rel` | `envs_mdp.joint_vel_rel` | 24 | Joint velocities |
| `last_action` | `envs_mdp.last_action` | 24 | Previous action |

**Actions** (24-dim):

`JointPositionActionCfg` with `use_default_offset=True` — actions are deltas from the keyframe joint positions, scaled by `PHYBOT_BIG_V2_ACTION_SCALE = 0.25`.

**Rewards** (7 terms):

| Term | Function | Weight | Purpose |
|------|----------|--------|---------|
| `base_height` | `mdp.base_height` | +5.0 | Climb higher |
| `torso_orientation` | `mdp.torso_orientation` | +40.0 | Stay upright |
| `torso_ang_vel_xy` | `mdp.torso_ang_vel_xy` | -2.0 | Don't shake |
| `lin_vel_xy` | `mdp.lin_vel_xy` | -2.0 | Don't drift laterally |
| `action_rate` | `envs_mdp.action_rate_l2` | -0.6 | Smooth actions |
| `joint_torques` | `envs_mdp.joint_torques_l2` | -1e-6 | Energy efficiency |
| `dof_pos_limits` | `envs_mdp.joint_pos_limits` | -10.0 | Stay in joint limits |

**Events:**

| Event | Function | Mode | Purpose |
|-------|----------|------|---------|
| `reset_scene_to_default` | `envs_mdp.reset_scene_to_default` | reset | Reset all entities to keyframe + env_origins |
| `reset_robot_joints` | `envs_mdp.reset_joints_by_offset` | reset | Add small velocity noise to joints |
| `reset_robot_base` | `envs_mdp.reset_root_state_uniform` | reset | Randomize y-position ±0.3m |

**Terminations:**

| Term | Function | Condition |
|------|----------|-----------|
| `time_out` | `envs_mdp.time_out` | Episode exceeds 10s |
| `base_height` | `envs_mdp.root_height_below_minimum` | Robot z < 0.3m (fallen) |

**Simulation**: timestep=0.002s, decimation=10 → 50 Hz control frequency.

### `tasks/climb_wall/mdp/rewards.py`

Custom reward functions for wall climbing. All follow the signature:

```python
def reward_fn(env: ManagerBasedRlEnv, ...) -> torch.Tensor:
```

Key mjlab entity data fields used (vs IsaacLab equivalents):

| mjlab | IsaacLab | Description |
|-------|----------|-------------|
| `entity.data.root_link_pos_w` | `asset.data.root_pos_w` | Root body world position |
| `entity.data.root_link_lin_vel_w` | `asset.data.root_lin_vel_w` | Root body world linear velocity |
| `entity.data.root_link_ang_vel_w` | `asset.data.root_ang_vel_w` | Root body world angular velocity |
| `entity.data.projected_gravity_b` | `asset.data.projected_gravity_b` | Gravity in body frame (same name) |
| `env.scene.env_origins` | `env.scene.env_origins` | Per-env origin offsets (same name) |

### `tasks/climb_wall/config/phybot_big_v2/env_cfgs.py`

`phybot_big_v2_climb_wall_env_cfg(play=False)` — creates the base climb wall config and injects the PHYBot Big V2 robot:

```python
cfg.scene.entities = {"robot": get_phybot_big_v2_robot_cfg(), "wall": get_wall_cfg()}
cfg.actions["joint_pos"].scale = PHYBOT_BIG_V2_ACTION_SCALE  # 0.25
```

In play mode, sets episode length to infinity and disables observation corruption.

### `tasks/climb_wall/config/phybot_big_v2/rl_cfg.py`

`phybot_big_v2_ppo_runner_cfg()` — PPO training hyperparameters using RSL-RL:

- **Networks**: actor and critic both use `[256, 128, 64]` hidden dims with ELU activation and observation normalization
- **Actor distribution**: Gaussian with learnable scalar std (init 1.0)
- **PPO**: lr=3e-4 (adaptive schedule), gamma=0.99, lambda=0.95, clip=0.2, entropy=0.003
- **Training**: 24 steps/env, 5 epochs, 4 mini-batches, up to 100k iterations

### `tasks/climb_wall/config/phybot_big_v2/__init__.py`

Registers the task with mjlab's task registry:

```python
register_mjlab_task(
    task_id="Mjlab-Climb-Wall-Phybot-Big-V2",
    env_cfg=phybot_big_v2_climb_wall_env_cfg(),
    play_env_cfg=phybot_big_v2_climb_wall_env_cfg(play=True),
    rl_cfg=phybot_big_v2_ppo_runner_cfg(),
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
python scripts/train.py Mjlab-Climb-Wall-Phybot-Big-V2

# Custom settings
python scripts/train.py Mjlab-Climb-Wall-Phybot-Big-V2 \
  --env.scene.num-envs 256 \
  --agent.max-iterations 50000

# Multi-GPU
python scripts/train.py Mjlab-Climb-Wall-Phybot-Big-V2 --gpu-ids "[0, 1]"
```

### Evaluation

```bash
# Zero agent (all-zeros action)
python scripts/play.py Mjlab-Climb-Wall-Phybot-Big-V2 --agent zero --num-envs 4

# Random agent
python scripts/play.py Mjlab-Climb-Wall-Phybot-Big-V2 --agent random --num-envs 4

# Trained policy
python scripts/play.py Mjlab-Climb-Wall-Phybot-Big-V2 \
  --checkpoint-file logs/rsl_rl/phybot_big_v2_climb_wall/*/model_*.pt

# With video recording
python scripts/play.py Mjlab-Climb-Wall-Phybot-Big-V2 \
  --checkpoint-file path/model.pt --video
```

### List Registered Tasks

```bash
python scripts/list_envs.py
```

### Standalone Robot Viewer

```bash
python -m spider_rl_mjlab.assets.phybot_big_v2.phybot_big_v2_constants
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
