# Stanford Climbing Robot & PHYBOT MJLab

MJLab-based reinforcement learning environments for PHYBOT C2, using MuJoCo
Warp GPU simulation.

The main task is `Mjlab-Climb-Wall-Phybot-C2`: PHYBot C2 climbs with four
microspine pads (`left/right_hand_spine`, `left/right_foot_spine`), each using
all 12 CAD units, on a wall-angle curriculum ending at 90 degrees.
`Mjlab-Climb-Wall-Phybot-C2-Corners-A60` uses the four corner units per pad
and a curriculum ending at 60 degrees. Both use the empirical unit model:
ordinary contacts supply compression, equality anchors supply grip, and
per-unit load limits determine release. Every active unit must reach at
least 10 N of estimated compression before a new engage.

## Registered Tasks

```bash
python scripts/list_envs.py
```

Expected project tasks:

- `Mjlab-Climb-Wall-Phybot-C2`
- `Mjlab-Climb-Wall-Phybot-C2-Corners-A60`
- `Mjlab-Climb-Wall-Phybot-C2-Corners-A80` (fixed-angle diagnostic variant)
- `Mjlab-Microspine-Rizon4`
- `Mjlab-Walk-Phybot-C2`
- `Mjlab-Microspine-Brick-Phybot-C2` (dual-hand floor pickup, carry and placement;
  `-Hold`, `-Engage`, `-Place`, `-DR`, and `-NoGrip` variants)

The brick task keeps both feet planted and uses a free rigid brick with
body-relative microspine anchors. See [the brick task guide](docs/tasks/brick/README.md)
for staged training, reference generation, evaluation, and the external-pose
deployment interface.

## Setup

```bash
conda activate env_mjlab
pip install -e .
```

For 3Dconnexion SpaceMouse teleop:

```bash
pip install -e ".[teleop]"
```

## Common Commands

Train:

```bash
python scripts/train.py Mjlab-Climb-Wall-Phybot-C2 --gpu-ids "[0, 1]"
```

Train the four-corner task with a curriculum ending at 60 degrees:

```bash
python scripts/train.py Mjlab-Climb-Wall-Phybot-C2-Corners-A60 --gpu-ids "[0, 1]"
```

This task writes to `logs/rsl_rl/phybot_c2_climb_wall_corners60`; the main
12-unit task keeps `logs/rsl_rl/phybot_c2_climb_wall`.

Train the PHYBot C2 walk task:

```bash
python scripts/train.py Mjlab-Walk-Phybot-C2 --gpu-ids "[0, 1]"
```

Run a longer headless training job:

```bash
nohup python scripts/train.py Mjlab-Climb-Wall-Phybot-C2 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 12000 \
  > train.log 2>&1 &
```

For long training over SSH, prefer `tmux` so the job keeps running if the SSH
session disconnects:

```bash
tmux new -s train
python scripts/train.py Mjlab-Walk-Phybot-C2 --gpu-ids "[0, 1]"
```

Detach from the session without stopping training:

```text
Ctrl-b then d
```

Reconnect to the training session:

```bash
tmux attach -t train
```

List active tmux sessions:

```bash
tmux ls
```

Kill the training session:

```bash
tmux kill-session -t train
```

Resume walk training from a local checkpoint:

```bash
python scripts/train.py Mjlab-Walk-Phybot-C2 \
  --gpu-ids "[0, 1]" \
  --agent.resume \
  --agent.load-run <run> \
  --agent.load-checkpoint model_<iter>.pt
```

Play with zero actions:

```bash
python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 --agent zero --num-envs 1
```

Play the walk task with zero actions:

```bash
python scripts/play.py Mjlab-Walk-Phybot-C2 --agent zero --num-envs 1
```

Run the Rizon4 microspine Cartesian teleop task:

```bash
python scripts/teleop_rizon4_microspine.py --input spacemouse
```

Run the same task over SSH with Viser browser controls. When the remote process
has no `DISPLAY`/`WAYLAND_DISPLAY`, `--viewer auto` also selects Viser:

```bash
ssh -L 8080:localhost:8080 <remote-host>
python scripts/teleop_rizon4_microspine.py --viewer viser --input viser
```

Then open `http://localhost:8080`.

If the 3Dconnexion SpaceMouse is plugged into the remote host, combine it with
Viser display and browser controls:

```bash
python scripts/teleop_rizon4_microspine.py --viewer viser --input spacemouse --spacemouse-source hid
```

If the SpaceMouse is plugged into your local computer while MuJoCo runs on the
remote host, forward it through SSH:

```bash
ssh -L 8080:localhost:8080 -L 8765:localhost:8765 <remote-host>
```

On the remote host:

```bash
python scripts/teleop_rizon4_microspine.py \
  --viewer viser \
  --input spacemouse \
  --spacemouse-source bridge
```

On the local computer:

```bash
python scripts/spacemouse_bridge.py --host 127.0.0.1 --port 8765
```

If Linux sees the SpaceMouse in `lsusb` but `hidapi` cannot open it, add USB
and hidraw udev rules and reconnect the device. The USB rule is needed when
Python `hidapi` uses the libusb backend; the hidraw rule is needed when it uses
the hidraw backend.

```bash
sudo groupadd -f plugdev
sudo usermod -aG plugdev $USER
cat <<'EOF' | sudo tee /etc/udev/rules.d/99-3dconnexion.rules
SUBSYSTEM=="usb", ATTR{idVendor}=="256f", MODE="0660", GROUP="plugdev", TAG+="uaccess"
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0660", GROUP="plugdev", TAG+="uaccess"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then unplug/replug the SpaceMouse and start a new login shell.

Keyboard fallback:

```bash
python scripts/teleop_rizon4_microspine.py --input keyboard
```

Play from a checkpoint:

```bash
python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 \
  --checkpoint-file logs/rsl_rl/<run>/model_<iter>.pt \
  --num-envs 1
```

Play walk from a checkpoint:

```bash
python scripts/play.py Mjlab-Walk-Phybot-C2 \
  --checkpoint-file logs/rsl_rl/phybot_c2_walk/<run>/model_<iter>.pt \
  --num-envs 1
```

Override the play-mode wall angle:

```bash
python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 \
  --checkpoint-file logs/rsl_rl/<run>/model_<iter>.pt \
  --wall-angle-deg 90 \
  --num-envs 1
```

Record video:

```bash
python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 \
  --checkpoint-file <path> \
  --video
```

TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl --bind_all
```

Tune the initial pose in a Viser browser UI:

```bash
python scripts/tune_pose.py
```

Inspect the C2 robot model without RL:

```bash
python -m scr_phybot_mjab.assets.phybot_c2.phybot_c2_constants
```

## Climb-Wall Task

The two curriculum tasks share the environment implementation, PPO settings,
rewards, reference-state initialization, observations and PD control:

| Task suffix | Active units per pad | Training wall angles | Default evaluation |
| --- | ---: | --- | --- |
| `Phybot-C2` | 12 (3×4) | 45±5 → 75±12 → 80±5 → 85±3 → 88±2 → 90° | Exact 90° |
| `Phybot-C2-Corners-A60` | 4 corners | 45±5 → 55±5 → 60° | Exact 60° |
| `Phybot-C2-Corners-A80` | 4 corners | Fixed 80° | Exact 80° |

The main task keeps the original stage boundaries at iterations 0, 1500,
3300, 3900, 4550 and 5150 (`mdp.curriculums.WALL_ANGLE_STAGES`). The four-corner
60-degree task changes stages at 0, 1500 and 3300
(`CORNERS60_WALL_ANGLE_STAGES`). Its final stage explicitly disables jitter:
training never samples above 60 degrees. Each resetting environment picks
the current stage; walls of running episodes stay stationary.

The main task uses `climb_cad_empirical.npy`; the 60-degree task uses
`climb_cad_empirical_corners_a60.npy`. Both references loop every 9 seconds
and advance 20 mm per cycle. The velocity command follows this average
along-wall rate. Training samples the full cycle with a failure-adaptive RSI
histogram (one-second bins, EMA 0.001, additive uniform floor 0.1). Reset
restores reference pose, velocity, passive gimbals and scheduled initial
grips. Physical anchors are installed before the first physics step;
zero-time resets do not advance the reference or engage debounce.
Evaluation starts at frame 0 with no additional reset jitter.

Empirical tasks now use 16 rewards: four BeyondMimic body kernels, root
orientation, height/landing/swing tracking, four contact terms, three
regularizers, and a failure penalty. Existing position/orientation/climbing
scales and low-shear/lateral release shaping are retained. The mismatch gate
applies to body position, body orientation and root orientation. There is no
explicit survival reward. Legacy Wang transition rewards are unchanged.

See the [reward/reset review and verification results](docs/reports/empirical_climb/beyondmimic_2026-09-07/README.md)
before training. Reset/reward/termination integration passes GPU checks, but
reference-position PD still loses grips rapidly and has not demonstrated a
successful climb or regrip. The reference contains kinematics, not force or
feedforward-torque targets. No training was started for this review.

Gravity is always physically down. The curriculum TILTS THE WALL about its
base line (per env, mocap pose): shallow angles are a ramp the robot lies on,
90 degrees is standard vertical-wall climbing — and exactly the world the
climb-transition (mount) task runs in, so the two tasks chain.

Terminations:

- `time_out`
- `base_height`: base fell below `0.3` m of along-wall altitude
- `bad_state`: NaN or excessive velocity guard
- `bad_tracking`: any tracked body's aligned position error exceeds `0.3` m,
  global root orientation error exceeds `1.0` rad, or along-wall root height
  error exceeds `0.25` m
- `lost_support`: all pads remain unlatched continuously for `0.4` s

## Walk Task

Command:

- Task id: `Mjlab-Walk-Phybot-C2`
- Command term: `twist`
- Resampling interval: `3.0` to `8.0` s
- Training ranges:
  - `lin_vel_x`: `(-1.0, 1.0)` m/s
  - `lin_vel_y`: `(-1.0, 1.0)` m/s
  - `ang_vel_z`: `(-0.5, 0.5)` rad/s
  - `heading`: `(-pi, pi)` rad
- Heading command is enabled with heading-control stiffness `0.5`.
- Standing-env ratio is `0.1`; heading-env ratio is `0.3`.
- Flat-terrain play mode widens `lin_vel_x` to `(-1.5, 2.0)` m/s and
  `ang_vel_z` to `(-0.7, 0.7)` rad/s.

The actor observes the generated `twist` command, base sensors from the C2 XML,
policy joint position/velocity, and last action. Contact sensors are used for
critic terms, rewards, and self-collision termination; they are not actor
observations.

## Project Layout

```text
scripts/
  train.py              # Register project tasks, then delegate to MJLab train.
  play.py               # Play/evaluate with climb-wall angle controls.
  tune_pose.py          # Viser-based pose tuning.
  list_envs.py          # List registered tasks.

src/scr_phybot_mjab/
  assets/
    motions/           # Runtime motion clips and shared clip/initial-state paths.
    microspine/        # Packaged empirical unit distribution; no raw bench logs.
    phybot_c2/          # PHYBot C2 MJCF, meshes, actuator config.
    rizon4/             # Flexiv Rizon4 URDF/meshes and microspine pad config.
    wall/               # Wall entity.
  microspine/
    config.py           # Microspine config and affordance command cfg.
    affordance.py       # Contact-sensor engage/disengage state machine.
    contact.py          # Contact-grid names and ordinary wall contact pairs.
    constraint.py       # Runtime equality anchors and load readout.
    pad_geometry.py     # C/unit geometry read from asset sites.
    wrench.py           # Shared Torch/NumPy per-unit wrench mechanics.
    empirical_limit_surface.py  # Conditional-distribution lookup and unit limits.
  tools/
    climbing_reference/ # Public offline IK, cycle synthesis and audit model tools.
    microspine/         # Fitting and stochastic pad analysis using runtime mechanics.
    plotting.py         # Shared figure style/export helpers.
  tasks/
    climb_wall/         # C2 wall-climbing task.
    microspine_rizon4/  # Rizon4 Cartesian-control microspine test task.
    walk/               # C2 walking task.
```

Raw microspine trials live in `data/microspine/raw/` (not in git); the fit
report and calibration/baseline tables live in `data/microspine/empirical_unit_v1/`;
generated analyses go to the ignored `artifacts/`.
See [the empirical climbing code layout](docs/reports/empirical_climb/code_layout.md)
for active resource paths, shared computations, and regeneration commands.
