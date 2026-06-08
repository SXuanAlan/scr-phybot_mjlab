# Stanford Climbing Robot & PHYBOT MJLab

MJLab-based reinforcement learning environments for PHYBOT C2, using MuJoCo
Warp GPU simulation.

The main task is `Mjlab-Climb-Wall-Phybot-C2`: a 23-DOF humanoid climbs a
vertical wall with four fixed microspine end-effectors
(`left/right_hand_spine`, `left/right_foot_spine`). Each spine has a 2x2 sphere
contact grid. Microspine adhesion is applied as external force on the sphere
bodies along the wall inward normal; engage/disengage is driven by contact
sensor force semantics.

## Registered Tasks

```bash
python scripts/list_envs.py
```

Expected project tasks:

- `Mjlab-Climb-Wall-Phybot-C2`
- `Mjlab-Microspine-Rizon4`
- `Mjlab-Walk-Phybot-C2`

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

Override the play-mode gravity curriculum angle:

```bash
python scripts/play.py Mjlab-Climb-Wall-Phybot-C2 \
  --checkpoint-file logs/rsl_rl/<run>/model_<iter>.pt \
  --gravity-angle-deg 90 \
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

Command:

- World-frame velocity command is fixed at `(0.0, 0.0, 0.08)` m/s.
- `gait_clock` is removed from the actor observation; gait is shaped by
  engagement state and rewards.

Active rewards:

| Name | Weight | Purpose |
| --- | ---: | --- |
| `track_lin_vel_xy` | `4.0` | Track zero lateral world velocity. |
| `track_lin_vel_z` | `8.0` | Track upward velocity `0.08` m/s. |
| `climb_height_progress` | `10.0` | Reward new episode-best base height. |
| `ee_grip_time` | `10.0` | Reward exactly two engaged EEs with stable grip/swing timers. |
| `ee_face_parallel_to_wall` | `2.0` | Keep every spine face aligned with the wall. |
| `ee_slide_penalty` | `-0.1` | Penalize tangential slip while engaged. |
| `ang_vel_xy_l2` | `-0.05` | Penalize base roll/pitch angular velocity. |
| `dof_torques_l2` | `-1e-5` | Penalize joint torque. |
| `dof_acc_l2` | `-1e-5` | Penalize joint acceleration. |
| `action_rate_l2` | `-0.01` | Penalize action changes. |
| `termination_penalty` | `-200.0` | Penalize non-timeout termination. |

Disabled reward terms are still present for quick experiments:
`flat_orientation_l2 = 0.0`, `dof_pos_limits = 0.0`.

Gravity curriculum is stepwise, without interpolation:

| Iteration range | Effective wall angle |
| --- | ---: |
| `0 <= iter < 3000` | `30 deg` |
| `3000 <= iter < 5000` | `60 deg` |
| `5000 <= iter` | `90 deg` |

The wall geometry stays vertical. The curriculum changes simulator gravity:
30 degrees pushes the robot mostly into the wall, while 90 degrees is standard
vertical-wall climbing.

Terminations:

- `time_out`
- `base_height`: base fell below `0.3` m
- `bad_state`: NaN or excessive velocity guard

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
  play.py               # Play/evaluate with climb-wall gravity controls.
  tune_pose.py          # Viser-based pose tuning.
  list_envs.py          # List registered tasks.
  measure_contacts.py   # Estimate contact/constraint limits.

src/scr_phybot_mjab/
  assets/
    phybot_c2/          # PHYBot C2 MJCF, meshes, actuator config.
    forearm_hand/       # Small microspine arm/hand test asset.
    rizon4/             # Flexiv Rizon4 URDF/meshes and microspine pad config.
    wall/               # Wall entity.
  microspine/
    config.py           # Microspine config and affordance command cfg.
    affordance.py       # Contact-sensor engage/disengage state machine.
    physics.py          # External sphere-body adhesion forces.
  tasks/
    climb_wall/         # C2 wall-climbing task.
    microspine_rizon4/  # Rizon4 Cartesian-control microspine test task.
    walk/               # C2 walking task.
```
