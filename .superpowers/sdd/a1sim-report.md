# A1-Sim Report: Headless MuJoCo Backend + Live CPG Walk

## Status: PASS

Robot walks successfully in headless MuJoCo simulation.

---

## 1. Joint/Sensor Mapping

### Joint Names (21, policy order) — from `deploy/phybot_deploy_c2/MujocoInterface/config/mujoco_sim.yaml`

| idx | joint name              |
|-----|-------------------------|
| 0   | left_hip_pitch          |
| 1   | left_hip_roll           |
| 2   | left_hip_yaw            |
| 3   | left_knee               |
| 4   | left_ankle_pitch        |
| 5   | left_ankle_roll         |
| 6   | right_hip_pitch         |
| 7   | right_hip_roll          |
| 8   | right_hip_yaw           |
| 9   | right_knee              |
| 10  | right_ankle_pitch       |
| 11  | right_ankle_roll        |
| 12  | waist_yaw               |
| 13  | left_shoulder_pitch     |
| 14  | left_shoulder_roll      |
| 15  | left_shoulder_yaw       |
| 16  | left_elbow_pitch        |
| 17  | right_shoulder_pitch    |
| 18  | right_shoulder_roll     |
| 19  | right_shoulder_yaw      |
| 20  | right_elbow_pitch       |

Each joint is resolved via `mj_name2id(m, mjOBJ_JOINT, name)` → `jnt_qposadr[id]` / `jnt_dofadr[id]`.
Each actuator is resolved via `mj_name2id(m, mjOBJ_ACTUATOR, name)` → actuator id used to index `d->ctrl[act_id]`.
The model has 23 actuators (2 neck actuators hold at zero; the 21 policy actuators are addressed by name).

### Sensors (4 sensors from yaml)

| sensor name          | field in DataPackage             |
|----------------------|----------------------------------|
| `baselink-quat`      | IMU quaternion → `imu_zyx`       |
| `baselink-gyro`      | `imu_angular_vel[0..2]`          |
| `baselink-velocity`  | (not used in CPG obs)            |
| `baselink-baseAcc`   | (not used in CPG obs)            |

Sensor addresses resolved via `mj_name2id(m, mjOBJ_SENSOR, name)` + `m->sensor_adr[sid]`.

---

## 2. Quaternion → ZYX Convention

MuJoCo sensor order for `baselink-quat`: `[w, x, y, z]` at offsets `[0, 1, 2, 3]`.

The `quatToZyx` function replicates `MujocoInterface::quatToZyx` exactly:

```cpp
double as = -2.0 * (x*z - w*y);
as = std::min(as, 0.99999);
as = std::max(as, -0.99999);

zyx[0] = atan2(2*(x*y + w*z),   w*w + x*x - y*y - z*z);  // yaw
zyx[1] = asin(as);                                          // pitch
zyx[2] = atan2(2*(y*z + w*x),   w*w - x*x - y*y + z*z);  // roll
```

---

## 3. Control Loop Structure

```
XML timestep: 0.002 s
Policy decimation: kDecimation = 10 (policy fires every 10 physics steps = every 20 ms)
Control rate: 500 Hz (physics), 50 Hz (policy inference)

Per tick:
  backend.readToPackage(io)     // joint pos/vel + IMU → DataPackage
  walk_task.run(io)             // CPG WalkTask; infers policy every 10th tick
  backend.writeFromPackage(io)  // PD formula → d->ctrl[act_id[i]]
  backend.step()                // mj_step
```

PD formula (replicates `MujocoInterface::SetTorque`):
```
ctrl[act_id[i]] = motor_Torque_desire[i]
                + sim_P[i] * (motor_Pos_desire[i] - motor_pos[i])
                + sim_D[i] * (motor_Vel_desire[i] - motor_vel[i])
```

Initialization: `mj_resetDataKeyframe(m, d, 0)` then `mj_forward(m, d)`.

---

## 4. Simulation Results

**3-second walk (3000 ticks × 0.002 s = 6 sim seconds):**

| metric           | value       |
|------------------|-------------|
| initial height   | 0.700 m     |
| min height       | 0.7009 m    |
| final height     | 0.7296 m    |
| x-displacement   | 2.47 m      |
| threshold        | > 0.4 m     |
| result           | **PASS**    |

**5-second extended run (5000 ticks = 10 sim seconds):**

| tick | sim time (s) | height (m) | x (m) |
|------|-------------|------------|-------|
| 0    | 0.002       | 0.7009     | 0.000 |
| 500  | 1.002       | 0.7101     | 0.170 |
| 1000 | 2.002       | 0.7307     | 0.654 |
| 1500 | 3.002       | 0.7272     | 1.118 |
| 2000 | 4.002       | 0.7200     | 1.577 |
| 2500 | 5.002       | 0.7298     | 2.014 |
| 3000 | 6.002       | 0.7297     | 2.475 |
| 3500 | 7.002       | 0.7179     | 2.941 |
| 4000 | 8.002       | 0.7175     | 3.401 |
| 4500 | 9.002       | 0.7308     | 3.843 |
| final| 10.002      | 0.7282     | 4.271 |

Min height: 0.7009 m | Final height: 0.7282 m | x-displacement: 4.27 m | **WALK_OK**

---

## 5. Files Created

| file (relative to `deploy/bdml_phybot_deploy_c2/`) | description |
|-----------------------------------------------------|-------------|
| `core/include/phybot/robot_backend.h`               | Abstract `RobotBackend` interface |
| `backends/mujoco/mujoco_backend.h`                  | `MujocoBackend` declaration |
| `backends/mujoco/mujoco_backend.cpp`                | MuJoCo physics backend implementation |
| `apps/sim_walk.cpp`                                 | Headless sim walk app |
| `core/test/test_walk_sim.cpp`                       | CTest: asserts base height > 0.4 m |
| `CMakeLists.txt`                                    | Updated: added mujoco_backend lib, sim_walk, walk_sim ctest |

---

## 6. Build Output

All targets compiled cleanly with GCC 9.4.0, C++17.

```
[100%] Built target mujoco_backend
[100%] Built target test_walk_sim
[100%] Built target sim_walk
```

---

## 7. CTest Results (all 5 tests)

```
1/5  core_task_manager  PASSED  0.00 s
2/5  walk_numeric       PASSED  0.00 s
3/5  obs_alignment      PASSED  0.01 s
4/5  walk_e2e           PASSED  0.33 s
5/5  walk_sim           PASSED  1.80 s

100% tests passed, 0 tests failed out of 5
```

---

## 8. Key Design Decisions

- **Actuator addressing by name**: Unlike the official `SetTorque` which uses sequential `ctrl[i]`, we use `ctrl[act_id_[i]]` where `act_id_[i]` is resolved by name. This is correct regardless of XML actuator ordering.
- **No noise**: `metorvel_noise_std=0` in official yaml; not applied in sim.
- **MuJoCo library**: linked from `../phybot_deploy_c2/ThirdParty/mujoco/lib/libmujoco.so`; include dir `../phybot_deploy_c2/ThirdParty/mujoco` providing `<mujoco/mujoco.h>`. Marked interim pending vendoring.
- **Timestep**: XML reports 0.002 s; the TICKS constant in the test (3000) gives 6 sim seconds.
