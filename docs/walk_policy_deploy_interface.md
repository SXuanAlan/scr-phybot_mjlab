# Walk Policy Deploy Interface (Mjlab-Walk-Phybot-C2)

字节级契约：未来的 C++ 部署模块（参照 RL_deploy_taichi/RL_deploy_cpg 克隆）按本文构造观测、解释动作。
导出由 `scripts/export_policy.py` 完成，会在模型旁生成 `policy_meta.json`（本文档的机器可读版，
以导出时的实际数值为准）。

## 模型

- 格式：TorchScript（`torch::jit::load` 直接加载），输入 `(1, 720)` float32，输出 `(1, 21)` float32。
- **观测归一化（EmpiricalNormalization）已烘焙在模型内部**。部署侧必须喂**原始物理量**：
  不要乘旧管线的 `ang_vel_scale=0.25`、`dof_vel_scale=0.05`，不要做 `clip_obs` 截断。
- 控制频率 50 Hz（sim dt 0.005 × decimation 4；deploy 侧 decimation 10 @ 500 Hz 不变）。

## 观测布局（720 维）

逐 term 分块（term-major）；每个 term 块内部按时间堆叠 **10 帧、旧→新**，每帧连续存放。
即 `[term1: f(t-9)…f(t)] ++ [term2: f(t-9)…f(t)] ++ …`，不是逐帧交错。

| 区间 | term | 每帧维度 | 内容 / 单位 |
|---|---|---|---|
| [0, 30) | base_ang_vel | 3 | IMU 陀螺仪角速度，机体系，rad/s，**raw** |
| [30, 60) | projected_gravity | 3 | 单位重力方向 (0,0,-1) 旋转到机体系（由姿态四元数计算），模长 1 |
| [60, 270) | joint_pos | 21 | q − default_dof_pos，下表关节顺序，rad |
| [270, 480) | joint_vel | 21 | q̇，rad/s，**raw** |
| [480, 690) | actions | 21 | 上一步策略原始输出（缩放/偏移之前的值） |
| [690, 720) | command | 3 | (vx, vy, ωz)，机体系，m/s 与 rad/s（joystick 直接给物理值） |

**Reset / 上电进入 policy 状态时**：用第一帧观测回填全部 10 个历史槽（mjlab 的语义），
actions 块回填 0。

## 动作解释（21 维）

```
q_target[i] = default_dof_pos[i] + 0.25 * action[i]
```

- `action_scale = 0.25`（注意：旧 RL_deploy_cpg yaml 是 0.2，新模块必须用 0.25）。
- 训练时无动作截断；部署侧建议不截断或只做宽幅安全钳位。
- q_target 进机载 PD：`τ = kp*(q_target − q) − kd*q̇`，再按力矩上限钳位。

## 关节顺序与参数（21 DOF，neck 不归 policy，机上自行 PD 保持）

顺序 = 左腿(6) + 右腿(6) + 腰(1) + 左臂(4) + 右臂(4)，与 RL_deploy_cpg yaml 相同：

| # | joint | default_dof_pos (rad) | kp | kd | τ_max (Nm) |
|---|---|---|---|---|---|
| 0 | left_hip_pitch | -0.242 | 100 | 5 | 200 |
| 1 | left_hip_roll | 0.029 | 100 | 5 | 80 |
| 2 | left_hip_yaw | 0.007 | 100 | 5 | 58 |
| 3 | left_knee | 0.490 | 100 | 5 | 80 |
| 4 | left_ankle_pitch | -0.295 | 50 | 5 | 58 |
| 5 | left_ankle_roll | 0.0 | 50 | 5 | 58 |
| 6 | right_hip_pitch | -0.242 | 100 | 5 | 200 |
| 7 | right_hip_roll | -0.029 | 100 | 5 | 80 |
| 8 | right_hip_yaw | -0.007 | 100 | 5 | 58 |
| 9 | right_knee | 0.490 | 100 | 5 | 80 |
| 10 | right_ankle_pitch | -0.295 | 50 | 5 | 58 |
| 11 | right_ankle_roll | 0.0 | 50 | 5 | 58 |
| 12 | waist_yaw | 0.0 | 20 | 10 | 58 |
| 13 | left_shoulder_pitch | 0.0 | 50 | 5 | 58 |
| 14 | left_shoulder_roll | 0.0 | 50 | 5 | 58 |
| 15 | left_shoulder_yaw | 0.0 | 50 | 5 | 58 |
| 16 | left_elbow_pitch | -0.170 | 50 | 5 | 58 |
| 17 | right_shoulder_pitch | 0.0 | 50 | 5 | 58 |
| 18 | right_shoulder_roll | 0.0 | 50 | 5 | 58 |
| 19 | right_shoulder_yaw | 0.0 | 50 | 5 | 58 |
| 20 | right_elbow_pitch | -0.170 | 50 | 5 | 58 |

default_dof_pos 与真机 ZERO.yaml 的 `zero_joint_position`、RL_deploy_cpg yaml 的
`default_dof_pos` **完全一致**——ZERO → policy 切换零跳变，零位无需任何部署侧改动。
（注：-0.295 下鞋底有 ~1.5° 脚跟下沉，训练里由 feet_flat 奖励处理，刻意保留以对齐真机零位。）

## 手柄 / 命令通道

观测末段 [690, 720) 的 command 由手柄给出，policy 已 conditioned on 该速度目标，
无需额外 goal 输入。真机现有链路（Publisher/Joystick/.../joystick_int.cpp）：

- 左摇杆 Y → vx，量程 ±0.9 m/s；左摇杆 X → vy，量程 ±0.1 m/s；右摇杆 X → ωz，量程 ±0.6 rad/s。
- 死区 5000/32767，加速度限幅 0.4 m/s²（角 0.3 rad/s²），经 LCM 发给 RL 模块。
- ⚠️ 旧 cpg 模块把 ωz 乘了 `ang_vel_scale=0.25` 再进观测；本 policy 的 command 三个分量
  **都喂原始物理值**，不乘任何 scale。
- 训练命令范围 x∈[-1,1]→课程到 [-2,3]、y∈[-1,1]、ωz∈[-0.5,0.5]→[-0.7,0.7]，
  完全覆盖手柄量程；|vx|+|vy|+|ωz| 范数 < 0.05 视为站立命令（训练含 25% 站立采样）。

## 与旧 RL_deploy 模块的差异清单（写新模块时逐条核对）

1. 不乘 obs scales、不做 clip_obs（归一化在模型内）。
2. 观测是 term-major 全历史堆叠（旧 taichi 是逐帧 69 维拼接 + 单独历史 buffer，布局不同）。
3. command 在观测末段且自身也带 10 帧历史。
4. 没有 lin_vel、没有站立 flag、没有 CPG 相位输入。
5. action_scale 0.25（cpg yaml 旧值 0.2）。
6. command 的 ωz 不乘 ang_vel_scale（见上文手柄小节）。
